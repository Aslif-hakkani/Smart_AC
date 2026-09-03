import cv2
import time
import os
import tempfile
import statistics
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import torch
from collections import deque, Counter
from ultralytics import YOLO

# ── Hardware detection (done once at import time) ─────────────────────────────
_CUDA_AVAILABLE = torch.cuda.is_available()
_DEVICE          = "cuda" if _CUDA_AVAILABLE else "cpu"
_USE_HALF        = _CUDA_AVAILABLE   # FP16 only makes sense on CUDA

# Set streamlit page config
st.set_page_config(
    page_title="Smart Classroom AC Control",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Module-level shared helpers (used by both Upload and Live modes)
# ─────────────────────────────────────────────────────────────────────────────

# RTSP path templates for common camera brands.
# "Custom RTSP URL" means the user provides the complete URL themselves.
RTSP_PRESETS = {
    "Hikvision":      "/Streaming/Channels/101",
    "Dahua":          "/cam/realmonitor?channel=1&subtype=0",
    "CP Plus":        "/live/ch0",
    "Generic/ONVIF":  "/stream1",
    "Custom RTSP URL": "",
}


def map_ac_mode(occupancy: int) -> str:
    """Map a stable occupancy count to an AC tier string.

    Tiers (only reached when the re-activation lock is cleared):
      0       → OFF
      1–3     → LOW
      4–10    → MEDIUM
      11+     → HIGH
    """
    if occupancy == 0:
        return "OFF"
    elif 1 <= occupancy <= 3:
        return "LOW"
    elif 4 <= occupancy <= 10:
        return "MEDIUM"
    else:
        return "HIGH"


def apply_ac_state_machine(stable_occupancy: int, ac_locked_off: bool):
    """Re-activation threshold state machine.

    A single person after room empties is treated as staff/cleaner, not a new
    class - AC stays off until 2+ people confirm an actual class session.

    Args:
        stable_occupancy: Current stable (mode-filtered) occupancy count.
        ac_locked_off:    Current lock state carried in from the previous tick.

    Returns:
        (ac_mode_str, new_ac_locked_off): Updated AC mode and lock flag.
    """
    if stable_occupancy == 0:
        # Room fully empty → engage lock and force AC off
        return "OFF", True
    elif ac_locked_off:
        # Lock is active: only 2+ people can clear it
        if stable_occupancy >= 2:
            return map_ac_mode(stable_occupancy), False   # Unlock: real class
        else:
            # Single person while locked = staff/cleaner, keep AC off
            return "OFF", True
    else:
        # Normal operation: map occupancy to AC tier
        return map_ac_mode(stable_occupancy), False


def draw_hud(frame, raw_count: int, stable_occupancy: int,
             ac_mode: str, ac_locked_off: bool):
    """Draw the dashboard HUD overlay on a frame (in-place)."""
    cv2.rectangle(frame, (10, 10), (360, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"Raw People Count: {raw_count}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Stable Occupancy: {stable_occupancy}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    ac_color = (0, 255, 0) if not ac_locked_off else (0, 215, 255)   # Green or Amber
    lock_badge = "  [LOCKED]" if ac_locked_off else ""
    cv2.putText(frame, f"AC MODE:  {ac_mode}{lock_badge}",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ac_color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Offline Video Processing Logic
# ─────────────────────────────────────────────────────────────────────────────

def analyze_video(input_path, output_path, conf_threshold, iou_threshold,
                  sample_interval_sec: float = 1.0,
                  device: str = "cpu", use_half: bool = False,
                  progress_bar=None, status_text=None):
    """Process an uploaded video file with YOLOv8 detection and AC state machine.

    Performs YOLOv8 human detection, applies rolling-buffer stabilization,
    maps occupancy to AC modes via the re-activation threshold state machine,
    writes an annotated video to output_path, and collects time-series metrics.

    Args:
        sample_interval_sec: How often (in video seconds) to update the rolling
                             occupancy buffer. Larger = faster processing, lower
                             temporal resolution.
        device:    Torch device string – 'cuda' or 'cpu'.
        use_half:  Enable FP16 half-precision (CUDA only, ignored on CPU).

    Returns:
        dict: Pandas-ready arrays for plotting + occupancy statistics, or None
              on error.
    """
    # yolov8s (small) + imgsz=1280 gives better recall on distant/seated/occluded
    # people in wide-angle CCTV footage compared to the nano/640 default.
    # NOTE: slower per-frame than nano/640; GPU + FP16 largely recovers the speed.
    model = YOLO("yolov8s.pt")  # auto-downloads on first run
    model.to(device)
    if use_half:
        model.half()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        st.error("Error: Could not open video file.")
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps if fps > 0 else 0.0
    _fps_safe              = fps if fps > 0 else 25.0
    sample_interval_frames = max(1, int(_fps_safe * sample_interval_sec))

    # Try H.264 first for browser compatibility, fall back to MPEG-4
    fourcc     = cv2.VideoWriter_fourcc(*"avc1")
    out        = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    codec_used = "avc1 (H.264)"
    if not out.isOpened():
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        out        = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        codec_used = "mp4v (MPEG-4)"

    # State tracking
    people_buffer    = deque(maxlen=5)
    stable_occupancy = 0
    current_ac_mode  = "OFF"
    previous_ac_mode = "OFF"

    # Re-activation threshold state machine:
    # A single person after room empties is treated as staff/cleaner, not a new
    # class - AC stays off until 2+ people confirm an actual class session.
    ac_locked_off  = True   # Starts locked; requires 2+ people to unlock
    ac_mode_changes = 0

    # Data collection for charts
    timestamps       = []
    raw_counts       = []
    stable_counts    = []
    ac_modes_history = []

    frame_count   = 0
    _proc_t0      = time.time()   # wall-clock start for ETA calculation

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count    += 1
        current_time_sec = frame_count / fps if fps > 0 else frame_count / 25.0

        # YOLOv8 inference (persons only)
        results         = model(frame, classes=[0], conf=conf_threshold,
                                iou=iou_threshold, imgsz=1280, verbose=False)
        boxes           = results[0].boxes
        raw_people_count = len(boxes)

        # Draw bounding boxes
        for box in boxes:
            x1, y1, x2, y2  = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person: {confidence:.2f}",
                        (x1, max(15, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Sample every 1 second of video time to update the rolling buffer
        if frame_count == 1 or frame_count % sample_interval_frames == 0:
            people_buffer.append(raw_people_count)
            stable_occupancy = Counter(people_buffer).most_common(1)[0][0]

            # Apply shared re-activation state machine
            current_ac_mode, ac_locked_off = apply_ac_state_machine(
                stable_occupancy, ac_locked_off
            )

            if current_ac_mode != previous_ac_mode:
                ac_mode_changes += 1
                previous_ac_mode = current_ac_mode

        # Record data point
        timestamps.append(current_time_sec)
        raw_counts.append(raw_people_count)
        stable_counts.append(stable_occupancy)
        ac_modes_history.append(current_ac_mode)

        # Draw HUD overlay
        draw_hud(frame, raw_people_count, stable_occupancy,
                 current_ac_mode, ac_locked_off)

        out.write(frame)

        if progress_bar and frame_count % 10 == 0:
            pct = min(1.0, frame_count / total_frames)
            progress_bar.progress(pct)
            if status_text:
                elapsed_proc = time.time() - _proc_t0
                if pct > 0:
                    eta_sec = elapsed_proc / pct * (1.0 - pct)
                    eta_str = f"{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
                else:
                    eta_str = "calculating…"
                status_text.text(
                    f"Processing frame {frame_count} / {total_frames} "
                    f"({pct:.0%}) · ETA {eta_str} · "
                    f"Device: {device.upper()}"
                )

    cap.release()
    out.release()

    avg_occ  = statistics.mean(raw_counts) if raw_counts else 0.0
    peak_occ = max(raw_counts)             if raw_counts else 0

    return {
        "timestamps":    timestamps,
        "raw_counts":    raw_counts,
        "stable_counts": stable_counts,
        "ac_modes":      ac_modes_history,
        "ac_changes":    ac_mode_changes,
        "duration":      duration,
        "avg_occupancy": avg_occ,
        "peak_occupancy": peak_occ,
        "codec_used":    codec_used,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Helper: build RTSP URL from components
# ─────────────────────────────────────────────────────────────────────────────

def build_rtsp_url(brand: str, ip: str, port: int, user: str, password: str,
                   custom_url: str) -> str:
    """Construct a complete RTSP URL from panel inputs."""
    if brand == "Custom RTSP URL":
        return custom_url.strip()
    path = RTSP_PRESETS.get(brand, "/stream1")
    if user and password:
        return f"rtsp://{user}:{password}@{ip}:{port}{path}"
    return f"rtsp://{ip}:{port}{path}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper: render analytics charts (reused in upload results)
# ─────────────────────────────────────────────────────────────────────────────

def render_analytics_charts(df):
    """Render occupancy and AC mode timeline Plotly charts."""
    # Occupancy chart
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(
        x=df["Time (s)"], y=df["Raw Occupancy"],
        mode="lines", name="Raw Detections",
        line=dict(color="#FF4B4B", width=1.5, dash="dot"), opacity=0.6
    ))
    fig_occ.add_trace(go.Scatter(
        x=df["Time (s)"], y=df["Stable Occupancy (Mode)"],
        mode="lines", name="Stable Occupancy (1s Mode)",
        line=dict(color="#00D2B4", width=3)
    ))
    fig_occ.update_layout(
        title="Student Occupancy Over Time",
        xaxis_title="Time (seconds)", yaxis_title="Occupants (count)",
        template="plotly_dark", hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_occ, width='stretch')

    # AC mode timeline chart
    mode_mapping = {"OFF": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    df["AC Mode Value"] = df["AC Mode"].map(mode_mapping)

    fig_ac = go.Figure()
    fig_ac.add_trace(go.Scatter(
        x=df["Time (s)"], y=df["AC Mode Value"],
        mode="lines", line_shape="hv", name="AC Output Mode",
        line=dict(color="#FFD166", width=2.5),
        fill="tozeroy", fillcolor="rgba(255, 209, 102, 0.15)"
    ))
    fig_ac.update_layout(
        title="Smart AC Operating Mode Timeline",
        xaxis_title="Time (seconds)",
        yaxis=dict(
            title="AC Level", tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["OFF / LOCKED (0-1)", "LOW (2-3)", "MEDIUM (4-10)", "HIGH (11+)"]
        ),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_ac, width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# 4. Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🌡️ Smart Classroom AC Control")
st.markdown(
    "Monitor classroom occupancy with **YOLOv8** detection and control the AC "
    "automatically using the **re-activation threshold state machine** — "
    "preventing false wake-ups from cleaning staff."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Detection Settings")
st.sidebar.markdown("Tune the computer-vision thresholds below:")

conf_val = st.sidebar.slider(
    "Confidence Threshold", min_value=0.10, max_value=0.60, value=0.25, step=0.05,
    help="Minimum model confidence to count a person. Lower = more detections but more noise."
)
iou_val = st.sidebar.slider(
    "IoU Threshold", min_value=0.20, max_value=0.80, value=0.45, step=0.05,
    help="NMS overlap threshold. Lower = fewer overlapping boxes."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Performance")

# ── GPU / CPU detection banner ────────────────────────────────────────────────
if _CUDA_AVAILABLE:
    _gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"🚀 Running on: **GPU (CUDA)**  \n{_gpu_name}  \nFP16 half-precision enabled.")
else:
    st.sidebar.warning(
        "⚠️ No GPU detected — running on **CPU**.  \n"
        "Processing will be slower. Consider increasing the sampling interval below "
        "to reduce the number of frames that need inference."
    )

# ── Frame sampling interval slider ───────────────────────────────────────────
sample_interval_val = st.sidebar.select_slider(
    "Frame Sampling Interval (s)",
    options=[0.5, 1.0, 1.5, 2.0],
    value=1.0,
    help=(
        "How often (in video-seconds) the occupancy buffer is updated.  \n"
        "• 0.5 s → highest temporal resolution, slowest processing.  \n"
        "• 2.0 s → fastest processing, coarser occupancy tracking."
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**AC Mode tiers** (after 2+ people unlock)\n"
    "- 🔵 LOW: 1–3 people\n"
    "- 🟠 MEDIUM: 4–10 people\n"
    "- 🔴 HIGH: 11+ people\n"
    "- ⚫ OFF / LOCKED: 0–1 person"
)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_upload, tab_live = st.tabs(["📁  Upload Video", "📷  Live CCTV Feed"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload Video (existing behaviour, unchanged)
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### 📤 Upload Classroom Video")
    st.markdown(
        "Upload a pre-recorded classroom video. The system will process every frame "
        "offline and produce an annotated video with occupancy + AC mode overlays."
    )

    uploaded_file = st.file_uploader("Select a video file (.mp4)", type=["mp4"])

    if uploaded_file is not None:
        temp_dir         = tempfile.mkdtemp()
        temp_input_path  = os.path.join(temp_dir, "input_video.mp4")
        temp_output_path = os.path.join(temp_dir, "annotated_video.mp4")

        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.read())

        st.success("✅ File uploaded successfully! Ready for processing.")

        if st.button("🚀 Analyze Occupancy & AC Controls", type="primary",
                     key="btn_analyze"):
            progress_bar = st.progress(0.0)
            status_text  = st.empty()
            t0           = time.time()

            with st.spinner("Analyzing frames…"):
                metrics = analyze_video(
                    input_path=temp_input_path,
                    output_path=temp_output_path,
                    conf_threshold=conf_val,
                    iou_threshold=iou_val,
                    sample_interval_sec=sample_interval_val,
                    device=_DEVICE,
                    use_half=_USE_HALF,
                    progress_bar=progress_bar,
                    status_text=status_text,
                )

            elapsed = time.time() - t0
            progress_bar.progress(1.0)
            status_text.text(f"✅ Analysis complete in {elapsed:.1f} s!")

            if metrics is not None:
                st.markdown("---")
                st.subheader("📊 Room Occupancy & AC Status Summary")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg Occupancy",  f"{metrics['avg_occupancy']:.1f} people")
                c2.metric("Peak Occupancy", f"{metrics['peak_occupancy']} people")
                c3.metric("AC Mode Changes", f"{metrics['ac_changes']} changes")
                c4.metric("Video Duration",  f"{metrics['duration']:.1f} s")

                df = pd.DataFrame({
                    "Time (s)":               metrics["timestamps"],
                    "Raw Occupancy":          metrics["raw_counts"],
                    "Stable Occupancy (Mode)": metrics["stable_counts"],
                    "AC Mode":                metrics["ac_modes"],
                })

                chart_col, video_col = st.columns([3, 2])

                with chart_col:
                    st.markdown("### 📈 Time-series Analytics")
                    render_analytics_charts(df)

                with video_col:
                    st.markdown("### 🎥 Processed Video Output")
                    if "mp4v" in metrics["codec_used"]:
                        st.info(
                            "ℹ️ Written using MPEG-4. If the player below doesn't load, "
                            "download the file to view locally."
                        )
                    else:
                        st.success("Web-compatible H.264 (avc1) encoding successful!")

                    if os.path.exists(temp_output_path):
                        with open(temp_output_path, "rb") as vf:
                            video_bytes = vf.read()
                        st.video(video_bytes)
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=video_bytes,
                            file_name="annotated_occupancy.mp4",
                            mime="video/mp4",
                            type="secondary",
                            key="btn_download_video",
                        )
                    else:
                        st.error("Output video file was not found.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Live CCTV Feed
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### 📷 Live CCTV Feed — RTSP Connection")
    st.markdown(
        "Connect to any IP camera over RTSP. Select your camera brand to "
        "auto-fill the stream path, or enter a fully custom URL."
    )

    # ── Initialise session state keys ────────────────────────────────────────
    _ss_defaults = {
        "rtsp_brand":       "Hikvision",
        "rtsp_ip":          "192.168.1.64",
        "rtsp_port":        554,
        "rtsp_user":        "admin",
        "rtsp_pass":        "",
        "rtsp_custom_url":  "",
        "rtsp_connected":   False,
        "live_running":     False,
    }
    for k, v in _ss_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Camera Brand Preset ───────────────────────────────────────────────────
    st.markdown("#### 📡 Camera Connection Settings")

    brand_options = list(RTSP_PRESETS.keys())
    selected_brand = st.selectbox(
    "Camera Brand Preset",
    options=brand_options,
    key="rtsp_brand",
    help="Select your camera brand to auto-fill the RTSP stream path.",
    )

    is_custom = selected_brand == "Custom RTSP URL"

    if is_custom:
        # Custom mode: user pastes a full RTSP URL
        custom_url = st.text_input(
        "Full RTSP URL",
        placeholder="rtsp://user:pass@192.168.1.x:554/stream1",
        key="rtsp_custom_url",
        )

        final_rtsp_url = custom_url.strip()
    else:
        # Standard mode: assemble URL from components
        path_template = RTSP_PRESETS[selected_brand]

        col_ip, col_port, col_user, col_pass = st.columns([3, 1, 2, 2])
        with col_ip:
            ip = st.text_input("IP Address", key="rtsp_ip", placeholder="192.168.1.64")

        with col_port:
            port = st.number_input("Port", min_value=1, max_value=65535,
                                    key="rtsp_port")
        with col_user:
           user = st.text_input("Username", key="rtsp_user")

        with col_pass:
           password = st.text_input("Password", key="rtsp_pass", type="password")

        final_rtsp_url = build_rtsp_url(
            selected_brand, ip, int(port), user, password, ""
        )

        # Preview of the constructed URL (password redacted)
        safe_url = build_rtsp_url(selected_brand, ip, int(port), user,
                                  "●●●●" if password else "", "")
        st.markdown(
            f"**Constructed RTSP URL:** `{safe_url}`  \n"
            f"*Stream path for {selected_brand}:* `{path_template}`"
        )

    st.markdown("---")

    # ── Test Connection ───────────────────────────────────────────────────────
    col_test, col_start, col_stop = st.columns([2, 2, 1])

    with col_test:
        if st.button("🔌 Test Connection", key="btn_test_rtsp"):
            if not final_rtsp_url:
                st.error("⚠️ Please enter a valid RTSP URL or fill in all fields.")
            else:
                with st.spinner(f"Connecting to `{final_rtsp_url}` …"):
                    test_cap = cv2.VideoCapture()
                    # 4-second open timeout so the UI doesn't hang on bad IPs
                    test_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000)
                    test_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 4000)
                    test_cap.open(final_rtsp_url, cv2.CAP_FFMPEG)

                    if test_cap.isOpened():
                        ret, test_frame = test_cap.read()
                        test_cap.release()
                        if ret and test_frame is not None:
                            st.session_state["rtsp_connected"] = True
                            st.success(
                                f"✅ Connection successful! "
                                f"Frame size: {test_frame.shape[1]}×{test_frame.shape[0]} px"
                            )
                            # Show the test frame as a preview
                            preview_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                            st.image(preview_rgb, caption="📸 Live camera preview frame",
                                     width='stretch')
                        else:
                            st.session_state["rtsp_connected"] = False
                            st.error(
                                "⚠️ Camera opened but could not read a frame. "
                                "Check stream path and sub-stream settings."
                            )
                    else:
                        test_cap.release()
                        st.session_state["rtsp_connected"] = False
                        st.error(
                            "❌ Could not connect to the camera. "
                            "Verify the IP address, port, credentials, and that the "
                            "camera is reachable on this network."
                        )

    with col_start:
        start_disabled = not st.session_state.get("rtsp_connected", False)
        if st.button("▶ Start Live Analysis", key="btn_start_live",
                     disabled=start_disabled,
                     type="primary",
                     help="Test the connection first to enable this button."):
            st.session_state["live_running"] = True

    with col_stop:
        if st.button("⏹ Stop", key="btn_stop_live"):
            st.session_state["live_running"] = False

    # ── Live streaming loop ───────────────────────────────────────────────────
    if st.session_state.get("live_running", False):
        st.markdown("---")
        st.subheader("🔴 Live Feed — Real-time Occupancy Analysis")

        # Metric KPI badges (refreshed each frame batch)
        kpi_raw, kpi_stable, kpi_mode, kpi_lock = st.columns(4)
        raw_badge    = kpi_raw.empty()
        stable_badge = kpi_stable.empty()
        mode_badge   = kpi_mode.empty()
        lock_badge_ph = kpi_lock.empty()

        # Frame display placeholder
        frame_placeholder = st.empty()

        status_live = st.empty()
        status_live.info("🔴 Live analysis running… click **Stop** to end.")

        # Initialise detection model + state
        # yolov8s + imgsz=1280: better recall on small/seated/occluded people;
        # slower than nano/640. GPU + FP16 largely recovers the speed; on CPU
        # increase the sampling interval via the sidebar slider if FPS drops.
        live_model = YOLO("yolov8s.pt")  # auto-downloads on first run
        live_model.to(_DEVICE)
        if _USE_HALF:
            live_model.half()

        live_cap = cv2.VideoCapture()
        live_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 6000)
        live_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 6000)
        # Request a low buffer size so we always get the latest frame
        live_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        live_cap.open(final_rtsp_url, cv2.CAP_FFMPEG)

        if not live_cap.isOpened():
            st.error(
                "❌ Lost connection to the camera when starting live analysis. "
                "Please re-run Test Connection."
            )
            st.session_state["live_running"] = False
        else:
            live_people_buffer    = deque(maxlen=5)
            live_stable_occupancy = 0
            live_ac_mode          = "OFF"
            live_ac_locked_off    = True   # Re-activation lock starts engaged

            live_frame_count     = 0
            live_fps_estimate    = live_cap.get(cv2.CAP_PROP_FPS) or 25.0
            # Use the sidebar-configured sampling interval converted to frames
            live_sample_interval = max(1, int(live_fps_estimate * sample_interval_val))

            try:
                while st.session_state.get("live_running", False):
                    ret, live_frame = live_cap.read()
                    if not ret:
                        status_live.warning(
                            "⚠️ Frame read failed — camera may have dropped the connection."
                        )
                        break

                    live_frame_count += 1

                    # YOLOv8 inference
                    results          = live_model(
                        live_frame, classes=[0], conf=conf_val,
                        iou=iou_val, imgsz=1280, verbose=False
                    )
                    boxes            = results[0].boxes
                    live_raw_count   = len(boxes)

                    # Draw bounding boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf_score      = float(box.conf[0])
                        cv2.rectangle(live_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            live_frame, f"Person: {conf_score:.2f}",
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )

                    # Update rolling buffer and AC state every ~1 second of footage
                    if (live_frame_count == 1
                            or live_frame_count % live_sample_interval == 0):
                        live_people_buffer.append(live_raw_count)
                        live_stable_occupancy = Counter(
                            live_people_buffer
                        ).most_common(1)[0][0]

                        live_ac_mode, live_ac_locked_off = apply_ac_state_machine(
                            live_stable_occupancy, live_ac_locked_off
                        )

                        # Refresh KPI badges
                        raw_badge.metric("Raw Count",       live_raw_count)
                        stable_badge.metric("Stable Count", live_stable_occupancy)
                        mode_badge.metric("AC Mode",        live_ac_mode)
                        lock_badge_ph.metric(
                            "Lock State",
                            "🔒 LOCKED" if live_ac_locked_off else "🔓 Active"
                        )

                    # Draw HUD overlay and push frame to browser
                    draw_hud(live_frame, live_raw_count, live_stable_occupancy,
                             live_ac_mode, live_ac_locked_off)

                    frame_rgb = cv2.cvtColor(live_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        frame_rgb, channels="RGB",
                        width='stretch',
                        caption=(
                            f"Frame {live_frame_count} | "
                            f"Raw: {live_raw_count} | "
                            f"Stable: {live_stable_occupancy} | "
                            f"AC: {live_ac_mode}"
                            + (" [LOCKED]" if live_ac_locked_off else "")
                        )
                    )

            finally:
                live_cap.release()
                st.session_state["live_running"] = False
                status_live.success("⏹ Live analysis stopped.")

    elif not st.session_state.get("rtsp_connected", False):
        st.info(
            "👆 Fill in your camera details and click **Test Connection** to verify "
            "the stream before starting live analysis."
        )
    else:
        st.info(
            "✅ Camera connected. Click **▶ Start Live Analysis** to begin real-time "
            "occupancy monitoring."
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Smart Classroom AC Control · Built with **YOLOv8** & **Streamlit** · "
    "Re-activation lock prevents false AC wake-ups from cleaning staff."
)