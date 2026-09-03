import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO

def get_ac_mode(occupancy):
    """
    Maps the stable occupancy count to an AC operating mode.
    
    Occupancy mappings:
      - 0 people: OFF
      - 1-3 people: LOW
      - 4-10 people: MEDIUM
      - 11+ people: HIGH
    """
    if occupancy == 0:
        return "OFF"
    elif 1 <= occupancy <= 3:
        return "LOW"
    elif 4 <= occupancy <= 10:
        return "MEDIUM"
    else:
        return "HIGH"

def main():
    video_path = "test_video.mp4"
    
    # 1. Load the YOLOv8n pretrained model
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    
    # 2. Open test_video.mp4 using OpenCV VideoCapture
    print(f"Opening video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get original video playback properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_delay = 1000.0 / fps if fps > 0 else 33.3
    
    # Define sampling interval: 1 second of video time equals the FPS count
    # e.g., if FPS is 25, we sample every 25 frames
    sample_interval_frames = max(1, int(fps)) if fps > 0 else 25
    
    print(f"Video loaded: {fps:.2f} FPS. Sampling rate: Every {sample_interval_frames} frames (1 second).")
    print("Starting processing. Press 'q' in the video window to quit early.\n")

    # 3. Initialize Rolling Buffer (deque with max length of 5)
    # Using a deque automatically removes the oldest elements when new ones are added
    # past the max limit, making it the perfect choice for a rolling buffer.
    people_buffer = deque(maxlen=5)
    
    # Initial state tracking
    stable_occupancy = 0
    current_ac_mode = "OFF"
    previous_ac_mode = "OFF"
    
    frame_count = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        
        # End of video
        if not ret:
            break
            
        frame_count += 1
        
        # 4. Run inference using final chosen configuration:
        #   - conf=0.25: ensures we catch most occupants
        #   - iou=0.45: suppresses double detections of overlapping/close individuals
        results = model(frame, classes=[0], conf=0.25, iou=0.45, verbose=False)
        
        result = results[0]
        boxes = result.boxes
        raw_people_count = len(boxes)
        
        # Draw bounding boxes and labels for all detected people
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(
                frame, 
                label, 
                (x1, max(15, y1 - 10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1, 
                cv2.LINE_AA
            )
            
        # 5. Occupancy Stabilization: Sample the count every 1 second (or on the first frame)
        if frame_count == 1 or frame_count % sample_interval_frames == 0:
            people_buffer.append(raw_people_count)
            
            # Calculate "stable occupancy" as the mode (most frequent value) in the buffer
            # Counter(people_buffer).most_common(1) returns list of tuples: [(value, frequency)]
            stable_occupancy = Counter(people_buffer).most_common(1)[0][0]
            
            # Map stable occupancy to the AC Mode
            current_ac_mode = get_ac_mode(stable_occupancy)
            
            # Trigger state change logs only when the actual AC mode changes
            if current_ac_mode != previous_ac_mode:
                current_time = time.strftime("%H:%M:%S")
                print(f"[{current_time}] AC Mode changed: {previous_ac_mode} -> {current_ac_mode} (occupancy: {stable_occupancy})")
                previous_ac_mode = current_ac_mode
        
        # 6. Draw Dashboard Text Overlay on the frame
        # Black background panel
        cv2.rectangle(frame, (10, 10), (330, 120), (0, 0, 0), -1)
        
        # Draw counts and AC status
        cv2.putText(frame, f"Raw People Count: {raw_people_count}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Stable Occupancy: {stable_occupancy}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"AC MODE Status:  {current_ac_mode}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the video frame
        cv2.imshow("Smart AC Occupancy Controller", frame)
        
        # Maintain playback speed
        elapsed_time = (time.time() - start_time) * 1000.0
        wait_time = max(1, int(original_delay - elapsed_time))
        
        # Press 'q' to quit early
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            print("\nProcessing stopped early by user.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nProcessing complete. Resources released.")

if __name__ == "__main__":
    main()
