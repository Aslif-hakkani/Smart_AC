\# Smart Classroom AC Control System



A computer-vision-based system that automatically detects classroom occupancy from CCTV footage and adjusts air-conditioning intensity accordingly — reducing energy waste while keeping classrooms comfortable.



\## 🔧 How It Works



1\. \*\*Detection\*\* — YOLOv8 detects and counts people in video/CCTV frames

2\. \*\*Stabilization\*\* — A rolling-buffer smoothing layer filters frame-to-frame detection noise

3\. \*\*Decision Logic\*\* — Stable occupancy maps to 4 AC tiers: OFF (0) → LOW (1-3) → MEDIUM (4-10) → HIGH (11+)

4\. \*\*Dashboard\*\* — Streamlit interface visualizes live occupancy, AC mode timeline, and analytics



\## 🛠️ Tech Stack



\- \*\*Language:\*\* Python 3.12

\- \*\*Detection Model:\*\* YOLOv8 (Ultralytics)

\- \*\*Computer Vision:\*\* OpenCV

\- \*\*Deep Learning:\*\* PyTorch (CUDA-accelerated)

\- \*\*Dashboard:\*\* Streamlit + Plotly

\- \*\*Hardware (planned):\*\* ESP32 + IR blaster for physical AC control



\## ✨ Features



\- Upload pre-recorded classroom video for offline analysis

\- Live CCTV feed integration over RTSP (brand-agnostic — Hikvision, Dahua, CP Plus, and other RTSP cameras)

\- Adjustable confidence/IoU detection thresholds

\- GPU/CPU auto-detection with FP16 acceleration

\- Configurable frame sampling interval

\- Occupancy-over-time and AC mode timeline charts

\- Downloadable annotated video output



\## 🚀 Running Locally



\\`\\`\\`bash

pip install -r requirements.txt

streamlit run dashboard.py

\\`\\`\\`



\## 📋 Current Limitations



\- Physical AC hardware control (ESP32 + IR) not yet implemented

\- Cannot distinguish between a lone student and cleaning staff based on appearance alone

\- Live CCTV tested on personal camera; university classroom deployment pending IT access



\## 🔮 Future Scope



\- ESP32 + IR blaster integration for real AC control

\- Movement-pattern or object-detection-based cleaner/student distinction

\- Historical occupancy logging and multi-classroom support



\## 👤 Author



\*\*Aslif Hakkani\*\*

Physical Science Undergraduate, University of Jaffna

\[LinkedIn](https://www.linkedin.com/in/aslif-hakkani/) · \[GitHub](https://github.com/Aslif-hakkani)

