🛡️ GVPCEW Hostel Surveillance System (Fast Version)

A real-time AI-based face recognition surveillance system designed for hostel monitoring using Haar Cascade + ArcFace hybrid model.

This system tracks student movement and automatically logs:

✅ Entry (IN) → when a person appears

🚪 Exit (OUT) → when a person disappears

📊 Attendance → stored in Excel (session + master logs)

🚀 Features

⚡ Fast Detection

Haar Cascade runs every frame (ultra-fast)

ArcFace runs only once per new face

🧠 Accurate Recognition

Uses ArcFace embeddings for identity matching

Cosine similarity-based recognition

📹 Live Monitoring UI (Streamlit)

Real-time video feed

Status dashboard (IN / OUT / Inside)

Event tracking

📊 Excel Logging

Session-wise logs (timestamped file)

Master log (all sessions combined)

🔁 Smart Tracking

Prevents duplicate entries using cooldown

Handles temporary face disappearance

Tracks using centroid + identity cache

🏗️ Project Structure
CCTV_PROJECT/
│── app_fast.py                # Streamlit UI (main app)
│── cctv_engine.py            # Core detection & tracking engine
│── arcface_embeddings.pkl    # Face embeddings database
│── CCTV project dataset.xlsx # Student dataset (name, roll)
│── attendance_master.xlsx    # Master attendance log
│── session_*.xlsx            # Session logs (auto-generated)
│── .gitignore
⚙️ Tech Stack

Frontend/UI: Streamlit

Computer Vision: OpenCV

Face Recognition: InsightFace (ArcFace)

Tracking: Custom centroid tracker

Data Storage: Excel (OpenPyXL)

Language: Python

🧠 How It Works
🔹 Detection Pipeline

Haar Cascade detects faces every frame

Filters applied:

Face size

Aspect ratio

Brightness

Texture variance

Skin tone

Valid faces → sent to ArcFace (once)

🔹 Recognition

Extract embedding using ArcFace

Compare with stored embeddings

If similarity > threshold → identified

Else → Unknown (ignored)

🔹 Tracking Logic

Each face gets a temporary track ID

Once identified:

Stored in person_state

Never reprocessed again in session

🔹 IN / OUT Logic

👤 Appears in frame → IN logged

🚪 Disappears for few seconds → OUT logged

📊 Output (Excel)

Each entry contains:

S.No	Name	Roll No	Date	Time In	Time Out	Duration
🛠️ Installation
git clone https://github.com/your-username/CCTV_PROJECT.git
cd CCTV_PROJECT
pip install -r requirements.txt
▶️ Run the Project
streamlit run app_fast.py
📁 Requirements

Create requirements.txt:

streamlit
opencv-python
numpy
openpyxl
insightface
onnxruntime
📷 Usage

Upload arcface_embeddings.pkl

Select camera / RTSP stream

Click START MONITORING

System logs IN/OUT automatically

Click STOP & SAVE

Download Excel reports

⚙️ Configuration
Parameter	Description
Threshold	Face match strictness
Cooldown	Prevent duplicate logging
Camera	Webcam or RTSP
Filters	Improve detection accuracy
🧪 Example Use Cases

🏫 Hostel entry monitoring

🏢 Office attendance tracking

🎓 Campus security systems

🚪 Smart entry/exit logging

⚡ Performance Optimizations

Detection ≠ Recognition (decoupled)

Background thread for ArcFace

Small queue → fresh frames only

Identity caching → no reprocessing

Haar instead of heavy detectors → smooth video

❗ Limitations

Requires good lighting conditions

Accuracy depends on embedding quality

Haar may miss extreme angles

No mask detection
