# GVPCEW Hostel Surveillance System

> Real-time AI-powered face recognition system for monitoring student entry and exit.

---

## Overview

This project is a prototype for a **real-time surveillance system** that uses **Computer Vision + Deep Learning** to:

- Detect faces from live camera feed  
- Identify registered students  
- Detect unknown individuals  
- Automatically log entry (IN) and exit (OUT)  
- Generate attendance reports in Excel  

---

## Key Highlights

-  Fast detection using Haar Cascade  
-  Accurate recognition using ArcFace  
-  Handles both known and unknown faces  
-  Smart tracking to avoid duplicate entries  
-  Automatic Excel logging (session + master)  
-  Interactive dashboard using Streamlit  

---

## How It Works

### Step-by-step pipeline:

1. Capture video from webcam / RTSP  
2. Detect faces using OpenCV (Haar Cascade)  
3. Extract embeddings using ArcFace  
4. Compare with stored embeddings  
   - Match → Identified  
   - No match → Unknown  
5. Logging logic:
   - Appears → IN  
   - Disappears → OUT  
6. Save data into Excel automatically  

---

## ✨ Features

-  Real-time face detection  
-  Face recognition (known people)  
-  Unknown face detection  
-  Attendance tracking system  
-  Excel report generation  
-  Cooldown to prevent duplicate logging  
-  Identity caching (no reprocessing)  
-  Live dashboard (Streamlit UI)  

---

## 🛠️ Tech Stack

| Category         | Technology             |
|------------------|------------------------|
| Language         | Python                 |
| Computer Vision  | OpenCV                 |
| Face Recognition | InsightFace (ArcFace)  |
| UI               | Streamlit              |
| Data Storage     | Excel (OpenPyXL)       |

---

## 📂 Project Structure

CCTV_PROJECT/
│── app_fast.py                  # Main Streamlit application  
│── cctv_engine.py              # Core detection & tracking logic  
│── arcface_embeddings.pkl      # Face embeddings database  
│── CCTV project dataset.xlsx   # Student dataset (name, roll)  
│── attendance_master.xlsx      # Master attendance log  
│── session_*.xlsx              # Session logs (auto-generated)  
│── .gitignore  

---

## ⚙️ Installation

git clone https://github.com/your-username/CCTV_PROJECT.git  
cd CCTV_PROJECT  
pip install -r requirements.txt  

---

##  Requirements

Create a `requirements.txt` file with:

streamlit  
opencv-python  
numpy  
openpyxl  
insightface  
onnxruntime  

---

##  Run the Project

streamlit run app_fast.py  

---

##  Usage

1. Upload `arcface_embeddings.pkl`  
2. Select camera or enter RTSP URL  
3. Click **START MONITORING**  
4. System will:
   - Identify known faces  
   - Detect unknown faces  
   - Log IN/OUT automatically  
5. Click **STOP & SAVE**  
6. Download Excel reports  

---

##  Output

Each record contains:

- Name  
- Roll Number  
- Date  
- Time In  
- Time Out  
- Duration  

---

##  Use Cases

-  Hostel entry monitoring  
-  College attendance system  
-  Office employee tracking  
-  Security surveillance systems  

---

##  Performance Optimizations

- Detection ≠ Recognition (decoupled system)  
- ArcFace runs only once per new face  
- Background thread for recognition  
- Identity caching (no repeated processing)  
- Haar Cascade → smooth video performance  

---

##  Limitations

- Requires good lighting conditions  
- Accuracy depends on embedding quality  
- Haar Cascade may miss extreme angles  
- Unknown faces are detected but not stored  

---

##  Future Improvements

- Replace Haar with RetinaFace (better accuracy)  
- Add database (MongoDB / Firebase)  
- Deploy as web app (React + FastAPI)  
- Multi-camera support  
- Mobile notifications  

---

## 💡 For Recruiters

This project demonstrates:

- It's just a prototype for a Real-time computer vision  
- Face recognition using deep learning  
- System design (tracking + logging)  
- End-to-end ML application with UI  

---
