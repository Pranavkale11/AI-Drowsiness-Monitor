# 🚗 Driver Drowsiness & Alert Monitoring System

A real-time computer vision system designed to detect driver drowsiness and fatigue using facial landmark detection and intelligent monitoring techniques.

---

## 📌 Overview

This project uses **OpenCV, dlib, and Streamlit** to monitor a driver’s eye movements and detect signs of drowsiness in real time.
It provides live feedback, fatigue scoring, and alert mechanisms to improve road safety.

---

## 🎯 Features

* 🎥 **Real-Time Camera Monitoring**
* 👁️ **Eye Aspect Ratio (EAR) Detection**
* 😴 **Drowsiness & Sleep Detection**
* 📊 **Live Fatigue Score & Analytics**
* 📈 **Real-Time EAR Graph (Plotly)**
* 🔔 **Alert System (Wake-Up Warning)**
* ⚡ **Performance Modes (Optimized for smooth execution)**
* 🧵 **Multi-threaded Camera Processing**
* 💾 **Session Logging (CSV support)**

---

## 🛠️ Tech Stack

* **Python**
* **OpenCV**
* **dlib**
* **Streamlit**
* **Plotly**
* **NumPy**

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Pranavkale11/AI-Drowsiness-Monitor.git
cd AI-Drowsiness-Monitor
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
pip install dlib-bin
```

---

### 3️⃣ Download Required Model File

Download the dlib facial landmark model:

👉 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

* Extract the file
* Place it in the project root directory

---

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. Captures live video using webcam
2. Detects face and facial landmarks (68 points)
3. Calculates Eye Aspect Ratio (EAR)
4. Determines driver state:

   * 🟢 Active
   * 🟡 Drowsy
   * 🔴 Sleeping
5. Triggers alert system if drowsiness detected

---

## 📊 Performance Optimizations

* ⚡ Frame skipping for faster processing
* 🧵 Multi-threaded camera handling
* 📉 Reduced resolution for efficiency
* 🔄 Throttled UI updates
* 📈 Smoothed FPS calculation

---

## ⚠️ Limitations

* Performance may vary in low lighting
* Glasses/sunglasses can affect detection accuracy
* Uses rule-based detection (not fully trained ML model)

---

## 🚀 Future Improvements

* Deep learning-based drowsiness detection
* Mobile notifications / IoT integration
* Cloud logging & analytics
* Night vision enhancement

---

## 👨‍💻 Author

**Pranav Kale**
GitHub: https://github.com/Pranavkale11

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

---
