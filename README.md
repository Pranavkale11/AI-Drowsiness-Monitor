

---



# 🚗 AI-Based Driver Drowsiness & Alert Monitoring System

**A real-time computer vision system for detecting driver drowsiness and fatigue using facial landmark detection and intelligent monitoring.**

---

## 📌 **About the Project**

This project leverages **OpenCV**, **dlib**, and **Streamlit** to monitor a driver’s eye movements and detect signs of drowsiness in real time. It provides **live feedback**, **fatigue scoring**, and **alert mechanisms** to enhance road safety and prevent accidents caused by driver fatigue.

---

## ✨ **Key Features**

- **🎥 Real-Time Camera Monitoring**: Continuous video feed from your webcam.
- **👁️ Eye Aspect Ratio (EAR) Detection**: Accurate calculation of eye openness to detect drowsiness.
- **😴 Drowsiness & Sleep Detection**: Identifies signs of fatigue and sleepiness.
- **📊 Live Fatigue Score & Analytics**: Visual representation of fatigue levels.
- **📈 Real-Time EAR Graph**: Interactive graph using **Plotly** for monitoring eye aspect ratio.
- **🔔 Alert System**: Audible and visual warnings to wake up the driver.
- **⚡ Performance Modes**: Optimized for smooth and efficient execution.
- **🧵 Multi-threaded Camera Processing**: Improved performance with concurrent processing.
- **💾 Session Logging**: Logs data to **CSV** for further analysis.

---

## 🛠️ **Tech Stack**

- **Language**: Python
- **Computer Vision**: OpenCV, dlib
- **Web Framework**: Streamlit
- **Data Visualization**: Plotly, Matplotlib
- **Numerical Computing**: NumPy

---

## ⚙️ **Installation & Setup**

### **Prerequisites**
- Python 3.8+
- pip (Python package manager)

### **Steps**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Pranavkale11/AI-Drowsiness-Monitor.git
   cd AI-Drowsiness-Monitor
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install dlib-bin
   ```

3. **Download the Facial Landmark Model**
   - Download the **dlib facial landmark model** from:
     [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract the file and place it in the **project root directory** as `shape_predictor_68_face_landmarks.dat`.

4. **Run the Application**
   ```bash
   streamlit run app.py
   ```

---

## 🧠 **How It Works**

1. **Video Capture**: The system captures live video using your webcam.
2. **Face & Landmark Detection**: Detects the face and **68 facial landmarks** using dlib.
3. **Eye Aspect Ratio (EAR) Calculation**: Computes the EAR to determine eye openness.
4. **Driver State Classification**:
   - **🟢 Active**: Driver is alert and attentive.
   - **🟡 Drowsy**: Signs of fatigue detected.
   - **🔴 Sleeping**: Driver is asleep or highly fatigued.
5. **Alert System**: Triggers **audible and visual alerts** if drowsiness is detected.

---

## 📊 **Performance Optimizations**

- **⚡ Frame Skipping**: Reduces processing load for faster performance.
- **🧵 Multi-threaded Camera Handling**: Ensures smooth video processing.
- **📉 Reduced Resolution**: Optimizes for efficiency without compromising accuracy.
- **🔄 Throttled UI Updates**: Prevents UI lag during real-time monitoring.
- **📈 Smoothed FPS Calculation**: Provides stable frame rate metrics.

---

## ⚠️ **Limitations**

- **Lighting Conditions**: Performance may vary in low-light environments.
- **Accessories**: Glasses or sunglasses can affect detection accuracy.
- **Detection Method**: Uses **rule-based detection** (not a fully trained ML model).

---

## 🚀 **Future Improvements**

- **Deep Learning Integration**: Implement **CNN-based drowsiness detection** for higher accuracy.
- **Mobile Notifications**: Send alerts to the driver’s phone via **IoT integration**.
- **Cloud Logging**: Store and analyze data in the cloud for long-term insights.
- **Night Vision Enhancement**: Improve detection in low-light conditions.

---

## 👨‍💻 **Author**

**Pranav Kale**
🔗 [GitHub](https://github.com/Pranavkale11)

---

## ⭐ **Support & Contributions**

If you found this project useful, consider:
- **⭐ Star the repository** on GitHub.
- **🐛 Report issues** or suggest improvements via [GitHub Issues](https://github.com/Pranavkale11/AI-Drowsiness-Monitor/issues).
- **🤝 Contribute** by submitting a pull request!

---
</canvaentity
>
