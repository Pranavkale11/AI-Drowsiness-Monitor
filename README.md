# 🚗 Driver Drowsiness & Alert Monitoring System

A real-time computer vision system that detects driver drowsiness and fatigue using facial landmark detection, giving live feedback and alerts to help improve road safety.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-green?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#️-tech-stack)
- [Installation & Setup](#️-installation--setup)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Performance Optimizations](#-performance-optimizations)
- [Limitations](#️-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)
- [Support](#-support)

---

## 📌 Overview

This project uses **OpenCV**, **dlib**, and **Streamlit** to monitor a driver's eye movements and detect signs of drowsiness in real time. It calculates a live fatigue score, plots eye-behavior trends, and triggers audible/visual alerts the moment signs of sleepiness appear — aiming to reduce fatigue-related road accidents.

## 🎯 Features

| Feature | Description |
|---|---|
| 🎥 Real-Time Camera Monitoring | Live webcam feed processed frame-by-frame |
| 👁️ Eye Aspect Ratio (EAR) Detection | Tracks eye openness using 68-point facial landmarks |
| 😴 Drowsiness & Sleep Detection | Classifies driver state as Active, Drowsy, or Sleeping |
| 📊 Live Fatigue Score & Analytics | Continuously updated fatigue metric |
| 📈 Real-Time EAR Graph | Interactive Plotly chart of EAR over time |
| 🔔 Alert System | Wake-up warning triggered on sustained drowsiness |
| ⚡ Performance Modes | Tuned for smooth, low-latency execution |
| 🧵 Multi-threaded Camera Processing | Keeps UI responsive during capture |
| 💾 Session Logging | Exports session data to CSV for review |

## 🎬 Demo

> Add a screenshot or GIF of the app here, e.g.:
>
> `![App Demo](assets/demo.gif)`

## 🛠️ Tech Stack

- **Python** — core language
- **OpenCV** — video capture and image processing
- **dlib** — facial landmark detection
- **Streamlit** — web-based UI
- **Plotly** — real-time EAR visualization
- **NumPy** — numerical computations

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- A working webcam
- `pip` and `git` installed

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Pranavkale11/AI-Drowsiness-Monitor.git
cd AI-Drowsiness-Monitor
```

### 2️⃣ (Recommended) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
pip install dlib-bin
```

> 💡 `dlib-bin` provides prebuilt dlib wheels, which avoids the need to compile dlib from source on most systems.

### 4️⃣ Download the Facial Landmark Model

Download the dlib 68-point facial landmark predictor:

👉 [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Then:
1. Extract the `.bz2` archive to get `shape_predictor_68_face_landmarks.dat`
2. Place the extracted `.dat` file in the project root directory

### 5️⃣ Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your default browser (typically at `http://localhost:8501`).

## 🖥️ Usage

1. Launch the app with `streamlit run app.py`.
2. Grant camera permissions when prompted.
3. Position your face within the camera frame.
4. Monitor your live status (🟢 Active / 🟡 Drowsy / 🔴 Sleeping), fatigue score, and EAR graph.
5. Respond to alerts if drowsiness is detected.
6. Session data is logged to a CSV file for later review.

## 🧠 How It Works

1. **Capture** — Live video is captured from the webcam.
2. **Detect** — The face and 68 facial landmarks are detected per frame.
3. **Calculate** — The Eye Aspect Ratio (EAR) is computed from eye landmark coordinates.
4. **Classify** — Driver state is determined based on EAR thresholds over time:
   - 🟢 **Active** — Eyes open, EAR within normal range
   - 🟡 **Drowsy** — EAR dropping / eyes partially closed for a short period
   - 🔴 **Sleeping** — EAR below threshold for a sustained period
5. **Alert** — If sustained drowsiness or sleep is detected, the wake-up alert is triggered.

## 📁 Project Structure

```
AI-Drowsiness-Monitor/
├── app.py                                  # Main Streamlit application
├── requirements.txt                        # Python dependencies
├── shape_predictor_68_face_landmarks.dat   # Facial landmark model (downloaded separately)
└── README.md
```

> Update this section to match your actual repository layout if it differs.

## 📊 Performance Optimizations

- ⚡ Frame skipping for faster processing
- 🧵 Multi-threaded camera handling
- 📉 Reduced resolution for efficiency
- 🔄 Throttled UI updates
- 📈 Smoothed FPS calculation

## ⚠️ Limitations

- Performance may vary in low-light conditions
- Glasses/sunglasses can affect landmark detection accuracy
- Uses rule-based (EAR threshold) detection rather than a fully trained ML model
- Requires a visible, front-facing view of the driver's face

## 🚀 Future Improvements

- [ ] Deep learning-based drowsiness detection
- [ ] Mobile notifications / IoT integration
- [ ] Cloud logging & analytics dashboard
- [ ] Night vision / low-light enhancement
- [ ] Yawning and head-pose detection

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please open an issue first for major changes to discuss what you'd like to change.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

> Add a `LICENSE` file to the repo root if one doesn't already exist.

## 👨‍💻 Author

**Pranav Kale**
GitHub: [@Pranavkale11](https://github.com/Pranavkale11)

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on [GitHub](https://github.com/Pranavkale11/AI-Drowsiness-Monitor)!
