# 🎭 VoxSense — AI-Powered Emotion Recognition from Voice

<p align="center">
  <img src="Banner.png" alt="VoxSense Banner" />
</p>

<p align="center">
  <b>An AI-driven platform that analyzes human voice to detect emotions in real time</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.28.2-FF4B4B?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

---

## 📖 Overview

**VoxSense** is a sophisticated **AI-powered emotion recognition system** that analyzes vocal patterns to identify human emotional states.  
It delivers **real-time emotional intelligence** through a premium web interface, detailed analytics dashboards, and downloadable professional reports.

The platform supports **live voice recording**, **audio uploads**, and **deep learning–based inference**, making it suitable for **research, HR analytics, mental health insights, call-center intelligence, and human-computer interaction studies**.

---

## 🎥 Project Demo

> Complete walkthrough of VoxSense — from voice capture to emotion report generation.

▶ **Demo Video:**  
https://github.com/bhavyadoshi12/VoxSense_ER/raw/main/assets/video/VoxSense_ER.mp4

---

## 🖥️ Application Preview

<p align="center">
  <img src="assets/images/01.Home.png" width="900" alt="VoxSense Home Screen" />
</p>

<details>
<summary><b>📸 Click to view full application UI</b></summary>

### 🏠 Home Page
![Home](assets/images/01.Home.png)  
*Landing page with navigation and system overview.*

### 👤 Personal Profile
![Profile](assets/images/02.Personal_Profile.png)  
*User profile management and preferences.*

### 🎤 Voice Analysis
![Voice Analysis](assets/images/03.Voice_Analysis.png)  
*Analyze recorded or uploaded voice samples.*

### 🔴 Live Voice Session
![Live Session](assets/images/04.Live_Voice_Session.png)  
*Real-time voice recording and emotion inference.*

### ⬆️ Upload Session
![Upload](assets/images/05.Upload_Session.png)  
*Upload pre-recorded audio for analysis.*

### 📊 Dashboard
![Dashboard](assets/images/06.Dashboard_1.png)  
*Emotion probability charts and insights.*

### 📈 Advanced Analytics
![Dashboard 2](assets/images/07.Dashboard_2.png)  
*Comparative and deep-analysis metrics.*

### 📉 Summary View
![Dashboard 3](assets/images/08.Dashboard_3.png)  
*Tabular summaries and performance indicators.*

</details>

---

## ✨ Key Features

### 🧠 Emotion Intelligence Engine
- **AI-Powered Deep Learning Analysis** using PyTorch
- **Rule-Based Emotion Detection** as a robust fallback
- Multi-emotion probability prediction

### 🎙️ Flexible Audio Input
- Live voice recording via browser
- Upload audio files: `.wav`, `.mp3`, `.m4a`, `.ogg`

### 📊 Advanced Analytics Dashboard
- Emotion probability visualizations (Bar, Pie, Radar)
- Audio waveform & spectrogram analysis
- Acoustic metrics: pitch, energy, frequency patterns

### 📄 Professional Reports
- Downloadable **PDF Emotional Intelligence Reports**
- Detailed summaries and visual insights

### 🎨 Premium UI/UX
- Modern Streamlit interface
- Glassmorphism cards & gradients
- Smooth animations & responsive design

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-----|-------------|
| Backend | Python |
| Web Framework | Streamlit |
| AI / ML | PyTorch, Librosa, Scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Plotly, Matplotlib |
| Audio Handling | SoundFile, PyDub |
| Reports | ReportLab |

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/bhavyadoshi12/VoxSense_ER.git
cd VoxSense_ER
````

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

**Activate it:**

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
python install_dependencies.py
```

> Ensures correct versions and dependency compatibility.

---

## 🧠 Pre-trained Models

Located in `assets/models/`:

* `best_emotion_model.pth` — Primary audio-based emotion model
* `cnn_emotion_model.pth` — Optional image-based emotion model

📌 Custom training scripts available:

* `train_emotion_model.py`
* `train_cnn_emotion.py`

---

## 🚀 Run the Application

Ensure the virtual environment is active.

```bash
streamlit run main.py
```

The application will launch in your default browser.

---

## 🤝 Contributing

Contributions are welcome!

⚠️ **Known Area for Improvement:**
Emotion confidence may sometimes be ~30%.
Enhancements to model accuracy are highly encouraged.

### Contribution Steps

1. Fork the repository
2. Create a feature branch

   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit changes

   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push & open a Pull Request

---

## ✍️ Authors

Crafted with excellence by:

* **Pranjal Belalekar**
* **Bhavya Doshi**

---

## 📄 License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.

---

<p align="center">
  💎 <b>Thank you for exploring VoxSense!</b><br/>
  Turning voices into emotional intelligence.
</p>
