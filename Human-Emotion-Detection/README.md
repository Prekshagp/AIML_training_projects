# 🎤 Human Emotion Detection from Voice

This project identifies human emotions from speech using machine learning and audio signal processing. It uses the RAVDESS dataset and extracts key vocal features to classify emotions like happy, sad, angry, calm, and more. The app is powered by **Streamlit** and provides real-time voice analysis and visual feedback.

---

## 🌟 Features

- 🎧 **Audio Feature Extraction**: MFCC, Chroma, Mel Spectrogram  
- 🌳 **Random Forest Classifier** for emotion recognition  
- 🧠 **Real-Time Emotion Detection** from microphone input  
- 📊 **Emotion Trend Visualization** in graph format  
- 💾 Logs prediction results to `session_data.csv`

---

## ⚙️ Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

> 💡 Tip: It's recommended to use a virtual environment to avoid dependency issues.

---

## ▶️ Running the Application

To launch the Streamlit web app:

```bash
streamlit run app/streamlit_app.py
```

🧭 This will open the dashboard in your browser, where you can start recording voice and see real-time emotion predictions.

---

## 🎼 Dataset

This project uses the **RAVDESS** dataset (Ryerson Audio-Visual Database of Emotional Speech and Song).  
👉 Download it from: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

Once downloaded, place the dataset in a folder named `data/` or according to your project structure.

---

## 📤 Output

- 🎙️ Predicts emotions in real-time from recorded voice  
- 📈 Displays emotion trend across the session  
- 📝 Logs all predictions to: `saved_sessions/session_data.csv`

---

## 🤝 Contributing

Have ideas to improve the model or UI? You're welcome to contribute!

```bash
# Steps to contribute:
1. Fork the repo
2. Create a new branch (git checkout -b feature-name)
3. Commit your changes (git commit -m "Add feature")
4. Push to the branch (git push origin feature-name)
5. Open a Pull Request
```

---

## 📝 License

Licensed under the **MIT License** ⚖️  
You are free to use, modify, and share this project under the terms in the `LICENSE` file.