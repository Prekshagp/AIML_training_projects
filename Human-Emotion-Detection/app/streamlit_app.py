import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import joblib
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from utils.feature_extraction import extract_features
import pandas as pd
import tempfile
import datetime

st.title("🎙️ Human Emotion Detection from Voice")

model = joblib.load('model/emotion_model.pkl')
emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

def predict_emotion(file_path):
    features = extract_features(file_path).reshape(1, -1)
    result = model.predict(features)[0]
    return emotions[int(result)]

duration = st.slider("Select Recording Duration (seconds)", 2, 10, 4)
if st.button("🔴 Record"):
    st.write("Recording...")
    fs = 22050
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(tmpfile.name, fs, recording)
    st.audio(tmpfile.name)

    emotion = predict_emotion(tmpfile.name)
    st.success(f"Predicted Emotion: **{emotion}**")

    session_file = "saved_sessions/session_data.csv"
    now = datetime.datetime.now()
    log = pd.DataFrame([[now.strftime("%Y-%m-%d %H:%M:%S"), emotion]], columns=["Time", "Emotion"])
    if os.path.exists(session_file):
        log.to_csv(session_file, mode='a', header=False, index=False)
    else:
        log.to_csv(session_file, index=False)

if st.checkbox("📊 Show Emotion Trend Graph"):
    if os.path.exists("saved_sessions/session_data.csv"):
        df = pd.read_csv("saved_sessions/session_data.csv")
        st.line_chart(df["Emotion"].value_counts())
