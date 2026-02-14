import streamlit as st
import json
import os
import wave
import requests
import soundfile as sf
from vosk import Model, KaldiRecognizer
from langdetect import detect

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RAPID-100 Dispatch AI", layout="wide")

HF_TOKEN = ""

HF_SUMMARY_MODEL = "facebook/bart-large-cnn"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"

# ---------------- SPEECH TO TEXT ----------------

def transcribe_audio(audio_file):
    data, samplerate = sf.read(audio_file)

    # Convert to mono if needed
    if len(data.shape) > 1:
        data = data[:, 0]

    temp_wav = "temp.wav"
    sf.write(temp_wav, data, samplerate)

    wf = wave.open(temp_wav, "rb")
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, wf.getframerate())

    transcript = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            transcript += result.get("text", "") + " "

    final_result = json.loads(rec.FinalResult())
    transcript += final_result.get("text", "")

    return transcript.strip()

# ---------------- SUMMARIZATION ----------------

def summarize_text(text):
    payload = {
        "inputs": text,
        "parameters": {"max_length": 100}
    }
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        return text[:150]

# ---------------- EMERGENCY CLASSIFICATION ----------------

def classify_emergency(text):
    text_l = text.lower()

    emergency_type = "Unknown"
    severity = 2
    risks = []
    dispatch = []

    if any(word in text_l for word in ["fire", "burning", "smoke"]):
        emergency_type = "Fire"
        severity += 4
        risks.append("Active fire hazard")
        dispatch.append("Fire Brigade")

    if any(word in text_l for word in ["accident", "crash", "collision"]):
        emergency_type = "Traffic Accident"
        severity += 3
        risks.append("Vehicle collision")
        dispatch.append("Ambulance")

    if any(word in text_l for word in ["bleeding", "unconscious", "not breathing"]):
        emergency_type = "Medical Emergency"
        severity += 5
        risks.append("Critical health risk")
        dispatch.append("Ambulance")

    if any(word in text_l for word in ["gun", "knife", "attack"]):
        emergency_type = "Crime"
        severity += 4
        risks.append("Violent threat")
        dispatch.append("Police")

    severity = min(severity, 10)

    if not dispatch:
        dispatch.append("Local Response Unit")

    return emergency_type, severity, risks, dispatch

# ---------------- FAKE CALL DETECTION ----------------

def detect_fake_call(text):
    text_l = text.lower()

    suspicious_phrases = [
        "just kidding",
        "prank",
        "testing only",
        "nothing happened",
        "lol"
    ]

    if any(p in text_l for p in suspicious_phrases):
        return True

    if len(text.split()) < 5:
        return True

    return False

# ---------------- UI ----------------

st.sidebar.title("🚨 RAPID-100")
st.sidebar.success("System Online")

st.title("🚑 Real-Time AI Priority Incident Dispatch")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Incoming Call Stream")
    audio_file = st.file_uploader("Upload Emergency Call Audio", type=["wav"])

    if audio_file:
        st.audio(audio_file)

        if st.button("Process Call"):
            with st.spinner("Analyzing..."):

                transcript = transcribe_audio(audio_file)

                if transcript == "":
                    st.error("No speech detected.")
                else:
                    summary = summarize_text(transcript)
                    emergency_type, severity, risks, dispatch = classify_emergency(transcript)
                    is_fake = detect_fake_call(transcript)

                    language = detect(transcript)

                    result = {
                        "summary": summary,
                        "emergency_type": emergency_type,
                        "severity": severity,
                        "dispatch_routing": dispatch,
                        "key_risks": risks,
                        "language_detected": language,
                        "fake_call_detected": is_fake
                    }

                    st.session_state["transcript"] = transcript
                    st.session_state["result"] = result

with col2:
    if "result" in st.session_state:

        data = st.session_state["result"]
        score = data["severity"]

        color = "red" if score >= 8 else "orange" if score >=5 else "green"

        st.markdown(f"## ⚠️ Severity: :{color}[{score}/10]")
        st.write(f"### {data['summary']}")

        c1, c2 = st.columns(2)
        with c1:
            st.info(f"**Type:** {data['emergency_type']}")
        with c2:
            st.success(f"**Dispatch:** {', '.join(data['dispatch_routing'])}")

        st.warning(f"**Key Risks:** {', '.join(data['key_risks'])}")
        st.caption(f"Language: {data['language_detected']}")

        if data["fake_call_detected"]:
            st.error("⚠️ Potential Fake / Prank Call Detected")

        with st.expander("Live Transcript", expanded=True):
            st.write(st.session_state["transcript"])

        if st.button("🚑 DISPATCH UNITS NOW", type="primary"):
            if data["fake_call_detected"]:
                st.error("Dispatch blocked due to suspicious call.")
            else:
                st.toast("Units Dispatched Successfully!")
