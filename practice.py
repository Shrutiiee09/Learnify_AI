import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from gtts import gTTS
import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load Hugging Face token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Ensure token is available
if not HF_TOKEN:
    st.error("Hugging Face API token is missing. Please check your .env file.")
    st.stop()

# Authenticate Hugging Face
login(token=HF_TOKEN)

# Use `flan-t5-large` (Open-Source)
MODEL_NAME = "google/flan-t5-large"

try:
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Function: Generate AI response (Text-to-Text)
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function: Convert text to speech
def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# Streamlit Web App
st.title("📚 Visual AI: Learn STEM with AI-Powered Explanations")
user_input = st.text_area("Enter your STEM question:", "What is gravity?")

if st.button("Generate Explanation"):
    response = generate_response(user_input)
    st.write("### AI Explanation:")
    st.write(response)

    speech_file = text_to_speech(response)
    st.audio(speech_file)
