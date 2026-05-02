import streamlit as st
import joblib
import re
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(page_title="Sentiment Analysis AI", page_icon="🎭", layout="centered")

# --- CUSTOM CSS FOR DESIGN ---
st.markdown("""
    <style>
    .stApp {
        background-color: rgb(240, 245, 250);
    }
    h1 {
        color: rgb(30, 60, 90);
        font-weight: 800 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child {
        background-color: rgb(70, 130, 180);
        color: white;
        border-radius: 10px;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
    }
    div.stButton > button:hover {
        background-color: rgb(100, 149, 237);
        color: white;
    }
    .footer {
        text-align: center; 
        color: rgb(100, 100, 100); 
        font-weight: bold;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    return text

# 3. Model & Data Loading
@st.cache_resource
def load_assets():
    model_path = 'sentiment_model.pkl'
    vectorizer_path = 'vectorizer.pkl'
    dataset_path = 'Tweets.csv'
    
    assets = {"model": None, "vectorizer": None, "data": None}
    
    # Load Model and Vectorizer
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        assets["model"] = joblib.load(model_path)
        assets["vectorizer"] = joblib.load(vectorizer_path)
    
    # Load Dataset (Tweets.csv)
    if os.path.exists(dataset_path):
        assets["data"] = pd.read_csv(dataset_path)
        
    return assets

assets = load_assets()
model = assets["model"]
vectorizer = assets["vectorizer"]

# --- Page State Control ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

def go_to_home():
    st.session_state.page = "home"
    st.session_state.prediction_result = None

# --- 4. Page Routing ---

# A. Home Page
if st.session_state.page == "home":
    st.markdown("# **🎭 Sentiment Analysis AI**") 
    st.write("Enter a text below to analyze its emotional tone.")
    
    raw_text = st.text_area("Your Comment:", height=150, placeholder="Type something like 'I love this project!'")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("Analyze")
    
    if analyze_btn:
        if not raw_text.strip():
            st.warning("⚠️ Please enter some text first.")
        elif model is None or vectorizer is None:
            st.error("❌ Model files (PKL) not found. Please train and save your model first.")
        else:
            cleaned = clean_text(raw_text)
            vec = vectorizer.transform([cleaned])
            probs = model.predict_proba(vec)
            max_prob = probs.max()
            pred = model.predict(vec)[0]
            
            st.session_state.prediction_result = {
                "sentiment": pred,
                "confidence": max_prob,
                "text": raw_text
            }
            st.session_state.page = "result"
            st.rerun()

    # Optional: Show Data Preview from Tweets.csv
    if assets["data"] is not None:
        with st.expander("📂 View Sample Data (Tweets.csv)"):
            st.write(assets["data"].head(10))

# B. Result Page
elif st.session_state.page == "result":
    st.markdown("# **📊 Analysis Result**")
    res = st.session_state.prediction_result
    
    st.info(f"**Input Text:** '{res['text']}'")
    st.write("---")
    
    conf_val = res['confidence']
    if conf_val < 0.60:
        st.warning(f"⚠️ Low Confidence: ({conf_val:.2%}) - The tone might be neutral or unclear.")
    
    # Display Result with Emoji
    sentiment = res['sentiment'].lower()
    if sentiment == 'positive':
        st.success(f"### **Result: POSITIVE 😊** \n\n Confidence: {conf_val:.2%}")
        st.balloons()
    elif sentiment == 'negative':
        st.error(f"### **Result: NEGATIVE 😞** \n\n Confidence: {conf_val:.2%}")
    else:
        st.info(f"### **Result: NEUTRAL 😐** \n\n Confidence: {conf_val:.2%}")

    st.write("---")
    if st.button("⬅️ Back to Home"):
        go_to_home()
        st.rerun()
        
        
        
        
# Footer
st.markdown("---")
st.markdown("<div class='footer'>© 2026 AI Sentiment Project - Developed by Group 9</div>", unsafe_allow_html=True)