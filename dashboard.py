import streamlit as st
import time
import pandas as pd
import requests
import plotly.graph_objects as go
import google.generativeai as genai
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="VitalGuard AI", layout="wide", page_icon="🩺")

# 1. HARD-CODED RENDER URL (This is the fix)
API_URL = "https://vitalguard-ai.onrender.com/latest"

# 2. API SETUP (Secrets)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.error("⚠️ Gemini API Key missing! Add it in Streamlit Secrets.")

# --- FUNCTIONS ---
def fetch_data():
    """Gets the latest data from the Cloud Server"""
    try:
        # We use the variable we defined above
        response = requests.get(API_URL, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None

def get_ai_analysis(heart_rate, spo2, sys_bp, dia_bp):
    """Asks Gemini for a quick health check"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        prompt = (f"Patient Vitals: HR {heart_rate}, SpO2 {spo2}%, BP {sys_bp}/{dia_bp}. "
                  "In 2 sentences: Is this dangerous? What should the nurse do?")
        response = model.generate_content(prompt)
        return response.text
    except:
        return "AI Analysis unavailable."

# --- UI LAYOUT ---
st.title("🩺 VitalGuard AI Monitor")
st.markdown(f"**Data Source:** `{API_URL}`") # Debugging line to prove connection

# Placeholders for live updates
col1, col2, col3, col4 = st.columns(4)
with col1: hr_metric = st.empty()
with col2: spo2_metric = st.empty()
with col3: bp_metric = st.empty()
with col4: temp_metric = st.empty()

chart_placeholder = st.empty()
ai_placeholder = st.empty()

# Initialize session state for the graph
if 'history' not in st.session_state:
    st.session_state['history'] = []

# --- MAIN LOOP ---
while True:
    data = fetch_data()
    
    if data:
        # Extract data safely
        hr = data.get('heart_rate', 0)
        spo2 = data.get('spo2', 0)
        sys_bp = data.get('systolic_bp', 0)
        dia_bp = data.get('diastolic_bp', 0)
        temp = data.get('body_temperature', 0)

        # Update Top Metrics
        hr_metric.metric("Heart Rate", f"{hr} bpm")
        spo2_metric.metric("SpO2", f"{spo2} %")
        bp_metric.metric("Blood Pressure", f"{sys_bp}/{dia_bp}")
        temp_metric.metric("Temperature", f"{temp} °F")

        # Update Graph History
        st.session_state['history'].append(hr)
        if len(st.session_state['history']) > 50:
            st.session_state['history'].pop(0)

        # Draw Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state['history'], mode='lines', name='HR'))
        fig.update_layout(title="Live Heart Rate Trend", height=300, margin=dict(l=0, r=0, t=30, b=0))
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # AI Analysis (Only runs if data is alarming or every 10th update)
        if len(st.session_state['history']) % 10 == 0:
            analysis = get_ai_analysis(hr, spo2, sys_bp, dia_bp)
            ai_placeholder.info(f"🤖 **AI Analysis:** {analysis}")

    else:
        st.warning(f"Waiting for data from {API_URL}...")
    
    time.sleep(2) # Update every 2 seconds