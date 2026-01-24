import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import google.generativeai as genai
import numpy as np
from collections import deque
import hashlib

# PASTE THIS:
# 1. Replace with your Render URL from Part 3
NGROK_URL = "https://vitalguard-api.onrender.com/latest" 

# 2. Load API Key securely from Streamlit Secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except FileNotFoundError:
    st.error("Secrets not found. Please set GEMINI_API_KEY in Streamlit Cloud settings.")
# --- ENHANCED HELPER FUNCTIONS ---
def generate_unique_key(base_name):
    """Generate unique keys for Plotly charts to prevent ID conflicts"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    return f"{base_name}_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

def get_live_ai_insight(vitals, history_df, alert_context):
    """Generate sophisticated AI-powered clinical insights with context"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Calculate trends
        trend_info = ""
        if len(history_df) >= 5:
            recent_hr = history_df['HR'].tail(5).tolist()
            recent_temp = history_df['Temp'].tail(5).tolist()
            
            hr_trend = "increasing" if recent_hr[-1] > recent_hr[0] else "decreasing" if recent_hr[-1] < recent_hr[0] else "stable"
            temp_trend = "rising" if recent_temp[-1] > recent_temp[0] else "falling" if recent_temp[-1] < recent_temp[0] else "stable"
            
            trend_info = f"\nRecent trends: HR is {hr_trend}, Temperature is {temp_trend}"
        
        # Build context-aware prompt
        prompt = f"""You are an expert ICU monitoring AI analyzing real-time patient vitals. Provide a clinical assessment.

CURRENT VITALS:
- Heart Rate: {vitals['heart_rate']} BPM (Normal: 60-100)
- Temperature: {vitals['temperature']} °F (Normal: 97-100)
- Blood Pressure: {vitals['blood_pressure']} mmHg (Normal: 90-120/60-80)
- SpO2: {vitals.get('spo2', 98)}% (Normal: >95%)
- Computed Risk Score: {vitals['risk_score']}/10
{trend_info}

RECENT ALERTS: {', '.join(alert_context[-3:]) if alert_context else 'None'}

TASK:
1. Identify specific clinical concerns (be explicit about what's abnormal)
2. Explain the clinical significance (why it matters)
3. Provide ONE specific, actionable nursing intervention
4. Rate urgency: ROUTINE / MONITOR CLOSELY / URGENT / CRITICAL

Format: [URGENCY] Clinical finding | Significance | Action
Keep response to 2-3 sentences maximum."""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Fallback with actual analysis
        issues = []
        if vitals['heart_rate'] > 100:
            issues.append(f"Tachycardia ({vitals['heart_rate']} BPM)")
        elif vitals['heart_rate'] < 60:
            issues.append(f"Bradycardia ({vitals['heart_rate']} BPM)")
        
        if vitals['temperature'] > 100.4:
            issues.append(f"Fever ({vitals['temperature']}°F)")
        elif vitals['temperature'] < 97:
            issues.append(f"Hypothermia ({vitals['temperature']}°F)")
        
        bp_parts = vitals['blood_pressure'].split('/')
        if len(bp_parts) == 2:
            sys, dias = int(bp_parts[0]), int(bp_parts[1])
            if sys > 140 or dias > 90:
                issues.append(f"Hypertension ({vitals['blood_pressure']})")
            elif sys < 90 or dias < 60:
                issues.append(f"Hypotension ({vitals['blood_pressure']})")
        
        if issues:
            return f"[MONITOR CLOSELY] Detected: {', '.join(issues)}. Risk score {vitals['risk_score']}/10. Recommend: Continue monitoring and notify physician if trends worsen."
        else:
            return f"[ROUTINE] Vitals within acceptable parameters. Risk score {vitals['risk_score']}/10. Continue routine monitoring."

def calculate_advanced_risk_score(hr, temp, systolic, diastolic, spo2=98, history_df=None):
    """Enhanced risk calculation with trend analysis and multi-factor scoring"""
    risk = 0
    risk_factors = []
    
    # Heart rate with graduated severity
    if hr < 50:
        risk += 4
        risk_factors.append("Severe Bradycardia")
    elif hr < 60:
        risk += 2
        risk_factors.append("Bradycardia")
    elif hr > 130:
        risk += 4
        risk_factors.append("Severe Tachycardia")
    elif hr > 100:
        risk += 2
        risk_factors.append("Tachycardia")
    
    # Temperature with fever grades
    if temp > 103:
        risk += 4
        risk_factors.append("High-grade Fever")
    elif temp > 100.4:
        risk += 2
        risk_factors.append("Fever")
    elif temp < 95:
        risk += 4
        risk_factors.append("Severe Hypothermia")
    elif temp < 97:
        risk += 2
        risk_factors.append("Mild Hypothermia")
    
    # Blood pressure with hypertensive crisis detection
    if systolic > 180 or diastolic > 120:
        risk += 5
        risk_factors.append("Hypertensive Crisis")
    elif systolic > 140 or diastolic > 90:
        risk += 2
        risk_factors.append("Hypertension")
    elif systolic < 80 or diastolic < 50:
        risk += 5
        risk_factors.append("Severe Hypotension")
    elif systolic < 90 or diastolic < 60:
        risk += 3
        risk_factors.append("Hypotension")
    
    # SpO2 scoring
    if spo2 < 90:
        risk += 5
        risk_factors.append("Critical Hypoxemia")
    elif spo2 < 95:
        risk += 3
        risk_factors.append("Hypoxemia")
    
    # Trend analysis (if history available)
    if history_df is not None and len(history_df) >= 5:
        recent_risks = history_df['RiskScore'].tail(5).tolist()
        if len(recent_risks) >= 2:
            # Rapid deterioration detection
            if recent_risks[-1] - recent_risks[0] >= 3:
                risk += 2
                risk_factors.append("Rapid Deterioration")
            
            # Sustained elevation
            if all(r >= 5 for r in recent_risks[-3:]):
                risk += 1
                risk_factors.append("Sustained High Risk")
    
    # Multi-organ involvement (combination of abnormalities)
    abnormal_count = sum([
        hr < 60 or hr > 100,
        temp < 97 or temp > 100.4,
        systolic < 90 or systolic > 140,
        diastolic < 60 or diastolic > 90,
        spo2 < 95
    ])
    
    if abnormal_count >= 3:
        risk += 2
        risk_factors.append("Multi-system Involvement")
    
    return min(risk, 10), risk_factors

def get_risk_level(score):
    """Enhanced risk categorization"""
    if score <= 2:
        return "Low", "#00ff88", "✓"
    elif score <= 4:
        return "Moderate", "#ffdd00", "⚠"
    elif score <= 6:
        return "Elevated", "#ffaa00", "⚠⚠"
    elif score <= 8:
        return "High", "#ff6600", "⚠⚠⚠"
    else:
        return "Critical", "#ff3333", "🚨"

def get_vital_status(vital_type, value):
    """Determine if a vital is in normal, warning, or critical range"""
    ranges = {
        'hr': {'critical_low': 50, 'low': 60, 'high': 100, 'critical_high': 130},
        'temp': {'critical_low': 95, 'low': 97, 'high': 100, 'critical_high': 103},
        'systolic': {'critical_low': 80, 'low': 90, 'high': 140, 'critical_high': 180},
        'diastolic': {'critical_low': 50, 'low': 60, 'high': 90, 'critical_high': 120},
        'spo2': {'critical_low': 90, 'low': 95, 'high': 100, 'critical_high': 100}
    }
    
    r = ranges.get(vital_type, {})
    if not r:
        return 'normal', '#00ff88'
    
    if value <= r.get('critical_low', 0):
        return 'critical', '#ff3333'
    elif value < r.get('low', 0):
        return 'warning', '#ffaa00'
    elif value > r.get('critical_high', 999):
        return 'critical', '#ff3333'
    elif value > r.get('high', 999):
        return 'warning', '#ffaa00'
    else:
        return 'normal', '#00ff88'

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VitalGuard AI | Advanced Clinical Monitoring", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        box-shadow: 0 4px 15px rgba(0,212,255,0.2);
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 6px 25px rgba(0,212,255,0.4);
        transform: translateY(-2px);
    }
    
    .stMetric { 
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 100%);
        padding: 18px; 
        border-radius: 12px;
        transition: all 0.3s ease;
        border: 1px solid rgba(0,212,255,0.1);
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,212,255,0.3);
        border-color: rgba(0,212,255,0.3);
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff3333 0%, #cc0000 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        animation: pulse 2s infinite;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255,51,51,0.4);
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255,170,0,0.4);
    }
    
    .alert-info {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,212,255,0.3);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.85; transform: scale(1.02); }
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.5px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .risk-indicator {
        font-size: 32px;
        font-weight: 800;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .stat-highlight {
        background: rgba(0,212,255,0.1);
        padding: 8px 12px;
        border-radius: 6px;
        border-left: 3px solid #00d4ff;
        margin: 5px 0;
    }
    
    .trend-arrow-up {
        color: #ff3333;
        font-size: 20px;
        font-weight: bold;
    }
    
    .trend-arrow-down {
        color: #00ff88;
        font-size: 20px;
        font-weight: bold;
    }
    
    .ai-insight-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 12px;
        border: 2px solid #00d4ff;
        box-shadow: 0 4px 20px rgba(0,212,255,0.3);
        margin: 15px 0;
    }
    
    .vital-normal { color: #00ff88; font-weight: 600; }
    .vital-warning { color: #ffaa00; font-weight: 600; }
    .vital-critical { color: #ff3333; font-weight: 700; animation: pulse 2s infinite; }
    </style>
    """, unsafe_allow_html=True)

# --- ENHANCED STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        'Time', 'Timestamp', 'HR', 'Temp', 'Systolic', 'Diastolic', 'SpO2', 'RiskScore', 'RiskFactors'
    ])

if 'alerts' not in st.session_state:
    st.session_state.alerts = deque(maxlen=50)  # Use deque for better performance

if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = deque(maxlen=20)

if 'last_ai_call' not in st.session_state:
    st.session_state.last_ai_call = datetime.now() - timedelta(seconds=30)

if 'alert_context' not in st.session_state:
    st.session_state.alert_context = deque(maxlen=10)

if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()

if 'total_alerts' not in st.session_state:
    st.session_state.total_alerts = {'critical': 0, 'warning': 0, 'info': 0}

# --- ENHANCED HEADER ---
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("🩺 VitalGuard AI")
    st.markdown("### **Advanced Real-Time Patient Monitoring System**")
    session_duration = datetime.now() - st.session_state.session_start
    st.caption(f"Session Duration: {str(session_duration).split('.')[0]}")

with col2:
    st.markdown("### **Patient ID: 1BM23CD044**")
    st.caption("Room: ICU-3 | Bed: A")

with col3:
    current_time = datetime.now()
    st.markdown(f"### 🕐 {current_time.strftime('%H:%M:%S')}")
    st.caption(current_time.strftime('%B %d, %Y'))

st.write("---")

# --- ENHANCED SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    # Refresh rate control
    refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2, help="Update frequency for vital signs")
    
    # AI insight interval
    ai_interval = st.slider("AI Analysis Interval (seconds)", 15, 60, 30, help="How often to generate AI insights")
    
    st.write("---")
    st.header("📊 Session Statistics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total Alerts", len(st.session_state.alerts))
    with col_b:
        st.metric("Critical", st.session_state.total_alerts['critical'])
    
    col_c, col_d = st.columns(2)
    with col_c:
        st.metric("Warnings", st.session_state.total_alerts['warning'])
    with col_d:
        st.metric("Info", st.session_state.total_alerts['info'])
    
    st.write("---")
    st.header("⚠️ Clinical Alerts")
    
    if not st.session_state.alerts:
        st.info("✓ No active alerts - Patient stable")
    else:
        alert_filter = st.selectbox("Filter Alerts", ["All", "Critical", "Warning", "Info"], key="alert_filter_select")
        
        displayed_alerts = 0
        for alert in reversed(list(st.session_state.alerts)):
            # Determine alert type and display accordingly
            if "CRITICAL" in alert or "🚨" in alert:
                if alert_filter in ["All", "Critical"]:
                    st.markdown(f'<div class="alert-critical">{alert}</div>', unsafe_allow_html=True)
                    displayed_alerts += 1
            elif "Warning" in alert or "⚠️" in alert:
                if alert_filter in ["All", "Warning"]:
                    st.markdown(f'<div class="alert-warning">{alert}</div>', unsafe_allow_html=True)
                    displayed_alerts += 1
            else:
                if alert_filter in ["All", "Info"]:
                    st.markdown(f'<div class="alert-info">{alert}</div>', unsafe_allow_html=True)
                    displayed_alerts += 1
            
            if displayed_alerts >= 10:  # Limit displayed alerts
                break
    
    st.write("---")
    st.header("🤖 AI Clinical Insights")
    
    if not st.session_state.ai_insights:
        st.info("🔄 AI analysis initializing...")
    else:
        for idx, insight in enumerate(reversed(list(st.session_state.ai_insights)[:3])):
            urgency_color = "#ff3333" if "CRITICAL" in insight['text'] else "#ffaa00" if "URGENT" in insight['text'] else "#00d4ff"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                        padding: 12px; border-radius: 8px; margin: 8px 0;
                        border-left: 4px solid {urgency_color};">
                <p style="color: #aaa; font-size: 11px; margin: 0;">{insight['time']}</p>
                <p style="margin: 5px 0; font-size: 13px;">{insight['text']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.write("---")
    if st.button("🔄 Reset Dashboard", use_container_width=True):
        st.session_state.history = pd.DataFrame(columns=[
            'Time', 'Timestamp', 'HR', 'Temp', 'Systolic', 'Diastolic', 'SpO2', 'RiskScore', 'RiskFactors'
        ])
        st.session_state.alerts.clear()
        st.session_state.ai_insights.clear()
        st.session_state.total_alerts = {'critical': 0, 'warning': 0, 'info': 0}
        st.session_state.session_start = datetime.now()
        st.rerun()

# --- MAIN DASHBOARD ---
placeholder = st.empty()

while True:
    try:
        # Fetch data
        data = requests.get(NGROK_URL, timeout=5).json()
        
        # Data processing with robust type conversion
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        hr = round(float(data['heart_rate']), 1)
        temp = round(float(data['body_temperature']), 1)
        systolic = int(round(float(data['systolic_bp'])))
        diastolic = int(round(float(data['diastolic_bp'])))
        spo2 = int(round(float(data.get('spo2', 98))))  # Simulated if not provided
        
        # Calculate advanced risk score
        risk_score, risk_factors = calculate_advanced_risk_score(
            hr, temp, systolic, diastolic, spo2, st.session_state.history
        )
        risk_level, risk_color, risk_icon = get_risk_level(risk_score)
        
        # Update history
        new_row = {
            'Time': time_str,
            'Timestamp': now,
            'HR': hr,
            'Temp': temp,
            'Systolic': systolic,
            'Diastolic': diastolic,
            'SpO2': spo2,
            'RiskScore': risk_score,
            'RiskFactors': ', '.join(risk_factors) if risk_factors else 'None'
        }
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([new_row])], 
            ignore_index=True
        ).tail(200)  # Keep more history for better trending
        
        # Enhanced alert generation with severity classification
        new_alerts_generated = False
        
        # Heart rate alerts
        if hr > 130:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: Severe Tachycardia - {hr} BPM | Immediate intervention required"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Critical HR: {hr}")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        elif hr > 100:
            alert_msg = f"[{time_str}] ⚠️ WARNING: Tachycardia - {hr} BPM | Monitor closely"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"High HR: {hr}")
            st.session_state.total_alerts['warning'] += 1
            new_alerts_generated = True
        elif hr < 50:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: Severe Bradycardia - {hr} BPM | Urgent assessment needed"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Critical HR: {hr}")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        elif hr < 60:
            alert_msg = f"[{time_str}] ⚠️ WARNING: Bradycardia - {hr} BPM | Continue monitoring"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Low HR: {hr}")
            st.session_state.total_alerts['warning'] += 1
            new_alerts_generated = True
        
        # Temperature alerts
        if temp > 103:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: High-grade Fever - {temp} °F | Antipyretics & cooling measures"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Critical Temp: {temp}")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        elif temp > 100.4:
            alert_msg = f"[{time_str}] ⚠️ WARNING: Fever detected - {temp} °F | Consider antipyretics"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Elevated Temp: {temp}")
            st.session_state.total_alerts['warning'] += 1
            new_alerts_generated = True
        elif temp < 95:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: Severe Hypothermia - {temp} °F | Warming protocol initiated"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Critical Temp: {temp}")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        elif temp < 97:
            alert_msg = f"[{time_str}] ⚠️ WARNING: Mild Hypothermia - {temp} °F | Apply warming measures"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Low Temp: {temp}")
            st.session_state.total_alerts['warning'] += 1
            new_alerts_generated = True
        
        # Blood pressure alerts
        if systolic > 180 or diastolic > 120:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: Hypertensive Crisis - {systolic}/{diastolic} | Urgent BP control needed"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Critical BP: {systolic}/{diastolic}")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        elif systolic > 140 or diastolic > 90:
            alert_msg = f"[{time_str}] ⚠️ WARNING: Hypertension - {systolic}/{diastolic} | Monitor and document"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"High BP: {systolic}/{diastolic}")
            st.session_state.total_alerts['warning'] += 1
            new_alerts_generated = True
        elif systolic < 80 or diastolic < 50:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: Severe Hypotension - {systolic}/{diastolic} | Fluid resuscitation/pressors"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Critical BP: {systolic}/{diastolic}")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        elif systolic < 90 or diastolic < 60:
            alert_msg = f"[{time_str}] ⚠️ WARNING: Hypotension - {systolic}/{diastolic} | Assess perfusion"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Low BP: {systolic}/{diastolic}")
            st.session_state.total_alerts['warning'] += 1
            new_alerts_generated = True
        
        # SpO2 alerts
        if spo2 < 90:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: Severe Hypoxemia - SpO2 {spo2}% | Increase O2, assess airway"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Critical SpO2: {spo2}")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        elif spo2 < 95:
            alert_msg = f"[{time_str}] ⚠️ WARNING: Low SpO2 - {spo2}% | Supplemental oxygen recommended"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append(f"Low SpO2: {spo2}")
            st.session_state.total_alerts['warning'] += 1
            new_alerts_generated = True
        
        # Multi-system alert
        if len(risk_factors) >= 3:
            alert_msg = f"[{time_str}] 🚨 CRITICAL: Multi-system involvement detected | Factors: {', '.join(risk_factors[:3])}"
            st.session_state.alerts.append(alert_msg)
            st.session_state.alert_context.append("Multi-system alert")
            st.session_state.total_alerts['critical'] += 1
            new_alerts_generated = True
        
        # AI insight generation (configurable interval)
        time_since_last_ai = (now - st.session_state.last_ai_call).seconds
        should_generate_ai = time_since_last_ai >= ai_interval
        
        # Force AI generation if new critical alerts
        if new_alerts_generated and any("CRITICAL" in str(a) for a in list(st.session_state.alerts)[-3:]):
            should_generate_ai = True
        
        if should_generate_ai:
            vitals_summary = {
                'heart_rate': hr,
                'temperature': temp,
                'blood_pressure': f"{systolic}/{diastolic}",
                'spo2': spo2,
                'risk_score': risk_score
            }
            ai_insight = get_live_ai_insight(
                vitals_summary, 
                st.session_state.history,
                list(st.session_state.alert_context)
            )
            st.session_state.ai_insights.append({
                'time': time_str,
                'text': ai_insight,
                'risk_score': risk_score
            })
            st.session_state.last_ai_call = now
        
        # Render dashboard
        with placeholder.container():
            # --- ROW 1: KEY METRICS ---
            metric_cols = st.columns(5)
            
            hr_status, hr_color = get_vital_status('hr', hr)
            temp_status, temp_color = get_vital_status('temp', temp)
            sys_status, sys_color = get_vital_status('systolic', systolic)
            spo2_status, spo2_color = get_vital_status('spo2', spo2)
            
            with metric_cols[0]:
                delta_hr = "Normal" if hr_status == 'normal' else "ABNORMAL"
                st.metric(
                    "💓 Heart Rate", 
                    f"{hr} BPM", 
                    delta=delta_hr,
                    delta_color="off" if hr_status == 'normal' else "inverse"
                )
                st.markdown(f'<p class="vital-{hr_status}">Status: {hr_status.upper()}</p>', unsafe_allow_html=True)
            
            with metric_cols[1]:
                delta_temp = "Normal" if temp_status == 'normal' else "ABNORMAL"
                st.metric(
                    "🌡️ Temperature", 
                    f"{temp} °F", 
                    delta=delta_temp,
                    delta_color="off" if temp_status == 'normal' else "inverse"
                )
                st.markdown(f'<p class="vital-{temp_status}">Status: {temp_status.upper()}</p>', unsafe_allow_html=True)
            
            with metric_cols[2]:
                bp_status = "Normal" if sys_status == 'normal' else "ABNORMAL"
                st.metric(
                    "🩸 Blood Pressure", 
                    f"{systolic}/{diastolic}",
                    delta=bp_status,
                    delta_color="off" if sys_status == 'normal' else "inverse"
                )
                st.markdown(f'<p class="vital-{sys_status}">Status: {sys_status.upper()}</p>', unsafe_allow_html=True)
            
            with metric_cols[3]:
                delta_spo2 = "Optimal" if spo2_status == 'normal' else "LOW"
                st.metric(
                    "🫁 SpO2", 
                    f"{spo2}%", 
                    delta=delta_spo2,
                    delta_color="off" if spo2_status == 'normal' else "inverse"
                )
                st.markdown(f'<p class="vital-{spo2_status}">Status: {spo2_status.upper()}</p>', unsafe_allow_html=True)
            
            with metric_cols[4]:
                st.markdown(f'<div class="risk-indicator" style="background-color: {risk_color};">{risk_icon} {risk_level}</div>', unsafe_allow_html=True)
                st.metric(
                    "Risk Score",
                    f"{risk_score}/10",
                    delta=f"{len(risk_factors)} factors" if risk_factors else "No factors",
                    delta_color="inverse" if risk_score > 4 else "off"
                )
            
            # Risk factors display
            if risk_factors:
                st.markdown(f"""
                <div class="stat-highlight">
                    <strong>Active Risk Factors:</strong> {', '.join(risk_factors)}
                </div>
                """, unsafe_allow_html=True)
            
            st.write("---")
            
            # --- ROW 2: TREND GRAPHS ---
            graph_col1, graph_col2 = st.columns(2)
            
            with graph_col1:
                st.subheader("📈 Heart Rate Trend Analysis")
                fig_hr = go.Figure()
                
                # Normal range zones
                fig_hr.add_hrect(y0=0, y1=50, fillcolor="red", opacity=0.05, line_width=0)
                fig_hr.add_hrect(y0=50, y1=60, fillcolor="yellow", opacity=0.05, line_width=0)
                fig_hr.add_hrect(y0=60, y1=100, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Normal Range")
                fig_hr.add_hrect(y0=100, y1=130, fillcolor="yellow", opacity=0.05, line_width=0)
                fig_hr.add_hrect(y0=130, y1=200, fillcolor="red", opacity=0.05, line_width=0)
                
                # Add reference lines
                fig_hr.add_hline(y=60, line_dash="dash", line_color="yellow", opacity=0.5, annotation_text="Lower Limit")
                fig_hr.add_hline(y=100, line_dash="dash", line_color="yellow", opacity=0.5, annotation_text="Upper Limit")
                
                fig_hr.add_trace(go.Scatter(
                    x=st.session_state.history['Time'],
                    y=st.session_state.history['HR'],
                    mode='lines+markers',
                    name='Heart Rate',
                    line=dict(color='#00d4ff', width=2.5),
                    marker=dict(size=5, color=st.session_state.history['HR'],
                               colorscale=[[0, '#00ff88'], [0.5, '#ffaa00'], [1, '#ff3333']],
                               cmin=50, cmax=130),
                    fill='tozeroy',
                    fillcolor='rgba(0, 212, 255, 0.1)',
                    hovertemplate='<b>%{y} BPM</b><br>Time: %{x}<extra></extra>'
                ))
                
                fig_hr.update_layout(
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title="BPM",
                    xaxis_title="Time",
                    showlegend=False,
                    hovermode='x unified',
                    yaxis=dict(range=[40, 150])
                )
                st.plotly_chart(fig_hr, use_container_width=True, key=generate_unique_key("hr_chart"))
            
            with graph_col2:
                st.subheader("🌡️ Temperature Trend Analysis")
                fig_temp = go.Figure()
                
                # Temperature zones
                fig_temp.add_hrect(y0=94, y1=95, fillcolor="red", opacity=0.05, line_width=0)
                fig_temp.add_hrect(y0=95, y1=97, fillcolor="yellow", opacity=0.05, line_width=0)
                fig_temp.add_hrect(y0=97, y1=100, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Normal")
                fig_temp.add_hrect(y0=100, y1=103, fillcolor="yellow", opacity=0.05, line_width=0)
                fig_temp.add_hrect(y0=103, y1=106, fillcolor="red", opacity=0.05, line_width=0)
                
                fig_temp.add_hline(y=100.4, line_dash="dash", line_color="orange", opacity=0.5, annotation_text="Fever Threshold")
                
                fig_temp.add_trace(go.Scatter(
                    x=st.session_state.history['Time'],
                    y=st.session_state.history['Temp'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#ff6b6b', width=2.5),
                    marker=dict(size=5, color=st.session_state.history['Temp'],
                               colorscale=[[0, '#00d4ff'], [0.5, '#ffaa00'], [1, '#ff3333']],
                               cmin=95, cmax=103),
                    fill='tozeroy',
                    fillcolor='rgba(255, 107, 107, 0.1)',
                    hovertemplate='<b>%{y} °F</b><br>Time: %{x}<extra></extra>'
                ))
                
                fig_temp.update_layout(
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title="°F",
                    xaxis_title="Time",
                    showlegend=False,
                    hovermode='x unified',
                    yaxis=dict(range=[94, 106])
                )
                st.plotly_chart(fig_temp, use_container_width=True, key=generate_unique_key("temp_chart"))
            
            # --- ROW 3: BLOOD PRESSURE & SPO2 ---
            graph_col3, graph_col4 = st.columns(2)
            
            with graph_col3:
                st.subheader("🩸 Blood Pressure Dynamics")
                fig_bp = go.Figure()
                
                # BP zones
                fig_bp.add_hrect(y0=90, y1=120, fillcolor="green", opacity=0.08, line_width=0)
                fig_bp.add_hrect(y0=60, y1=80, fillcolor="green", opacity=0.08, line_width=0)
                
                fig_bp.add_trace(go.Scatter(
                    x=st.session_state.history['Time'],
                    y=st.session_state.history['Systolic'],
                    mode='lines+markers',
                    name='Systolic',
                    line=dict(color='#ff6b6b', width=2.5),
                    marker=dict(size=6, symbol='circle'),
                    hovertemplate='<b>Systolic: %{y} mmHg</b><extra></extra>'
                ))
                
                fig_bp.add_trace(go.Scatter(
                    x=st.session_state.history['Time'],
                    y=st.session_state.history['Diastolic'],
                    mode='lines+markers',
                    name='Diastolic',
                    line=dict(color='#4ecdc4', width=2.5),
                    marker=dict(size=6, symbol='square'),
                    hovertemplate='<b>Diastolic: %{y} mmHg</b><extra></extra>'
                ))
                
                # Add reference lines
                fig_bp.add_hline(y=140, line_dash="dot", line_color="red", opacity=0.4, annotation_text="HTN Threshold", annotation_position="right")
                fig_bp.add_hline(y=90, line_dash="dot", line_color="yellow", opacity=0.4, annotation_text="Low BP", annotation_position="right")
                
                fig_bp.update_layout(
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title="mmHg",
                    xaxis_title="Time",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis=dict(range=[50, 180])
                )
                st.plotly_chart(fig_bp, use_container_width=True, key=generate_unique_key("bp_chart"))
            
            with graph_col4:
                st.subheader("🫁 SpO2 & Risk Score Monitor")
                
                # Create subplot with dual y-axes
                fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
                
                # SpO2 trace
                fig_dual.add_trace(
                    go.Scatter(
                        x=st.session_state.history['Time'],
                        y=st.session_state.history['SpO2'],
                        mode='lines+markers',
                        name='SpO2',
                        line=dict(color='#00d4ff', width=2.5),
                        marker=dict(size=6),
                        hovertemplate='<b>SpO2: %{y}%</b><extra></extra>'
                    ),
                    secondary_y=False
                )
                
                # Risk Score trace
                risk_colors = ['#00ff88' if s <= 2 else '#ffdd00' if s <= 4 else '#ffaa00' if s <= 6 else '#ff3333' 
                              for s in st.session_state.history['RiskScore']]
                
                fig_dual.add_trace(
                    go.Scatter(
                        x=st.session_state.history['Time'],
                        y=st.session_state.history['RiskScore'],
                        mode='lines+markers',
                        name='Risk Score',
                        line=dict(color='#ffa500', width=2),
                        marker=dict(size=8, color=risk_colors),
                        hovertemplate='<b>Risk: %{y}/10</b><extra></extra>'
                    ),
                    secondary_y=True
                )
                
                # Add SpO2 critical zone
                fig_dual.add_hrect(y0=95, y1=100, fillcolor="green", opacity=0.1, line_width=0, secondary_y=False)
                fig_dual.add_hrect(y0=90, y1=95, fillcolor="yellow", opacity=0.05, line_width=0, secondary_y=False)
                
                fig_dual.update_xaxes(title_text="Time")
                fig_dual.update_yaxes(title_text="SpO2 (%)", range=[85, 100], secondary_y=False)
                fig_dual.update_yaxes(title_text="Risk Score", range=[0, 10], secondary_y=True)
                
                fig_dual.update_layout(
                    template="plotly_dark",
                    height=320,
                    margin=dict(l=20, r=20, t=20, b=20),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_dual, use_container_width=True, key=generate_unique_key("dual_chart"))
            
            # --- ROW 4: ADVANCED ANALYTICS ---
            st.write("---")
            analytics_col1, analytics_col2, analytics_col3 = st.columns([2, 2, 1])
            
            with analytics_col1:
                st.subheader("📊 Multi-Vital Radar Analysis")
                
                if len(st.session_state.history) > 0:
                    latest = st.session_state.history.iloc[-1]
                    
                    # Normalize values
                    hr_norm = min(max((hr / 100) * 100, 0), 150)
                    temp_norm = min(max(((temp - 95) / 8) * 100, 0), 150)
                    bp_norm = min(max((systolic / 140) * 100, 0), 150)
                    spo2_norm = spo2
                    
                    categories = ['Heart Rate', 'Temperature', 'Blood Pressure', 'SpO2']
                    values = [hr_norm, temp_norm, bp_norm, spo2_norm]
                    
                    fig_radar = go.Figure()
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values + [values[0]],
                        theta=categories + [categories[0]],
                        fill='toself',
                        fillcolor='rgba(0, 212, 255, 0.3)',
                        line=dict(color='#00d4ff', width=3),
                        name='Current Vitals'
                    ))
                    
                    # Optimal baseline
                    optimal = [75, 50, 75, 98]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=optimal + [optimal[0]],
                        theta=categories + [categories[0]],
                        line=dict(color='#00ff88', width=2, dash='dash'),
                        name='Optimal'
                    ))
                    
                    fig_radar.update_layout(
                        template="plotly_dark",
                        height=380,
                        polar=dict(
                            radialaxis=dict(
                                visible=True, 
                                range=[0, 150],
                                tickvals=[0, 50, 100, 150],
                                ticktext=['0', '50', '100', '150']
                            )
                        ),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key=generate_unique_key("radar_chart"))
            
            with analytics_col2:
                st.subheader("🤖 Latest AI Clinical Assessment")
                
                if st.session_state.ai_insights:
                    latest_insight = list(st.session_state.ai_insights)[-1]
                    insight_risk = latest_insight.get('risk_score', risk_score)
                    
                    urgency_badge = "🚨 CRITICAL" if "CRITICAL" in latest_insight['text'] else \
                                   "⚠️ URGENT" if "URGENT" in latest_insight['text'] else \
                                   "👁️ MONITOR" if "MONITOR" in latest_insight['text'] else \
                                   "✓ ROUTINE"
                    
                    st.markdown(f"""
                    <div class="ai-insight-box">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span style="color: #00d4ff; font-weight: bold; font-size: 14px;">{urgency_badge}</span>
                            <span style="color: #888; font-size: 12px;">{latest_insight['time']}</span>
                        </div>
                        <p style="font-size: 14px; line-height: 1.6; margin: 10px 0;">{latest_insight['text']}</p>
                        <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(0,212,255,0.3);">
                            <span style="color: #00d4ff; font-size: 12px;">Computed Risk: {insight_risk}/10</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show trend if available
                    if len(st.session_state.history) >= 5:
                        recent_risks = st.session_state.history['RiskScore'].tail(5).tolist()
                        risk_trend = "↗️ Increasing" if recent_risks[-1] > recent_risks[0] else \
                                    "↘️ Decreasing" if recent_risks[-1] < recent_risks[0] else \
                                    "→ Stable"
                        
                        trend_color = "#ff3333" if "Increasing" in risk_trend else "#00ff88" if "Decreasing" in risk_trend else "#ffaa00"
                        
                        st.markdown(f"""
                        <div class="stat-highlight">
                            <strong>5-Reading Trend:</strong> <span style="color: {trend_color};">{risk_trend}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("🔄 AI analysis initializing... First assessment in {ai_interval - time_since_last_ai} seconds")
            
            with analytics_col3:
                st.subheader("📈 Session Stats")
                
                if len(st.session_state.history) >= 2:
                    # Calculate changes
                    hr_change = st.session_state.history['HR'].iloc[-1] - st.session_state.history['HR'].iloc[-2]
                    temp_change = st.session_state.history['Temp'].iloc[-1] - st.session_state.history['Temp'].iloc[-2]
                    bp_change = st.session_state.history['Systolic'].iloc[-1] - st.session_state.history['Systolic'].iloc[-2]
                    
                    st.metric("ΔHR", f"{hr_change:+.1f}", "BPM", delta_color="off")
                    st.metric("ΔTemp", f"{temp_change:+.2f}", "°F", delta_color="off")
                    st.metric("ΔBP", f"{bp_change:+.0f}", "mmHg", delta_color="off")
                    
                    # Averages
                    st.write("**10-Min Averages:**")
                    avg_hr = st.session_state.history['HR'].tail(10).mean()
                    avg_temp = st.session_state.history['Temp'].tail(10).mean()
                    avg_risk = st.session_state.history['RiskScore'].tail(10).mean()
                    
                    st.markdown(f"""
                    <div class="stat-highlight">
                        HR: {avg_hr:.1f} BPM<br>
                        Temp: {avg_temp:.1f} °F<br>
                        Risk: {avg_risk:.1f}/10
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Data quality
                    data_points = len(st.session_state.history)
                    st.metric("Data Points", data_points)
                else:
                    st.info("Collecting data...")
        
        time.sleep(refresh_rate)
        
    except requests.exceptions.Timeout:
        with placeholder.container():
            st.error("🔌 Connection Timeout: Sensor not responding")
            st.info("Retrying in 3 seconds...")
            st.caption("Check network connection and sensor status")
        time.sleep(3)
        
    except requests.exceptions.ConnectionError:
        with placeholder.container():
            st.error("🔌 Connection Error: Unable to reach remote sensor")
            st.info("Attempting to reconnect...")
            st.caption(f"Target: {NGROK_URL}")
        time.sleep(3)
        
    except requests.exceptions.RequestException as e:
        with placeholder.container():
            st.error(f"🔌 Network Error: {type(e).__name__}")
            st.info("Attempting to reconnect...")
            st.caption(f"Details: {str(e)}")
        time.sleep(3)
        
    except KeyError as e:
        with placeholder.container():
            st.error(f"📊 Data Format Error: Missing field {str(e)}")
            st.info("Sensor may be sending incomplete data")
        time.sleep(2)
        
    except Exception as e:
        with placeholder.container():
            st.error(f"⚠️ System Error: {type(e).__name__}")
            st.warning(f"Details: {str(e)}")
            st.info("Dashboard will retry in 2 seconds...")
            if st.button("Force Reset", key=f"force_reset_{int(time.time())}"):
                st.rerun()
        time.sleep(2)