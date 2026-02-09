🩺 VitalGuard AI - Real-Time Patient Monitoring Dashboard
Professional ICU-grade remote patient monitoring system with AI-powered insights
📋 Overview
VitalGuard AI is a sophisticated real-time patient monitoring dashboard that provides:

Live vital signs tracking (Heart Rate, Temperature, Blood Pressure, SpO2)
AI-powered clinical insights using Google's Gemini 1.5 Flash
Automated risk assessment with color-coded alerts
Interactive trend visualizations with 6 comprehensive graphs
Multi-vital correlation analysis for pattern recognition
Real-time alert system with 3-tier severity levels

Perfect for remote patient monitoring, telehealth applications, and ICU observation systems.

🚀 Features
Real-Time Monitoring

✅ Live data streaming from remote sensors via ngrok tunnel
✅ 1-10 second adjustable refresh rates
✅ Automatic data validation and error handling
✅ Connection status monitoring

Advanced Visualizations

Heart Rate Trend - Line graph with normal range shading (60-100 BPM)
Temperature Trend - Thermal monitoring with fever thresholds
Blood Pressure Dual-Line - Systolic/Diastolic tracking
Risk Score Timeline - Color-coded severity zones
Multi-Vital Radar Chart - Comparative analysis across all metrics
Statistical Summaries - 10-minute averages and trend analysis

AI Clinical Assistant

🤖 Gemini 1.5 Flash integration for real-time analysis
🧠 Automatic anomaly detection
📊 Clinical interpretation of vital patterns
💡 Nursing action recommendations
🕐 30-second update intervals

Smart Alert System

🚨 Critical Alerts - Severe tachycardia (HR > 120), High fever (> 102°F), Hypotension
⚠️ Warning Alerts - Mild tachycardia/bradycardia, Fever, Hypertension
ℹ️ Info Alerts - General status updates

Risk Scoring
Automated 0-10 risk calculation based on:

Heart rate deviations
Temperature abnormalities
Blood pressure extremes
Combined vital trends


🛠️ Installation
Prerequisites
bashPython 3.8 or higher
pip (Python package manager)
Google Gemini API key
ngrok account (for remote sensor tunneling)
Step 1: Clone or Download
bash# If using git
git clone https://github.com/yourusername/vitalguard-ai.git
cd vitalguard-ai

# Or download the .py file directly
Step 2: Install Dependencies
bashpip install streamlit requests pandas plotly google-generativeai numpy
Step 3: Configure API Keys
Option A: Direct in Code (Quick Start)
python# Line 8 in the code
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")
Option B: Environment Variables (Recommended)
bash# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Update code to use:
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
Step 4: Set Up Data Source
Update the ngrok URL (Line 7):
pythonNGROK_URL = "https://your-ngrok-url.ngrok-free.dev/latest"

🎯 Usage
Quick Start
bashstreamlit run vitalguard_ai.py
The dashboard will open in your browser at http://localhost:8501
Expected Data Format
Your endpoint should return JSON in this format:
json{
  "heart_rate": 75.5,
  "body_temperature": 98.6,
  "systolic_bp": 120,
  "diastolic_bp": 80
}
Dashboard Controls
Sidebar Options:

Refresh Rate Slider - Adjust update frequency (1-10 seconds)
Clinical Alerts - View recent alerts by severity
AI Insights History - Past 5 AI assessments

Main Dashboard:

Top Row - Real-time vital metrics with status indicators
Middle Rows - 4 trend graphs showing historical patterns
Bottom Row - Radar chart + AI assessment panel


📊 Understanding the Metrics
Heart Rate (BPM)

Normal: 60-100 BPM (green zone)
Bradycardia: < 60 BPM (warning)
Tachycardia: > 100 BPM (warning/critical)

Temperature (°F)

Normal: 97-100°F (green zone)
Fever: > 100.4°F (warning)
High Fever: > 102°F (critical)

Blood Pressure (mmHg)

Normal: 90-120 / 60-80
Hypertension: > 140/90
Hypotension: < 90/60

Risk Score (0-10)

Low Risk (0-2): Green - No immediate concerns
Moderate Risk (3-5): Yellow - Enhanced monitoring
High Risk (6-10): Red - Immediate attention required


🔧 Customization
Modify Alert Thresholds
python# Line 90-100 - Edit these values
if hr > 120:  # Change tachycardia threshold
    st.session_state.alerts.append(...)

if temp > 100.4:  # Change fever threshold
    st.session_state.alerts.append(...)
Adjust Data History
python# Line 85 - Change number of stored readings
.tail(100)  # Change to 50, 200, etc.
Change AI Update Frequency
python# Line 103 - Modify seconds between AI calls
if (now - st.session_state.last_ai_call).seconds >= 30:  # Change 30 to desired interval
Customize Color Scheme
python# Line 50-70 - Edit CSS colors
border-left: 4px solid #00d4ff;  # Change accent color
background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);  # Change background

🐛 Troubleshooting
Issue: "Connection Error: Unable to reach remote sensor"
Solution:

Verify ngrok URL is correct and active
Check if sensor endpoint is running
Test URL in browser: https://your-url.ngrok-free.dev/latest

Issue: "AI analysis unavailable"
Solution:

Verify Gemini API key is valid
Check API quota at Google AI Studio
Ensure google-generativeai package is installed

Issue: Graphs not displaying
Solution:

Ensure plotly is installed: pip install plotly
Clear browser cache
Check browser console for errors

Issue: Dashboard freezes
Solution:

Reduce refresh rate in sidebar
Check if endpoint is responding slowly
Restart Streamlit: Ctrl+C then rerun


📁 Project Structure
vitalguard-ai/
├── vitalguard_ai.py          # Main dashboard application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env (optional)           # Environment variables
└── .gitignore               # Git ignore file

📦 Dependencies
txtstreamlit>=1.28.0
requests>=2.31.0
pandas>=2.0.0
plotly>=5.17.0
google-generativeai>=0.3.0
numpy>=1.24.0

🔐 Security Considerations
API Key Protection

⚠️ Never commit API keys to Git
✅ Use environment variables
✅ Add .env to .gitignore

Data Privacy

Patient data is stored only in session memory
No data persistence between sessions
No external data transmission except to Gemini API
Consider HIPAA compliance for production use

Network Security

ngrok provides HTTPS encryption
Validate all incoming data
Implement authentication for production deployment


🚀 Deployment
Local Network
bashstreamlit run vitalguard_ai.py --server.port 8501 --server.address 0.0.0.0
Streamlit Cloud

Push code to GitHub
Visit share.streamlit.io
Connect repository
Add secrets in dashboard settings

Docker
dockerfileFROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "vitalguard_ai.py"]

🎓 Use Cases

Telehealth - Remote patient monitoring for home care
ICU Observation - Real-time vital tracking for critical patients
Research - Clinical trial monitoring and data collection
Medical Training - Student simulation and education
Emergency Response - Field monitoring for paramedics



