import requests
import json
import time
import os

# 1. Verify File Exists
FILE_PATH = r"C:\Users\Prajwal\Downloads\patient_data.json"
if not os.path.exists(FILE_PATH):
    print(f"CRITICAL ERROR: The file was not found at {FILE_PATH}")
    exit()

# 2. Load Data
print("Loading JSON data...")
with open(FILE_PATH, 'r') as f:
    all_records = json.load(f)
print(f"Total records found in file: {len(all_records)}")

# 3. Filter for Patient 1
# Note: Check if patient_id in your JSON is a string "1" or integer 1
patient_1_records = [r for r in all_records if str(r.get("patient_id")) == "1"]
print(f"Filtered records for Patient 1: {len(patient_1_records)}")

if len(patient_1_records) == 0:
    print("ERROR: No records found for Patient ID 1. Check the ID in your JSON file.")
    exit()

# 4. Sorting
patient_1_records.sort(key=lambda x: x["timestamp"])

# 5. The Stream Loop
API_URL = "http://127.0.0.1:8000/update"
print("Starting live data stream...")

for record in patient_1_records:
    try:
        # Convert types to ensure FastAPI accepts them
        payload = {
            "patient_id": int(record["patient_id"]),
            "timestamp": str(record["timestamp"]),
            "heart_rate": float(record["heart_rate"]),
            "body_temperature": float(record["body_temperature"]),
            "systolic_bp": float(record["systolic_bp"]),
            "diastolic_bp": float(record["diastolic_bp"])
        }
        
        response = requests.post(API_URL, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"SENT: Time={payload['timestamp']} | HR={payload['heart_rate']}")
        else:
            print(f"SERVER ERROR: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"CONNECTION ERROR: {e}")
    
    time.sleep(1)