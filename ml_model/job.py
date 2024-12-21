import joblib
import os

MODEL_PATH = r"C:\Users\spand\OneDrive\Desktop\Minor_project_bite\ml_model\model.joblib"

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {MODEL_PATH}")



