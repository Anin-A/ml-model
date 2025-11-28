from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Inisialisasi FastAPI
app = FastAPI(title="ML API with FastAPI")

# Load model
model = load("model.joblib")

# Label encoder
encoder = LabelEncoder()
encoder.fit(["Balanced Learner", "Consistent Learner", "Fast Learner", "Reflective Learner"])

# Schema input menggunakan Pydantic
class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: Features):
    # Prediksi dengan model
    prediction = model.predict([data.features])
    
    # Inverse encoding
    decoded = encoder.inverse_transform(prediction)
    
    return {
        "prediction_numeric": prediction.tolist(),
        "prediction_label": decoded.tolist()
    }
