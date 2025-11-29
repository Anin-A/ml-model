from sklearn.preprocessing import LabelEncoder
from joblib import load

encoder = LabelEncoder()
y = ["Balanced Learner", "Consistent Learner", "Fast Learner", "Reflective Learner"]
encoder.fit(y)

# Load model
model = load("model.joblib")

# Input manual
features = [5, 1.5, 5, 0.5, 3, 0.37, 2, 85, 0.37]

# Prediksi
prediction = model.predict([features])
decoded = encoder.inverse_transform(prediction)

print("Hasil prediksi (numerik):", prediction)
print("Hasil prediksi (label):", decoded)
