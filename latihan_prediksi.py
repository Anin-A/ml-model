from sklearn.preprocessing import LabelEncoder
from joblib import load

encoder = LabelEncoder()
y = ["Balanced Learner", "Consistent Learner", "Fast Learner", "Reflective Learner"]
encoder.fit(y)

# Load model
model = load("model.joblib")

# Input manual
features = [5, 1.5, 4, 64, 2, 45, 3, 81, 30]

# Prediksi
prediction = model.predict([features])
decoded = encoder.inverse_transform(prediction)

print("Hasil prediksi (numerik):", prediction)
print("Hasil prediksi (label):", decoded)
