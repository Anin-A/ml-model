# Proyek Analisis Performa Belajar Siswa

## 📌 Tujuan
Tujuan dari proyek ini adalah untuk membuat sebuah model yang dapat mengidentifikasi pola belajar siswa dan menyajikan insight mengenai performa siswa.  
Dataset yang digunakan adalah **dataset sintesis mandiri** dengan izin dari Admin Capstone Project.

---

## 🎯 Kategori Pola Belajar Siswa
1. **Consistent Learner**  
   - Ritme belajar stabil dan rutin  
   - Frekuensi tinggi, durasi moderat, pengulangan konsisten  
   - Nilai kuis cenderung stabil (75–85%)  
   - Cocok untuk pembelajaran jangka panjang  

2. **Fast Learner**  
   - Cepat memahami materi dengan durasi singkat  
   - Jarang melakukan pengulangan  
   - Nilai kuis tinggi (85–95%)  
   - Efisien, namun kurang reflektif  

3. **Reflective Learner**  
   - Belajar mendalam dengan durasi panjang  
   - Sering mengulang materi (≥4 kali/minggu)  
   - Nilai kuis tinggi (80–95%)  
   - Cocok untuk bidang analitis  

4. **Balanced Learner**  
   - Menyeimbangkan kecepatan dan kedalaman belajar  
   - Frekuensi, durasi, dan pengulangan moderat  
   - Nilai kuis stabil (80–90%)  
   - Fleksibel dan adaptif  

---

## 📊 Dataset Sintetis
- **Jumlah sampel:** 2000  
- **Distribusi kelas:** Seimbang (500 sampel per label)  
- **Aturan generatif per label:**  
  - Consistent Learner → Materi 3–5, frekuensi 5–7 hari/minggu, durasi belajar 60–120 menit  
  - Fast Learner → Materi ≥6, kecepatan 1.5–2.0 materi/jam, pengulangan 0–1 kali/minggu  
  - Reflective Learner → Materi 2–4, durasi belajar 180–240 menit, pengulangan ≥4 kali/minggu  
  - Balanced Learner → Materi 4–6, durasi belajar 120–180 menit, pengulangan 2–3 kali/minggu  

---

## 🤖 Model Machine Learning
Model ini menjadi dasar dari aplikasi **CerdasKu**, sebuah platform asesmen pendidikan berbasis Web dan AI.  
- **Tujuan:** Mengatasi masalah *Cold Start* dalam personalisasi pembelajaran  
- **Fitur utama:** Pretest untuk analisis gaya belajar (Visual, Auditori, Kinestetik) dan pola belajar (Consistent, Fast, Reflective, Balanced)  
- **Algoritma:** Random Forest (Scikit-learn)  

---

## 🛠️ Teknologi yang Digunakan
- **Machine Learning:** Scikit-learn (Random Forest)  
- **API Framework:** FastAPI (Python)  
- **Deployment:** Heroku / Cloud Run  
- **Frontend & Backend:** Integrasi interaktif dengan layanan backend yang kuat  

---

## 👥 Tim
Proyek ini merupakan bagian dari **Capstone Project (Tim A25-CS225 - DC-08)** yang mengintegrasikan:  
- Backend yang kuat  
- Frontend interaktif  
- Model Machine Learning untuk memberikan insight yang dipersonalisasi  

---

## 📌 Insight
Dashboard dan model memberikan rekomendasi strategi belajar sesuai kategori siswa, membantu analisis performa serta mendukung personalisasi pembelajaran.
