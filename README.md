# 🧠 Human Activity Recognition (Walk vs Run) — Machine Learning & Deep Learning
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

This project focuses on classifying **human activities (walking vs running)** using **accelerometer and gyroscope sensor data**.  

It demonstrates an end-to-end **data science pipeline** — from data cleaning, EDA, and feature engineering to building, evaluating, and saving **Machine Learning (ML)** and **Deep Learning (DL)** models.  

The final **LSTM (Long Short-Term Memory)** model achieved nearly **99.9% accuracy**, outperforming traditional ML methods and proving its effectiveness in modeling temporal dependencies in sensor data.

---

## 🔹 Impact Summary
- 🚀 **99.9% Accuracy** achieved using LSTM sequence modeling  
- 🤖 Combined **Machine Learning** and **Deep Learning** pipelines for comparison  
- 🩺 Applicable to **wearable tech, healthcare monitoring, and IoT safety systems**  
- 💾 Deployment-ready models (`.joblib`, `.h5`) stored in `output models/`  

---
## 🚀 Key Highlights
- 🔍 Conducted **comprehensive EDA** on multi-axis time-series data to uncover distinct walking and running signal patterns.  
- 🧹 Engineered **magnitude-based features** (`accel_mag`, `gyro_mag`), boosting model accuracy by ~4%.  
- ⚙️ Designed a **unified ML/DL training pipeline** comparing Logistic Regression, Random Forest, MLP, and LSTM.  
- 📈 Achieved **99.9% test accuracy** with the LSTM model by effectively capturing temporal dependencies.  
- 🧠 Evaluated using **Precision, Recall, F1, ROC-AUC**, confirming strong model generalization.  
- 💾 Saved trained models in `.joblib` and `.h5` formats for reproducible deployment.  
- 🧩 Framework easily extendable to multi-class HAR tasks (sitting, climbing, jumping).  
- 🏭 Demonstrated **real-world impact** in healthcare, wearables, and industrial monitoring.  

---

## 🧩 Problem Statement
In wearable and smart health applications, devices continuously record motion data from sensors like accelerometers and gyroscopes.  

The challenge is to automatically **differentiate activities** (e.g., walking vs running) based on subtle differences in motion patterns.

**Goal:**  
Develop a robust classification system capable of detecting whether a person is *walking* or *running* using only raw 3-axis sensor data, while ensuring high generalization across users and devices.

---



## 📁 **Project Structure**
```
├── data
│ └── walkrun.csv
│
├── notebook
│ └── WalkRunClass_LSTM_completed.ipynb
│
├── output models
│ ├── scaler.joblib
│ ├── random_forest.joblib
│ ├── mlp_model.h5
│ └── LSTM_Model.h5
│
├── .gitignore
└── README.md
```

## 📊 Data Understanding

The dataset contains timestamped **accelerometer** and **gyroscope** readings across three axes (`x`, `y`, `z`) for two activities — *walking* and *running*.

**Key Observations from EDA:**
- Running shows higher acceleration and gyroscope variance.
- Strong correlation between `accel_y` and `gyro_y`, representing forward motion.
- The dataset is balanced: ≈50% walk, 50% run.

---

## ⚙️ Feature Engineering

| Feature | Description | Purpose |
|----------|--------------|----------|
| `accel_mag` | √(x² + y² + z²) | Total acceleration magnitude |
| `gyro_mag`  | √(x² + y² + z²) | Total angular velocity magnitude |
| Normalized features | StandardScaler | To stabilize training and improve convergence |

---

## 🤖 Modeling Approach

Four models were developed and compared:

| Model | Type | Framework | Key Strength |
|--------|-------|------------|---------------|
| Logistic Regression | ML | Scikit-learn | Baseline linear model |
| Random Forest | ML | Scikit-learn | Handles non-linear interactions |
| MLP (Neural Network) | DL | TensorFlow/Keras | Dense layers for pattern extraction |
| LSTM | DL | TensorFlow/Keras | Captures temporal dependencies |

---

## 📊 Model Comparison & Results

| Model              | Accuracy | Precision | Recall | F1 Score | AUC     |
|--------------------|-----------|------------|---------|-----------|----------|
| Logistic Regression | 0.9525 | 0.9687 | 0.9354 | 0.9517 | 0.9900 |
| Random Forest       | 0.9884 | 0.9936 | 0.9861 | 0.9884 | 0.9991 |
| LSTM (Deep Learning)| **0.9998** | **0.9998** | **0.9995** | **0.9992** | **0.9995** |
| MLP (Neural Net)    | 0.9747 | 0.9958 | 0.9676 | 0.9815 | 0.9963 |

✅ **Best Model:** LSTM — effectively captured sequential temporal dependencies in sensor data, leading to superior accuracy and generalization.

---

## 🔍 Interpretation & Insights
- **LSTM** learned temporal motion patterns more effectively than static ML models.  
- **Random Forest** provided explainable feature importance — `accel_y` & `gyro_y` dominate.  
- Consistently high **AUC (>0.99)** confirms strong class separation and reliability.

---

## 🧪 Experimental Learnings
- Feature engineering significantly impacts sensor data performance.  
- Deep learning outperforms ML for time-series due to sequence awareness.  
- Proper scaling and stratified splits stabilize generalization.  
- Random Forest remains a great interpretable baseline.

---

## 🏭 Business & Research Applications
- 🏃‍♂️ **Fitness Trackers:** Automatic activity recognition.  
- 🏥 **Healthcare:** Mobility monitoring & rehabilitation tracking.  
- 🏭 **Industrial Safety:** Detecting abnormal or hazardous motions.  
- 🧠 **Smart Devices:** Gesture recognition & IoT context-awareness.  

---

## 🧭 Future Enhancements
- Expand to **multi-class activities** (sitting, jumping, climbing).  
- Integrate **CNN-LSTM hybrid** for longer time windows.  
- Deploy real-time **Streamlit dashboard**.  
- Convert to **TensorFlow Lite** for mobile inference.  
- Serve predictions via **FastAPI REST API**.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python |
| ML Frameworks | Scikit-learn, TensorFlow, Keras |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib, H5 |
| Version Control | Git, GitHub |
| IDEs | Jupyter Notebook, VS Code |

---
## 🚀 How to Run Locally

**Clone the Repository**
```bash
git clone https://github.com/Bharath-19/Human-Activity-Recognition-Walk-Run.git
cd Human-Activity-Recognition-Walk-Run
```
## 👨‍💻 Author
[![LinkedIn Badge](https://img.shields.io/badge/LinkedIn-blue)](https://www.linkedin.com/in/bharath-budhi-45b701184)
[![GitHub Badge](https://img.shields.io/badge/GitHub-black)](https://github.com/Bharath-19)

**Veera Bharath Chandra Budhi**  
🎓 Master’s in Autonomy Technologies — Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)  
💼 Data Scientist 

📫 **Contact:**  
- 📧 veerabharathchandrabudhi@gmail.com  

💬 **Data Scientist skilled in building end-to-end machine learning and deep learning solutions — from data analysis and feature engineering to model deployment and visualization for real-world decision making.**

---

⭐ **If you found this project helpful, please star the repository!**
