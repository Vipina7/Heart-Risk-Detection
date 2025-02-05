# Heart Risk Detection App

This is a **machine learning-powered web application** that predicts the risk of heart disease based on user-provided health parameters. The app uses a trained model to analyze input data and assess the likelihood of heart-related conditions.

## 🚀 Features

- **User-friendly Interface:** Built using Streamlit for easy data input and interpretation.
- **Machine Learning Model:** Utilizes a trained model to predict heart disease risk.
- **Real-Time Predictions:** Instant feedback based on user input.
- **Preprocessing Pipeline:** Ensures categorical and numerical inputs are handled correctly before making predictions.

## 📌 How It Works

1. Users enter health details like **age, gender, chest pain type, blood pressure, cholesterol levels, etc.**
2. The data is preprocessed and fed into a trained model.
3. The model predicts the likelihood of heart disease and displays the result.

## 🛠️ Installation

To run the app locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/Vipina7/HeartRiskDetection.git
cd HeartRiskDetection

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 🏥 Input Parameters

| Parameter               | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| Age                     | Patient's age                                                    |
| Gender                  | Male or Female                                                   |
| Origin                  | Country of origin (e.g., Cleveland, Hungary)                     |
| Chest Pain Type         | Type of chest pain (e.g., asymptomatic, non-anginal)             |
| Resting Blood Pressure  | Blood pressure level in mmHg                                     |
| Cholesterol Level       | Total cholesterol level                                          |
| Fasting Blood Sugar     | Indicates if fasting blood sugar is above 120 mg/dl (True/False) |
| Resting ECG             | Results from electrocardiogram test                              |
| Max Heart Rate Achieved | Maximum recorded heart rate                                      |
| Exercise Induced Angina | Presence of exercise-induced angina (True/False)                 |
| ST Depression           | Depression induced by exercise relative to rest                  |

## 📊 Dataset Description

The dataset used for training contains the following attributes:

- **Categorical Features:** Gender, Origin, Chest Pain Type, Resting ECG, Exercise Induced Angina
- **Numerical Features:** Age, Resting Blood Pressure, Cholesterol Level, Max Heart Rate Achieved, ST Depression
- **Target Variable:** Heart Disease (Binary: 0 = No Disease, 1 = Disease Present)

## 🔎 Model and Preprocessing

- **Preprocessor:**
  - Missing values handled appropriately.
  - Categorical variables encoded using **One-Hot Encoding**.
  - Numerical features standardized using **MinMaxScaler**.
- **Model Training:**
  - Evaluated multiple models including **Logistic Regression, Random Forest, and XGBoost**.
  - Selected the best-performing model based on **accuracy, precision, recall, and F1-score**.
  - **Final Model:** Trained using **XGBoost** due to its high accuracy.
- **Threshold:** If the model predicts a probability **> 50%**, the patient is considered at risk.

## 🎯 Example Output

- ✅ **Low Risk:** "The patient is unlikely to have a heart condition. However, regular check-ups are recommended."
- ⚠️ **High Risk:** "The patient may have a heart condition. Please consult a doctor for further evaluation."

## 📌 Folder Structure

```
HeartRiskDetection/
|---artifacts/
|   |--model.pkl
|   |--preprocessor.pkl
|   |--train.csv
|   |--test.csv
|   |--model_performance.csv
|___ Notebook/
|    |--Data/
|      |--heart-disease-data.csv
|    |--EDA_and_Model_Training.py
│── src/
|   |---components/
|   |    |---data_ingestion.py
|   |    |---data_transformation.py
|   |    |---model_trainer.py
│   ├── pipeline/
│   │   ├── predict_pipeline.py
│   ├── exception.py
|   |__ logger.py
|   |__ utils.py
│── app.py
│── requirements.txt
│── README.md
```
## 🤝 Contributing

Feel free to submit issues and pull requests to improve the project!

---

🔥 **Vipina Manjunatha** 🔥

