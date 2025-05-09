import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import plotly.express as px

st.title("Heart Health Risk Estimator")

tabs = st.tabs(['Single Prediction', 'CSV Upload', 'Model Performance'])


with tabs[0]:
    st.header("Enter Patient Information")


    age = st.number_input("Age", 0, 150)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cp_type = st.selectbox("Chest Pain", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    bp = st.number_input("Resting BP (mm Hg)", 0, 300)
    chol = st.number_input("Cholesterol (mg/dl)", 0)
    sugar = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
    max_hr = st.number_input("Max Heart Rate", 60, 202)
    angina = st.selectbox("Exercise-induced Angina", ["Yes", "No"])
    st_depression = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0)
    slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])


    gender = 0 if gender == "Male" else 1
    cp_type = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp_type)
    sugar = 1 if sugar == "> 120 mg/dl" else 0
    ecg = ["Normal", "ST-T Abnormality", "LV Hypertrophy"].index(ecg)
    angina = 1 if angina == "Yes" else 0
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope)


    user_input = pd.DataFrame({
        'Age': [age],
        'Sex': [gender],
        'ChestPainType': [cp_type],
        'RestingBP': [bp],
        'Cholesterol': [chol],
        'FastingBS': [sugar],
        'RestingECG': [ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [angina],
        'Oldpeak': [st_depression],
        'ST_Slope': [slope]
    })

    model_files = ['DecisionTree.pkl', 'LogisticR.pkl', 'RandomForest.pkl', 'SVM.pkl', 'gridrf.pkl']
    model_names = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'SVM', 'Grid-Optimized RF']

    def run_all_models(input_df):
        results = []
        for model_path in model_files:
            model = pickle.load(open(model_path, 'rb'))
            pred = model.predict(input_df)
            results.append(pred[0])
        return results

    if st.button("Predict"):
        st.subheader("Prediction Results")
        predictions = run_all_models(user_input)
        for name, result in zip(model_names, predictions):
            st.markdown(f"**{name}**: {'💓 Heart Disease Detected' if result else '✅ No Heart Disease'}")

with tabs[1]:
    st.header("Bulk Prediction via CSV Upload")

    st.markdown("""
    **CSV Format Guidelines**  
    • No missing values  
    • Required Columns: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`,  
      `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`  
    • Use proper encodings:  
      - Sex: `M`/`F`  
      - ChestPainType: `TA`, `ATA`, `NAP`, `ASY`  
      - RestingECG: `Normal`, `ST`, `LVH`  
      - ExerciseAngina: `Y`, `N`  
      - ST_Slope: `Up`, `Flat`, `Down`  
    """)

    file = st.file_uploader("Upload your dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)
        required_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS',
                         'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

        if set(required_cols).issubset(df.columns):
            def encode_bulk_data(data):
                maps = {
                    'Sex': {'M': 0, 'F': 1},
                    'ChestPainType': {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3},
                    'RestingECG': {'Normal': 0, 'ST': 1, 'LVH': 2},
                    'ExerciseAngina': {'Y': 1, 'N': 0},
                    'ST_Slope': {'Up': 0, 'Flat': 1, 'Down': 2}
                }
                for col, mapping in maps.items():
                    data[col] = data[col].map(mapping)
                return data

            df_encoded = encode_bulk_data(df.copy())
            model = pickle.load(open('LogisticR.pkl', 'rb'))
            df_encoded['Prediction'] = model.predict(df_encoded.values).astype(int)

            st.success("Predictions Completed")
            st.dataframe(df_encoded)

            def download_csv(data):
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                return f'<a href="data:file/csv;base64,{b64}" download="Heart_Predictions.csv">Download Results</a>'

            st.markdown(download_csv(df_encoded), unsafe_allow_html=True)
        else:
            st.warning("Uploaded file missing required columns.")
    else:
        st.info("Awaiting CSV file...")

with tabs[2]:
    st.header("Model Accuracy Comparison")

    scores = {
        'Decision Tree': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'SVM': 84.22,
        'Grid-Optimized RF': 89.75
    }

    accuracy_df = pd.DataFrame(scores.items(), columns=["Model", "Accuracy"])
    chart = px.bar(accuracy_df, x='Model', y='Accuracy', color='Model', title="Accuracy by Algorithm")
    st.plotly_chart(chart)