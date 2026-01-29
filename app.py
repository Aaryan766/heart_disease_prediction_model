import streamlit as st
import pandas as pd
import pickle
import random
import time

st.header("Heart Disease Prediction Using Machine Learning")

data = """Heart Disease Prediction using Machine Learning Heart disease prevention is critical, and data-driven prediction systems can significantly aid in early diagnosis and treatment. Machine Learning offers accurate prediction capabilities, enhancing healthcare outcomes. In this project, I analyzed a heart disease dataset with appropriate preprocessing. Multiple classification algorithms were implemented in Python using Scikit-learn and Keras to predict the presence of heart disease.

Algorithms Used:

**Logistic Regression**

**Naive Bayes**

**Support Vector Machine (Linear)**

**K-Nearest Neighbors**

**Decision Tree**

**Random Forest**

**XGBoost**

**Artificial Neural Network (1 Hidden Layer, Keras)**
"""
st.markdown(data)

st.image(
    "https://images.ctfassets.net/ut7rzv8yehpf/1DhC3uX3EeKnjU02LWyTXH/9c82e6ae82662ed5903eafb40d888d90/8_Main_Types_of_Heart_Disease.jpg?w=1800&h=900&fl=progressive&q=50&fm=jpg"
)

# ---------- Cache heavy loads so they don't rerun every time ----------
@st.cache_resource
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data(url: str):
    return pd.read_csv(url)

model = load_model("heart_disease_pred.pkl")

url = "https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true"
df = load_data(url)

st.sidebar.header("Select Features to Predict Heart Disease")
st.sidebar.image(
    "https://humanbiomedia.org/animations/circulatory-system/cardiac-cycle/heart-beating.gif"
)


with st.sidebar.form("predict_form"):
    all_values = []

    for col in df.columns[:-1]:
        mn, mx = map(int, df[col].agg(["min", "max"]))
        key = f"feat_{col}"


        if key not in st.session_state:
            st.session_state[key] = random.randint(mn, mx)

        val = st.slider(
            f"Select {col} value",
            mn,
            mx,
            key=key,  
        )
        all_values.append(val)

    submitted = st.form_submit_button("Predict Heart Disease")

# ---------- Prediction ----------
if submitted:
    final_value = [all_values]
    ans = model.predict(final_value)[0]

    progress_bar = st.progress(0)
    placeholder = st.empty()
    placeholder.subheader("Predicting Heart Disease")
    st.image("https://i.makeagif.com/media/1-17-2024/dw-jXM.gif", width=200)

    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    placeholder.empty()
    progress_bar.progress(0)

    if ans == 0:
        st.success("No Heart Disease Detected")
    else:
        st.warning("Heart Disease Found")

st.markdown('Designed by: Aaryan Bhardwaj')        

