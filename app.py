import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_diabetes_model.pkl")  # Ensure file is in the same folder

model = load_model()

# ----------------------------
# Header Section
# ----------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 38px;
    color: #2C3E50;
    font-weight: bold;
}
.sub-title {
    text-align: center;
    font-size: 16px;
    color: #34495E;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Smaller header image
st.image(
    "https://cdn.analyticsvidhya.com/wp-content/uploads/2022/01/Diabetes-Prediction-Using-Machine-Learning.webp",
    use_container_width=False,
    width=1200,
)
st.markdown('<div class="main-title">🩺 Diabetes Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict the likelihood of diabetes using patient health details</div>', unsafe_allow_html=True)

# ----------------------------
# Input Section
# ----------------------------
st.header("🔹 Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", 0, 20, 2)
    glucose = st.slider("Glucose Level (mg/dL)", 0, 300, 120)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 200, 80)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)

with col2:
    insulin = st.slider("Insulin Level (µU/mL)", 0, 900, 85)
    bmi = st.slider("BMI (Body Mass Index)", 0.0, 70.0, 26.0)
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age (years)", 1, 120, 33)

# ----------------------------
# Prepare Data
# ----------------------------
patient_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age
}
df = pd.DataFrame([patient_data])

# ----------------------------
# Sidebar Summary
# ----------------------------
st.sidebar.header("🧾 Patient Summary")
for key, value in patient_data.items():
    st.sidebar.markdown(f"**{key}:** {value}")

# ----------------------------
# Prediction Section
# ----------------------------
if st.button("🔍 Predict Diabetes Status"):
    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]
    confidence = prediction_proba[prediction]

    st.subheader("🧾 Prediction Result")

    # Split layout for text and chart
    left_col, right_col = st.columns([2, 1])  # Wider left, smaller right

    with left_col:
        if prediction == 1:
            st.error(f"⚠️ The patient is **likely to have diabetes.**")
            st.write(f"**Prediction Confidence:** {confidence * 100:.2f}%")
            st.info("### 🔴 Health Tips for Managing Diabetes")
            st.markdown("""
            - 🍎 **Eat healthy:** Focus on vegetables, fruits & whole grains.  
            - 🏃 **Exercise daily:** Try brisk walking or yoga for 30 minutes.  
            - 💧 **Stay hydrated** and avoid sugary drinks.  
            - 💊 **Monitor blood sugar** and take medicines regularly.  
            - 🚭 **Avoid smoking & limit alcohol.**
            """)
        else:
            st.success(f"✅ The patient is **not likely to have diabetes.**")
            st.write(f"**Prediction Confidence:** {confidence * 100:.2f}%")
            st.info("### 🟢 Tips to Stay Healthy")
            st.markdown("""
            - 🥗 **Maintain a balanced diet** with enough fiber.  
            - 🏃 **Stay active** — walk, jog, or do light exercises.  
            - 💧 **Drink plenty of water.**  
            - 🩺 **Go for routine checkups.**  
            - 😴 **Sleep well & manage stress.**
            """)

    with right_col:
        # ✅ Small pie chart on right
        fig, ax = plt.subplots(figsize=(1.8, 1.8))  # smaller chart
        colors = ['#32CD32', '#FF4C4C']  # Green = No, Red = Yes
        labels = ['No Diabetes', 'Diabetes']
        ax.pie(
            prediction_proba,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 6}
        )
        st.pyplot(fig)

    # ----------------------------
    # 📥 Download Patient Data Section
    # ----------------------------
    patient_data["Prediction"] = "Diabetes" if prediction == 1 else "No Diabetes"
    patient_data["Confidence (%)"] = round(confidence * 100, 2)

    df_download = pd.DataFrame([patient_data])
    csv = df_download.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Patient Report",
        data=csv,
        file_name="patient_report.csv",
        mime="text/csv"
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by **Dhaneshwari Bhatia** | Powered by Streamlit 💡")

