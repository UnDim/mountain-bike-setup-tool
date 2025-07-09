import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Load trained models
@st.cache_resource
def load_models():
    targets = [
        "Fork Pressure (psi)",
        "Front Tire Pressure (psi)",
        "Rear Tire Pressure (psi)",
        "Handlebar Width (mm)",
        "Estimated Saddle Height (in)"
    ]
    model_paths = {
        target: f"models/model_{target.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.joblib"
        for target in targets
    }
    models = {target: joblib.load(path) for target, path in model_paths.items()}
    return models

models = load_models()

st.title("🚵‍♂️ Hardtail MTB Setup Recommender")
st.write("Get setup recommendations based on your body and riding preferences.")

# Input fields
height = st.slider("Height (inches)", 60, 78, 68)
weight = st.slider("Weight (lbs)", 120, 240, 170)
style = st.selectbox("Riding Style", ["XC", "Trail", "Downcountry"])
terrain = st.selectbox("Terrain Type", ["Flowy", "Rocky", "Technical"])
skill = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])

# Create input DataFrame
input_df = pd.DataFrame.from_dict([{
    "Height (in)": height,
    "Weight (lbs)": weight,
    "Riding Style": style,
    "Terrain Type": terrain,
    "Skill Level": skill
}])

# Predict values using models
st.subheader("📋 Recommended Setup (ML-Powered)")
recommendations = {}
for label, model in models.items():
    prediction = model.predict(input_df)[0]
    unit = "psi" if "Pressure" in label else ("mm" if "Handlebar" in label else "in")
    recommendations[label] = round(prediction, 1)
    st.write(f"**{label}:** {recommendations[label]} {unit}")

# Feedback collection
st.subheader("📝 Feedback")
with st.form("feedback_form"):
    feedback_rating = st.radio("How did the setup feel?", ["👍 Good", "👎 Needs Adjustment"])
    feedback_notes = st.text_area("Additional comments (optional)")
    submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        feedback_data = input_df.copy()
        for key, value in recommendations.items():
            feedback_data[key] = value
        feedback_data["Rating"] = feedback_rating
        feedback_data["Notes"] = feedback_notes
        feedback_data["Timestamp"] = datetime.now().isoformat()

        # Save feedback to CSV
        feedback_file = "user_feedback.csv"
        if os.path.exists(feedback_file):
            old_data = pd.read_csv(feedback_file)
            updated_data = pd.concat([old_data, feedback_data], ignore_index=True)
        else:
            updated_data = feedback_data
        updated_data.to_csv(feedback_file, index=False)
        st.success("✅ Feedback submitted. Thank you!")