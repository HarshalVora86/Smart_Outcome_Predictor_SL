from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎓")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_bundle():
    with open(BASE_DIR / "model/smart_outcome_predictor_pipeline.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics():
    return json.loads((BASE_DIR / "model/metrics.json").read_text())


bundle = load_bundle()
m = load_metrics()
clf_model = bundle["classifier"]
reg_model = bundle["regressor"]

st.title("🎓 Smart Outcome Predictor")
st.caption(
    "Stacking Ensembles • Course Completion (Classification) • Final Score (Regression)")

with st.form("student"):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", 15, 60, 24)
        week_of_year = st.number_input("Course Start Week (1-52)", 1, 52, 26)
        sessions = st.number_input("Sessions", 0, 200, 17)
        time_spent_hours = st.number_input(
            "Time Spent (hours)", 0.0, 200.0, 16.0, 0.5)
        videos_watched = st.number_input("Videos Watched", 0, 300, 23)
        quiz_attempts = st.number_input("Quiz Attempts", 0, 30, 7)
        assignments_submitted = st.number_input(
            "Assignments Submitted", 0, 20, 4)
        forum_posts = st.number_input("Forum Posts", 0, 20, 1)

    with c2:
        avg_quiz_score = st.number_input(
            "Average Quiz Score (0-100)", 0.0, 100.0, 67.0, 1.0)
        attendance_rate = st.number_input(
            "Attendance Rate (0-1)", 0.0, 1.0, 0.85, 0.01)
        country_region = st.selectbox(
            "Country / Region", ["Africa", "Americas", "Asia", "Europe", "Oceania"], index=2)
        device_type = st.selectbox(
            "Device Type", ["Laptop", "Mobile", "Tablet"])
        education_background = st.selectbox("Education Background", [
                                            "HighSchool", "Undergrad", "Graduate", "WorkingPro"], index=1)
        course_level = st.selectbox(
            "Course Level", ["Beginner", "Intermediate", "Advanced"])
        course_category = st.selectbox(
            "Course Category", ["Business", "Data", "Design", "Marketing", "Programming"])

    submit = st.form_submit_button("🔮 Predict Outcome")

if submit:
    x = pd.DataFrame({
        "age": [age],
        "week_of_year": [week_of_year],
        "sessions": [sessions],
        "time_spent_hours": [time_spent_hours],
        "videos_watched": [videos_watched],
        "quiz_attempts": [quiz_attempts],
        "assignments_submitted": [assignments_submitted],
        "forum_posts": [forum_posts],
        "avg_quiz_score": [avg_quiz_score],
        "attendance_rate": [attendance_rate],
        "course_level": [course_level],
        "country_region": [country_region],
        "device_type": [device_type],
        "education_background": [education_background],
        "course_category": [course_category],
    })[bundle["feature_columns"]]

    # Same outlier capping as training (bounds learned from the training data)
    for col, (lower, upper) in bundle["cap_bounds"].items():
        x[col] = x[col].clip(lower, upper)

    p = float(clf_model.predict_proba(x)[0, 1])
    pred = int(clf_model.predict(x)[0])
    score = float(np.clip(reg_model.predict(x)[0], 0, 100))

    st.divider()

    if pred:
        st.success("✅ Prediction: STUDENT LIKELY TO COMPLETE THE COURSE")
    else:
        st.error("❌ Prediction: STUDENT LIKELY NOT TO COMPLETE THE COURSE")

    st.metric("Completion Probability", f"{p:.1%}")
    st.progress(p)

    st.metric("Predicted Final Score", f"{score:.1f} / 100")
    st.caption(
        f"Typical error of this score estimate is about ±{m['regression']['mae']:.0f} points (test MAE).")

    st.dataframe(x)

with st.expander("📊 Model Evaluation"):
    clf_m = m["classification"]
    reg_m = m["regression"]

    st.subheader("Classification: Stacking Classifier")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{clf_m['accuracy']:.1%}")
    b.metric("Precision", f"{clf_m['precision']:.1%}")
    c.metric("Recall", f"{clf_m['recall']:.1%}")
    d.metric("F1", f"{clf_m['f1_score']:.1%}")
    st.write(f"ROC-AUC: **{clf_m['roc_auc']:.3f}**")

    st.dataframe(pd.DataFrame(
        clf_m["confusion_matrix"],
        index=["Actual Not Completed", "Actual Completed"],
        columns=["Predicted Not Completed", "Predicted Completed"],
    ))

    st.subheader("Regression: Stacking Regressor")
    e, f, g = st.columns(3)
    e.metric("MAE", f"{reg_m['mae']:.2f}")
    f.metric("RMSE", f"{reg_m['rmse']:.2f}")
    g.metric("R²", f"{reg_m['r2']:.3f}")
