import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

def show():
    # Model 1 metrics
    model_1_metrics = {
        "Accuracy": 74.58,
        "Precision": 0.63,
        "Recall": 0.93,
        "F1 Score": 0.75,
        "ROC AUC": 0.87
    }
    model_1_cm_path = os.path.join(os.path.dirname(__file__), "Model 1 Confusion matrix.png")
    # Model 2 metrics
    model_2_metrics = {
        "Accuracy": 77.12,
        "Precision": 0.66,
        "Recall": 0.95,
        "F1 Score": 0.78,
        "ROC AUC": 0.89
    }
    model_2_cm_path = os.path.join(os.path.dirname(__file__), "Model 2 confusion matrix.png")

    # Page Title
    st.title("Model Evaluation Metrics")

    # Display Metrics Side by Side
    st.subheader("Model Performance Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model 1")
        for metric, value in model_1_metrics.items():
            st.markdown(f"- **{metric}**: {value:.2f}")
        st.image(model_1_cm_path, caption="Model 1 Confusion Matrix")

    with col2:
        st.markdown("### Model 2")
        for metric, value in model_2_metrics.items():
            st.markdown(f"- **{metric}**: {value:.2f}")
        st.image(model_2_cm_path, caption="Model 2 Confusion Matrix")
