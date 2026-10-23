# app.py
# Streamlit dashboard for Music Streaming Churn Prediction

import os
import json
import pandas as pd
import joblib
import streamlit as st
from datetime import datetime, date

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

# Page configuration
st.set_page_config(
    page_title="Music Streaming Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# Page 1: Project Overview
# ------------------------------------------------------------
def show_overview():
    st.header("Music Streaming Subscription Churn Prediction")
    st.markdown("""
    ### Problem Statement
    Music streaming services struggle to retain subscribers who cancel their subscriptions.
    This project uses machine learning to predict which customers are **likely to churn**.

    ### About Churn Prediction
    Churn prediction helps music streaming companies:
    - Identify customers at risk of leaving BEFORE they cancel
    - Take proactive measures (discounts, personalized content, outreach)
    - Retain valuable subscribers and reduce revenue loss

    ### Project Information
    - **Dataset:** PlaylistPro Streaming Subscription Churn Model
    - **Target:** `churned` (1 = churned/left, 0 = retained)
    """)

    st.markdown("---")

    st.subheader("Dataset Overview")
    st.markdown("""
    - **Training Set:** ~125,000 customers
    - **Test Set:** 75,000 customers (for Kaggle submission predictions)
    - **Feature Types:** Numerical (age, usage patterns) and Categorical (location, subscription type)
    """)

    st.markdown("---")

    st.subheader("Selected Model")
    st.markdown("The best model was selected based on validation performance.")

    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        st.metric("Best Model", metadata.get("model_name", "N/A"))
        st.metric("Validation Accuracy", f"{metadata.get('validation_accuracy', 0):.2%}")
        st.metric("Validation Precision", f"{metadata.get('validation_precision', 0):.2%}")
        st.metric("Validation Recall", f"{metadata.get('validation_recall', 0):.2%}")
        st.metric("Validation F1-score", f"{metadata.get('validation_f1', 0):.2%}")

        st.markdown("---")
        st.markdown("### Model Selection Rationale")
        st.markdown("""
        The model with the highest F1-score was selected. F1-score provides the best balance
        between precision and recall for identifying churners.
        """)
    else:
        st.warning("Model has not been trained yet. Run `train_model.py` first.")

    st.markdown("---")
    st.info("**Note:** This is a prediction model, not a guarantee. Churn prediction helps businesses "
              "prioritize retention efforts, but actual customer behavior can change.")


# ------------------------------------------------------------
# Page 2: Predict Churn
# ------------------------------------------------------------
def show_predict_page():
    st.header("Predict Customer Churn")
    st.markdown("Enter customer details to predict churn probability.")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found! Please train the model first using `python train_model.py`.")
        return

    if not os.path.exists(METADATA_PATH):
        st.error("Model metadata not found! Please train the model first.")
        return

    # Load model with caching
    @st.cache_resource
    def load_model():
        return joblib.load(MODEL_PATH)

    model = load_model()

    # Load metadata
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    numerical_cols = metadata.get("numerical_cols", [])
    categorical_cols = metadata.get("categorical_cols", [])

    # Create input form
    st.subheader("Customer Information Input")

    input_data = {}

    # Numerical inputs with sensible defaults and bounds
    # Integer inputs for count-based features; float for continuous features
    int_count_features = [
        'num_subscription_pauses', 'weekly_songs_played', 'weekly_unique_songs',
        'num_favorite_artists', 'num_platform_friends', 'num_playlists_created',
        'num_shared_playlists', 'notifications_clicked'
    ]

    float_continuous_features = [
        'weekly_hours', 'average_session_length', 'song_skip_rate'
    ]

    num_defaults = {
        'age': 35,
        'num_subscription_pauses': 0,
        'signup_date': date(2022, 1, 1),  # will convert to encoded days
        'weekly_hours': 5.0,
        'average_session_length': 20.0,
        'song_skip_rate': 0.2,
        'weekly_songs_played': 10,
        'weekly_unique_songs': 15,
        'num_favorite_artists': 5,
        'num_platform_friends': 10,
        'num_playlists_created': 5,
        'num_shared_playlists': 3,
        'notifications_clicked': 10,
    }

    num_bounds = {
        'age': (18, 80),
        'num_subscription_pauses': (0, 20),
        'signup_date': (date(2015, 1, 1), date(2022, 12, 31)),
        'weekly_hours': (0.5, 24.0),
        'average_session_length': (1.0, 100.0),
        'song_skip_rate': (0.0, 1.0),
        'weekly_songs_played': (1, 500),
        'weekly_unique_songs': (1, 100),
        'num_favorite_artists': (0, 50),
        'num_platform_friends': (0, 200),
        'num_playlists_created': (0, 100),
        'num_shared_playlists': (0, 20),
        'notifications_clicked': (0, 500),
    }

    # Display numerical inputs
    for col in numerical_cols:
        if col == 'signup_date':
            # Use date input for signup_date
            selected_date = st.date_input(
                f"**{col}**",
                value=date(2022, 1, 1),
                min_value=date(2015, 1, 1),
                max_value=date(2022, 12, 31),
                help="Customer subscription start date"
            )
            input_data[col] = selected_date
        elif col in float_continuous_features:
            default_val = float(num_defaults.get(col, 0))
            min_val = float(num_bounds.get(col, (0, 100))[0])
            max_val = float(num_bounds.get(col, (0, 100))[1])
            input_data[col] = st.number_input(
                f"**{col}**",
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                step=0.1,
            )
        else:
            # Integer input for count-based features
            default_val = int(num_defaults.get(col, 0))
            min_val = int(num_bounds.get(col, (0, 100))[0])
            max_val = int(num_bounds.get(col, (0, 100))[1])
            input_data[col] = st.number_input(
                f"**{col}**",
                value=default_val,
                min_value=min_val,
                max_value=max_val,
                step=1,
            )

    # Categorical inputs - use actual values from the dataset
    st.markdown("**Categorical Features**")

    cat_options_map = {
        'location': ['Alabama', 'California', 'Florida', 'Georgia', 'Idaho', 'Maine',
                     'Montana', 'Nebrasksa', 'New Jersey', 'New York', 'North Carolina',
                     'North Dakota', 'South Carolina', 'Utah', 'Vermont', 'Virginia',
                     'Washington', 'West Virginia', 'Wisconsin'],
        'subscription_type': ['Family', 'Free', 'Premium', 'Student'],
        'payment_plan': ['Monthly', 'Yearly'],
        'payment_method': ['Apple Pay', 'Credit Card', 'Debit Card', 'Paypal'],
        'customer_service_inquiries': ['High', 'Low', 'Medium']
    }

    for cat_col in categorical_cols:
        options = cat_options_map.get(cat_col, [f"Option_{i}" for i in range(3)])
        input_data[cat_col] = st.selectbox(f"**{cat_col}**", options)

    # Predict button
    if st.button("Predict Churn"):
        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_data])

        # Convert signup_date (Python date) to the encoded format expected by the model
        # The training data encoded signup_date as negative days from 2023-01-01
        # e.g., date(2022,12,31) -> -1, date(2022,1,1) -> -365, date(2015,1,1) -> -2922
        if 'signup_date' in input_df.columns and input_df['signup_date'].iloc[0] is not None:
            selected_date = input_df['signup_date'].iloc[0]
            if isinstance(selected_date, (str, datetime)):
                selected_date = pd.to_datetime(selected_date).date()
            if isinstance(selected_date, date):
                ref_date = datetime(2023, 1, 1).date()
                days_before = (ref_date - selected_date).days
                input_df['signup_date'] = float(-days_before)

        # Ensure correct column types match training data
        for col in numerical_cols:
            if col == 'signup_date':
                input_df[col] = float(input_df[col].iloc[0])
            else:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").astype(float)

        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype(str)

        # Ensure correct column order (numeric before categorical as in training)
        # The model expects columns in this order based on feature_names_in_
        expected_feature_names = getattr(model, 'feature_names_in_', None)
        if expected_feature_names is not None:
            col_mapping = {}
            for col_name in numerical_cols + [c for c in categorical_cols]:
                if col_name in input_df.columns:
                    col_mapping[col_name] = col_name
            input_df = input_df[list(col_mapping.keys())]

        try:
            prediction = int(model.predict(input_df)[0])
            prediction_proba = float(model.predict_proba(input_df)[0][1])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Display result using model.classes_ to determine churn class
        classes = list(model.named_steps["classifier"].classes_)
        if len(classes) > 1:
            churn_class = int(classes[1]) if int(classes[1]) == 1 else int(classes[-1])
        else:
            churn_class = int(classes[0])

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == churn_class:
            st.markdown("### Likely to Churn")
        else:
            st.markdown("### Likely to Retain")

        st.markdown(f"**Churn Probability:** {prediction_proba * 100:.2f}%")

        st.markdown("---")
        st.info("This is a **prediction**, not a guarantee. Customer behavior can change "
                "based on many factors.")


# ------------------------------------------------------------
# Page 3: Model Evaluation
# ------------------------------------------------------------
def show_evaluation():
    st.header("Model Evaluation")

    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    if not os.path.exists(comparison_path):
        st.error("Model comparison file not found! Please run `train_model.py` first.")
        return

    # Load comparison data
    comparison_df = pd.read_csv(comparison_path)

    st.subheader("Model Comparison")
    st.dataframe(comparison_df, use_container_width=True)

    # Display accuracy chart
    accuracy_path = os.path.join(OUTPUT_DIR, "accuracy_comparison.png")
    if os.path.exists(accuracy_path):
        st.subheader("Accuracy Comparison")
        st.image(accuracy_path, use_column_width=True)

    # Display precision/recall/F1 chart
    metrics_path = os.path.join(OUTPUT_DIR, "precision_recall_f1_comparison.png")
    if os.path.exists(metrics_path):
        st.subheader("Precision, Recall, and F1-score Comparison")
        st.image(metrics_path, use_column_width=True)

    # Display confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.subheader("Confusion Matrix (Best Model)")
        st.image(cm_path, use_column_width=True)

    st.markdown("---")
    st.markdown("""
    ### Metric Definitions
    - **Accuracy:** (True Positives + True Negatives) / Total Predictions
    - **Precision:** True Positives / (True Positives + False Positives)
    - **Recall:** True Positives / (True Positives + False Negatives)
    - **F1-score:** Harmonic mean of precision and recall

    **Note:** All metrics shown are **validation results**, not official Kaggle leaderboard scores.
    """)


# ------------------------------------------------------------
# Main App
# ------------------------------------------------------------
def main():
    st.sidebar.title("Music Churn Predictor")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Project Overview", "Predict Churn", "Model Evaluation"],
        help="Select a page to view"
    )

    st.markdown("---")

    if page == "Project Overview":
        show_overview()
    elif page == "Predict Churn":
        show_predict_page()
    elif page == "Model Evaluation":
        show_evaluation()


if __name__ == "__main__":
    main()
