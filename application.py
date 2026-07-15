import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="🥗 NutriClass – Smart Food Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
.main-header {
    font-size: 3rem; 
    color: #ff6f61; 
    font-weight: bold;
    text-align: center;
    text-shadow: 2px 2px #ffe0e0;
}
.metric-card {
    background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
    padding: 1rem; 
    border-radius: 12px; 
    color: #2f2f2f;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
.report-box {
    background: #fff8e7; 
    padding: 1rem; 
    border-radius: 12px;
    border-left: 6px solid #ff6f61;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

# ================================
# HEADER
# ================================
st.markdown('<h1 class="main-header">🥗 NutriClass – Food Intelligence</h1>', unsafe_allow_html=True)

# ================================
# LOAD ARTIFACTS
# ================================
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    pca = joblib.load("pca.joblib")
    best_model = joblib.load("best_model.joblib")
    le_meal = joblib.load("le_meal.joblib")
    le_prep = joblib.load("le_prep.joblib")
    le_target = joblib.load("le_target.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    return scaler, pca, best_model, le_meal, le_prep, le_target, feature_cols

scaler, pca, best_model, le_meal, le_prep, le_target, feature_cols = load_artifacts()

@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_food_dataset_imbalanced.csv")
    return df

df = load_data()

# ================================
# SIDEBAR INFO
# ================================
st.sidebar.title("ℹ️ About NutriClass")
st.sidebar.write("NutriClass predicts food items based on nutritional profiles using ML models.")
st.sidebar.write("👨‍🍳 Built for smart diet planning and food classification.")

# ================================
# LIVE PREDICTION
# ================================
st.header("🔮 Live Prediction – Enter Nutritional Profile")

row1 = st.columns(3)
calories = row1[0].slider("🔥 Calories", 50, 600, 250)
protein = row1[1].slider("💪 Protein (g)", 0.0, 50.0, 15.0)
fat = row1[2].slider("🥓 Fat (g)", 0.0, 40.0, 10.0)

row2 = st.columns(3)
carbs = row2[0].slider("🍞 Carbs (g)", 0, 80, 30)
sugar = row2[1].slider("🍬 Sugar (g)", 0, 40, 8)
fiber = row2[2].slider("🌿 Fiber (g)", 0, 20, 3)

row3 = st.columns(3)
sodium = row3[0].slider("🧂 Sodium (mg)", 0, 1000, 300)
cholesterol = row3[1].slider("🩸 Cholesterol (mg)", 0, 300, 20)
gi = row3[2].slider("📊 Glycemic Index", 0, 100, 55)

row4 = st.columns(2)
water = row4[0].slider("💧 Water Content (%)", 0, 100, 70)
serving = row4[1].slider("🍽️ Serving Size (units)", 1.0, 3.0, 1.0, step=0.1)

row5 = st.columns(4)
meal_type = row5[0].selectbox("🍴 Meal Type", sorted(df["Meal_Type"].unique()))
prep_method = row5[1].selectbox("👨‍🍳 Preparation Method", sorted(df["Preparation_Method"].unique()))
is_vegan = row5[2].checkbox("🌱 Vegan")
is_gluten_free = row5[3].checkbox("🚫 Gluten Free")

# ================================
# PREDICTION
# ================================
if st.button("🔮 **Predict Food Name**", type="primary", use_container_width=True):
    input_df = pd.DataFrame({
        "Calories": [calories],
        "Protein": [protein],
        "Fat": [fat],
        "Carbs": [carbs],
        "Sugar": [sugar],
        "Fiber": [fiber],
        "Sodium": [sodium],
        "Cholesterol": [cholesterol],
        "Glycemic_Index": [gi],
        "Water_Content": [water],
        "Serving_Size": [serving],
        "Meal_Type": [meal_type],
        "Preparation_Method": [prep_method],
        "Is_Vegan": [is_vegan],
        "Is_Gluten_Free": [is_gluten_free],
    })

    input_df["Meal_Type_enc"] = le_meal.transform(input_df["Meal_Type"])
    input_df["Preparation_Method_enc"] = le_prep.transform(input_df["Preparation_Method"])
    input_df["Is_Vegan_enc"] = input_df["Is_Vegan"].astype(int)
    input_df["Is_Gluten_Free_enc"] = input_df["Is_Gluten_Free"].astype(int)

    X_input = input_df[feature_cols]
    X_scaled = scaler.transform(X_input)
    X_pca = pca.transform(X_scaled)

    pred_enc = best_model.predict(X_pca)[0]
    probs = best_model.predict_proba(X_pca)[0]
    pred_name = le_target.inverse_transform([pred_enc])[0]
    confidence = probs.max()

    st.success(f"🍽️ Predicted Food: **{pred_name}**")
    #st.info(f"📈 Confidence Level: **{confidence:.1%}**")
