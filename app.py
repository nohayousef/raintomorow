
##$$$$$$$$$$$$$$$###################$$$$$$$$$$###################################

#########################################
import streamlit as st
import joblib # type: ignore
import pandas as pd
import os
import numpy as np
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# ✅ Set Streamlit Page
st.set_page_config(page_title="Rain in Australia", page_icon="☔", layout="wide")

# ✅ Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Visualization", "🔍 Prediction"])

# ✅ Load dataset
@st.cache_data
def load_data():
    file_path = r"C:\Users\NohaA\myenv\finallllllll project\New folder\weatherAUS.csv"
    if not os.path.exists(file_path):
        st.error("❌ Dataset file not found! Check the file path.")
        return None
    df = pd.read_csv(file_path, parse_dates=["Date"])
    df.dropna(subset=["WindGustDir", "RainTomorrow", "Location", "Rainfall", "WindSpeed9am", "Humidity9am", "Temp9am", "Temp3pm"], inplace=True)
    df["Season"] = df["Date"].dt.month.map(
        lambda m: "Winter" if m in [12, 1, 2] else 
                  "Spring" if m in [3, 4, 5] else 
                  "Summer" if m in [6, 7, 8] else 
                  "Autumn"
    )
    return df

df = load_data()

# ✅ Load Model
model_paths = [
    r"C:\Users\NohaA\myenv\finallllllll project\New folder\voting5_pipeline.pkl", 
    "voting3_pipeline.pkl"
]

model, scaler, le = None, None, None
for path in model_paths:
    if os.path.exists(path):
        try:
            model, scaler, le = joblib.load(path)
            if hasattr(model, "predict"):
                st.sidebar.success(f"✅ Model Loaded Successfully from {path}!")
            else:
                st.sidebar.error("❌ Model is not properly trained. Retrain and save it again.")
                model = None
            break
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model from {path}: {e}")
            model = None

if model is None:
    st.sidebar.warning("⚠️ No valid model file found. Please retrain the model.")

# 📌 **Home Page**
if page == "🏠 Home":
    st.title("Rain in Australia 🌧️🌦️")
    st.image(r"C:\Users\NohaA\myenv\finallllllll project\New folder\Screenshot 2025-03-09 201614.png", width=600)
    
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
    else:
        st.error("🚨 Dataset could not be loaded.")
    
    st.write("Use the sidebar to navigate to Visualization or Prediction.")

# 📌 **Visualization Page**
elif page == "📊 Visualization" and df is not None:
    st.title("📊 Weather Data Visualization")
    st.sidebar.title("🔍 Choose Visualization")
    visualization_option = st.sidebar.selectbox("Select an option", [
        "🌧️ Rainfall Distribution", "📊 Rain Probability by Wind Direction", "🔥 Correlation Heatmap",
        "💨 Wind Speed vs. Rain Probability", "🌦️ Seasonal Rain Effects"
    ])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    if visualization_option == "🌧️ Rainfall Distribution":
        selected_location = st.sidebar.selectbox("Select Location", df["Location"].unique())
        sns.histplot(df[df["Location"] == selected_location]["Rainfall"], bins=30, kde=True, color="blue", ax=ax)
        ax.set_title(f"Rainfall Distribution in {selected_location}")

    elif visualization_option == "📊 Rain Probability by Wind Direction":
        df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
        sns.barplot(x=df["WindGustDir"], y=df["RainTomorrow"], ax=ax, palette="coolwarm")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    elif visualization_option == "🔥 Correlation Heatmap":
        sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

    elif visualization_option == "💨 Wind Speed vs. Rain Probability":
        sns.boxplot(x=df["RainTomorrow"], y=df["WindSpeed9am"], palette="coolwarm", ax=ax)
        ax.set_title("Wind Speed in Morning vs. Rain Probability")

    elif visualization_option == "🌦️ Seasonal Rain Effects":
        sns.countplot(data=df, x="Season", hue="RainTomorrow", palette="coolwarm", ax=ax)
        ax.set_title("Seasonal Effect on Rain Tomorrow")
    
    st.pyplot(fig)

# 📌 **Prediction Page**
elif page == "🔍 Prediction" and model is not None:
    st.title("🔍 Rain Prediction")
    st.sidebar.title("🌍 Enter Weather Details")
    location = st.sidebar.text_input("🌏 Location (Type the city name)")
    wind_gust_dir = st.sidebar.text_input("💨 Wind Gust Direction (N, S, E, W, etc.)")
    wind_speed_9am = st.sidebar.slider("🌬️ Wind Speed 9AM (km/h)", 0, 100, 30)
    humidity_9am = st.sidebar.slider("💧 Humidity 9AM (%)", 0, 100, 50)
    temp_9am = st.sidebar.slider("🌡️ Temperature 9AM (°C)", -10, 50, 20)
    temp_3pm = st.sidebar.slider("🌡️ Temperature 3PM (°C)", -10, 50, 25)
    prev_day_rainfall = st.sidebar.number_input("🌧️ Previous Day Rainfall (mm)", 0.0, 100.0, 0.0, step=0.5)
    rain_today = st.sidebar.radio("🌦️ Rain Today?", ["No", "Yes"]) == "Yes"

    try:
        location_encoded = le.transform([location])[0] if location else 0
        wind_gust_encoded = le.transform([wind_gust_dir])[0] if wind_gust_dir else 0
    except ValueError:
        st.error("⚠️ Invalid Location or Wind Gust Direction! Check the spelling or dataset.")
        location_encoded, wind_gust_encoded = 0, 0

    input_data = np.array([
        location_encoded, wind_gust_encoded, wind_speed_9am, humidity_9am, temp_9am, temp_3pm, prev_day_rainfall, int(rain_today)
    ]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    if st.sidebar.button("Submit"):
        prediction = model.predict(input_data_scaled)
        result = "☔ Yes, it will rain!" if prediction[0] == 1 else "🌤️ No, it will not rain."
        st.success(result)
