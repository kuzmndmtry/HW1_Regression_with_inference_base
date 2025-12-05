import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

st.set_page_config(page_title="Car Price Prediction", layout="wide")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "my_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
TRAIN_DATA_PATH = MODEL_DIR / "df_train.csv"

@st.cache_resource
def load_model():
    """Загружаем модель через pickle"""
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURE_NAMES_PATH, 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

# Загружаем модель
try:
    MODEL, FEATURE_NAMES = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

@st.cache_data
def load_train_data():
    """Данные для EDA"""
    try:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        return train_df
    except FileNotFoundError:
        st.warning("Тренировочные данные не найдены.")
        return
    
@st.cache_resource    
def generate_profiling_report(df, title="Profiling Report"):
    profile = ProfileReport(df, title='Pandas Profiling Report')
    return profile

# --- Основной интерфейс ---
st.title("Предсказание цены автомобиля")

# EDA   
train_df = load_train_data()             
profile = generate_profiling_report(train_df)            
profile_html = profile.to_html()
            
st.subheader("Отчет ydata-profiling")
components.html(profile_html, height=800, scrolling=True)

# Загрузка CSV файла
st.subheader("Предсказание стоимости автомобиля")
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
if uploaded_file is None:
    st.info("Данные не загружены!")
    st.stop()

# Загружаем данные и делаем предсказания
df = pd.read_csv(uploaded_file)
try:
    features = df[FEATURE_NAMES]
    predictions = MODEL.predict(features)
    df['predicted_price'] = predictions
except Exception as e:
    st.error(e)
    st.stop()

# Скачивание результатов
st.success("Успех!")
st.subheader("Скачать результаты")
csv = df.to_csv(index=False)
st.download_button(
    label="Скачать",
    data=csv,
    file_name="predict.csv")

st.subheader("О модели")
st.write(f"Model {type(MODEL.best_estimator_).__name__}")
st.write(f"Params {MODEL.best_params_}")
st.write(f"Score {MODEL.best_score_:.4f}")
st.write(f"CV folds{MODEL.cv}")
    
# Делал при помощи ноутбука с урока + местами консультация GPT   
    
    


