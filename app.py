import joblib
import pandas as pd
import streamlit as st

# Настройка страницы
st.set_page_config(page_title="Оценка недвижимости", page_icon="🏠")
st.title("🏠 Прогнозирование стоимости недвижимости")
st.markdown("Введите параметры квартиры, и модель оценит её рыночную стоимость.")

# Загрузка обученной модели
@st.cache_resource
def load_model():
    return joblib.load("real_estate_model.pkl")

model = load_model()

with open('city_sorted.txt', 'r', encoding='utf-8') as file:
    CITIES = [city.strip() for city in file]

with st.form("prediction_form"):
    city = st.selectbox("Город", CITIES)
    total_square = st.number_input(
        "Площадь квартиры, м²", min_value=8, max_value=2070,
        value=50, step=1
    )
    rooms = st.number_input(
        "Количество комнат", min_value=1, max_value=15, value=2, step=1
    )
    floor = st.number_input(
        "Этаж", min_value=1, max_value=66, value=5, step=1
    )
    submitted = st.form_submit_button("Рассчитать стоимость")

if submitted:
    # Создание DataFrame с исходными данными
    input_data = pd.DataFrame([{
        "total_square": float(total_square),
        "rooms": int(rooms),
        "floor": int(floor),
        "city": city,
    }])

    # Применение того же one-hot кодирования, что и при обучении
    input_encoded = pd.get_dummies(input_data, columns=['city'], prefix='', prefix_sep='')

    # Приведение столбцов к точному соответствию с обучающей выборкой
    expected_features = model.feature_names_in_

    # Добавление отсутствующих столбцов (заполнение их нулями) и выстраивание в правильном порядке
    input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

    # Предсказание стоимости
    prediction = model.predict(input_encoded)[0]

    st.success("### 💰 Прогнозируемая стоимость")
    st.metric("Цена", f"{prediction:,.0f} ₽")
    st.info(
        f"Параметры запроса:\n"
        f"- Город: **{city}**\n"
        f"- Площадь: **{total_square} м²**\n"
        f"- Комнат: **{rooms}**\n"
        f"- Этаж: **{floor}**"
    )