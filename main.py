import joblib
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(
    title="API прогнозирования стоимости недвижимости",
    description="Сервис для предсказания цены квартиры",
)

# Загрузка обученной модели при старте приложения
model = joblib.load("real_estate_model.pkl")

# Создание списка городов (удаление из model элементов features_to_remove)
features_to_remove = ['total_square', 'rooms', 'floor']
CITIES = [col for col in model.feature_names_in_ if col not in features_to_remove]

class PredictionRequest(BaseModel):
    city: str
    total_square: float
    rooms: int
    floor: int

class Result(BaseModel):
    result: float

@app.get("/health")
def health_check():
    """Liveness-проба (health-check) для проверки работоспособности сервиса"""
    return JSONResponse(content={"status": "OK", "model_loaded": True}, status_code=200)

@app.get("/predict_get")
def predict_get(
    city: str = Query(..., description="Название города", enum=CITIES),
    total_square: float = Query(..., description="Общая площадь квартиры, м²"),
    rooms: int = Query(..., description="Количество комнат"),
    floor: int = Query(..., description="Этаж квартиры")
):
    """Получение предсказания стоимости квартиры через GET-запрос"""
    input_data = pd.DataFrame(
        [[total_square, rooms, floor, city]],
        columns=['total_square', 'rooms', 'floor', 'city']
    )

    # Приведение столбцов новых данных к виду, на котором модель обучалась
    input_encoded = pd.get_dummies(input_data, columns=['city'], prefix='', prefix_sep='')
    expected_features = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    formatted_price = f"{prediction:,.0f}".replace(',', ' ')
    prediction_1 = f"{formatted_price} руб."


    return {
        "city": city,
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "predicted_price": prediction_1
    }

@app.post("/predict_post")
def predict_post(request: PredictionRequest):
    """Получение предсказания стоимости квартиры через POST-запрос"""
    input_data = pd.DataFrame(
        [[request.total_square, request.rooms, request.floor, request.city]],
        columns=['total_square', 'rooms', 'floor', 'city']
    )

    if request.city not in CITIES:
        error_message = f"Неверный город. Выберите город из списка: {', '.join(CITIES)}"
        raise HTTPException(
            status_code=400,
            detail=error_message
        )

    # Приведение столбцов новых данных к виду, на котором модель обучалась
    input_encoded = pd.get_dummies(input_data, columns=['city'], prefix='', prefix_sep='')
    expected_features = model.feature_names_in_
    input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    formatted_price = f"{prediction:,.0f}".replace(',', ' ')
    prediction_1 = f"{formatted_price} руб."

    return {
        "city": request.city,
        "total_square": request.total_square,
        "rooms": request.rooms,
        "floor": request.floor,
        "predicted_price": prediction_1
    }
