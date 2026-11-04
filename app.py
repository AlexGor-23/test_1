import joblib
import pandas as pd

print("Прогнозирование стоимости недвижимости")

# Загрузка обученной модели при старте приложения
model = joblib.load("real_estate_model.pkl")

# Создание списка городов (удаление из model элементов features_to_remove)
features_to_remove = ['total_square', 'rooms', 'floor']
CITIES = [col for col in model.feature_names_in_ if col not in features_to_remove]

def get_list(prompt):
    total_words = sum(len(item.split()) for item in prompt)
    cities = str(prompt).split(',')
    print("Список городов:")
    for i in range(0, total_words, 7):
        print(', '.join(cities[i:i + 7]))

def get_city(prompt, CITIES):
    while True:
        get_list(CITIES)
        city = input(prompt)
        if city in CITIES:
            return city
        else:
            error_message = f"Неверный город. Введите город из списка"
            print(error_message)

def get_float(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Ошибка! Число должно быть в диапазоне от {min_val} до {max_val}.")
        except ValueError:
            print("Ошибка! Введите целое число.")

def get_integer(prompt, min_val, max_val):
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Ошибка! Число должно быть в диапазоне от {min_val} до {max_val}.")
        except ValueError:
            print("Ошибка! Введите целое число.")

get_city("Введите город из списка: ", CITIES)

total_square = get_float("Введите общую площадь квартиры (от 8 до 2070), м²: ", 8, 2070)

rooms = get_integer("Введите количество комнат (от 1 до 15): ", 1, 15)

floor = get_integer("Введите этаж квартиры (от 1 до 66): ", 1, 66)

input_data = pd.DataFrame(
    [[total_square, rooms, floor, get_city]],
    columns=['total_square', 'rooms', 'floor', 'city']
)

# Приведение столбцов новых данных к виду, на котором модель обучалась
input_encoded = pd.get_dummies(input_data, columns=['city'], prefix='', prefix_sep='')
expected_features = model.feature_names_in_
input_encoded = input_encoded.reindex(columns=expected_features, fill_value=0)

prediction = model.predict(input_encoded)[0]
formatted_price = f"{prediction:,.0f}".replace(',', ' ')
Prediction = f"{formatted_price} руб."

print("Стоимость квартиры:", Prediction)
