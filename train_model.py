import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Загрузка данных
df = pd.read_csv("data/realty_data.csv")
print(f"Загружено записей: {len(df)}")

# Заполнение значений NaN в rooms
df.loc[df['rooms'].isnull(), 'rooms'] = 1

# Заполнение значений Nan в city
df.loc[df['city'].isnull(), 'city'] = 'Москва'

# Получение уникальных значений из столбца city
unique_cities = df['city'].unique()
sorted_city = sorted(unique_cities)

# Сохранение списка городов в файл
with open('city_sorted.txt', 'w', encoding='utf-8') as f:
    for city in sorted_city:
        f.write(f"{city}\n")
print("Список городов сохранен в файл city_sorted.txt")

df = df.drop(columns=['product_name', 'period', 'postcode', 'address_name', 'lat', 'lon', 'object_type', 'settlement', 'district', 'area', 'description', 'source'])

df = pd.get_dummies(df, columns=['city'], prefix='', prefix_sep='', drop_first=False)

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели
print("Обучение модели...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Оценка качества
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nМетрики качества:")
print(f"MAE (Средняя абсолютная ошибка): {mae:,.0f} руб.")
print(f"R² (Коэффициент детерминации): {r2:.4f}")

# Сохранение модели
joblib.dump(model, "real_estate_model.pkl")
print("Модель сохранена в файл real_estate_model.pkl")