from   sklearn.datasets import fetch_california_housing
from   sklearn.model_selection import train_test_split  
from   sklearn.linear_model import LinearRegression
from   sklearn.metrics import mean_squared_error

# cargar el conjunto de datos california
california = fetch_california_housing()
x_train, x_test, y_train, y_test = train_test_split(california.data, california.target, test_size=0.2, random_state=42)

# crear el modelo de regresion lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(x_train, y_train)

# Predecir las etiquetas para los datos de prueba
y_pred = model.predict(x_test)

# calcular el error cuadratico medio
mse = mean_squared_error(y_test, y_pred)
print("Error cuadratico medio: ", mse)