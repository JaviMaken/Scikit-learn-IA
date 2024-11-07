from sklearn.dataset import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# cargar el conjunto de datos iris 
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_satet=42)

# crear el clasificador de vecinos mas cercanos
clf = KNeighborsClassifier(n_neighbors=3)

# entrenar el clasificador
clf.fit(X_train, y_train)

#Predecir las etiquetas para los dtos de prueba
y_pred = clf.predict(x_test)

# calcular la precision del calsificador
accuracy = accuracy_score(y_test, y_pred)
print("Precision del clasificador: ", accuracy)