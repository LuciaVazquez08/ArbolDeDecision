import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# Cargar el conjunto de datos Iris
#df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/star.csv")
df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/heart.csv")

print(df)

#X = df.drop(['TargetClass', 'SpType'], axis=1)
#y = df[['TargetClass']]
X = df.drop(['output'], axis=1)
y = df[['output']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un clasificador de árbol de decisión
tree_clf = DecisionTreeClassifier(random_state=42)

# Entrenar el clasificador de árbol de decisión
tree_clf.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred_tree = tree_clf.predict(X_test)

# Calcular la precisión del árbol de decisión
accuracy = accuracy_score(y_test, y_pred_tree)
precision = precision_score(y_test, y_pred_tree, average= 'weighted')
recall = recall_score(y_test, y_pred_tree, average= 'weighted')
f1 = f1_score(y_test, y_pred_tree, average= 'weighted')
matriz = confusion_matrix(y_test, y_pred_tree)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'{matriz}')

# Crear un clasificador de Random Forest
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el clasificador de Random Forest
forest_clf.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred_forest = forest_clf.predict(X_test)

# Calcular la precisión del Random Forest
accuracy = accuracy_score(y_test, y_pred_forest)
precision = precision_score(y_test, y_pred_forest, average= 'weighted')
recall = recall_score(y_test, y_pred_forest, average= 'weighted')
f1 = f1_score(y_test, y_pred_forest, average= 'weighted')
matriz = confusion_matrix(y_test, y_pred_forest)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'{matriz}')

