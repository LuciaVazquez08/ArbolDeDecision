import pandas as pd
import os
from ArbolC4_5 import ArbolC4_5
from ArbolID3 import ArbolID3
from DecisionTreeClassifier import DecisionTreeClassifier
from RandomForest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# ----------------------- TEST ID3 - DECISION TREE CLASSIFIER ------------------------------------------------------------------
# print("Dataset elegido:")
# directorio_actual = os.getcwd()
# ruta_archivo = os.path.join(directorio_actual,"datasets/tratamiento.csv")
# df = pd.read_csv(ruta_archivo)
# print(df)

# # Vemos el balance del target 
# balance = df['AdministrarTratamiento'].value_counts() 
# print(balance)

# X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
# y = df[['AdministrarTratamiento']]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Creamos y entrenamos el clasificador de árbol de decisión
# classifier = DecisionTreeClassifier(algoritmo="ID3")
# classifier.fit(X_train, y_train)

# print(classifier.arbol)

# # Evaluamos el modelo
# y_pred = classifier.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = recall_score(y_test, y_pred, average= 'weighted')
# recall = f1_score(y_test, y_pred, average= 'weighted')
# matriz = confusion_matrix(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'{matriz}')

# ----------------------- TEST ID3 - RANDOM FOREST CLASSIFIER ------------------------------------------------------------------
# print("Dataset elegido:")
# directorio_actual = os.getcwd()
# ruta_archivo = os.path.join(directorio_actual,"datasets/tratamiento.csv")
# df = pd.read_csv(ruta_archivo)

# # Vemos el balance del target 
# balance = df['AdministrarTratamiento'].value_counts() 
# print(balance)

# X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
# y = df[['AdministrarTratamiento']]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Creamos y entrenamos el clasificador de árbol de decisión
# classifier = RandomForestClassifier(algoritmo="ID3", tecnica_balanceo="RandomUnder")
# print("Lo balanceamos con RandomUnder")
# balance = df['AdministrarTratamiento'].value_counts() 
# print(balance)

# classifier.fit(X_train, y_train)

# # Evaluamos el modelo
# y_pred = classifier.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = recall_score(y_test, y_pred, average= 'weighted')
# recall = f1_score(y_test, y_pred, average= 'weighted')
# matriz = confusion_matrix(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'{matriz}')

# ----------------------- TEST C4.5 - DECISION TREE CLASSIFIER ------------------------------------------------------------------
# print("Dataset elegido:")
# directorio_actual = os.getcwd()
# ruta_archivo = os.path.join(directorio_actual,"datasets/star.csv")
# df = pd.read_csv(ruta_archivo)
# print(df)

# # Vemos el balance del target 
# balance = df['TargetClass'].value_counts() 
# print(balance)

# X = df.drop(['TargetClass', 'SpType'], axis=1)
# y = df[['TargetClass']]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Creamos y entrenamos el clasificador de árbol de decisión
# classifier = DecisionTreeClassifier(algoritmo = "C4.5")
# classifier.fit(X_train, y_train)

# print(classifier.arbol)

# # Evaluamos el modelo
# y_pred = classifier.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = recall_score(y_test, y_pred, average= 'weighted')
# recall = f1_score(y_test, y_pred, average= 'weighted')
# matriz = confusion_matrix(y_test, y_pred)
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'Recall: {recall}')
# print(f'{matriz}')

# ----------------------- TEST C4.5 - RANDOM FOREST CLASSIFIER ------------------------------------------------------------------
print("Dataset elegido:")
directorio_actual = os.getcwd()
ruta_archivo = os.path.join(directorio_actual,"datasets/star.csv")
df = pd.read_csv(ruta_archivo)
print(df)

# Vemos el balance del target 
balance = df['TargetClass'].value_counts() 
print(balance)

X = df.drop(['TargetClass', 'SpType'], axis=1)
y = df[['TargetClass']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos y entrenamos el clasificador de árbol de decisión
classifier = RandomForestClassifier(algoritmo = "C4.5", tecnica_balanceo="SMOTE")
classifier.fit(X_train, y_train)

# Evaluamos el modelo
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = recall_score(y_test, y_pred, average= 'weighted')
recall = f1_score(y_test, y_pred, average= 'weighted')
matriz = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'{matriz}')
