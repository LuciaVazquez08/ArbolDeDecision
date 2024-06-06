import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


# Función principal
def main():
    # Cargar el dataset desde un archivo CSV
    df = pd.read_csv("C:/Users/naiar/Downloads/heart.csv")
    print(df)

    # Vemos el balance del target 
    balance = df['output'].value_counts() 
    print(balance)

    X = df.drop(['output'], axis=1)
    y = df[['output']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print (X_train.shape, X_test.shape)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = DecisionTreeClassifier(algoritmo=ArbolC4_5)
    classifier.fit(X_train, y_train)

    print(classifier.arbol)

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

if __name__ == "__main__":
    main()