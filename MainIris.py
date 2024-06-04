from graphviz import Digraph
import numpy as np
import pandas as pd
import os
from ArbolID3 import ArbolID3
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def main():
    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    iris = 'datasets/IRIS.csv'
    ruta_iris = os.path.join(directorio_actual, iris)
    df = pd.read_csv(ruta_iris)

    # Vemos el balance del dataset (en el caso de estar desbalanceado tendríamos que manejarlo)
    balance = df['species'].value_counts()

    null_counts = df.isnull().sum() 
    for column, null_count in null_counts.items():
        if null_count > 0:
            print (f"{column}: {null_count}")

    X = df.drop(['species'], axis=1)
    y = df[['species']]

    X_train = X.iloc[:120].values
    y_train = y.iloc[:120].values

    X_test = X.iloc[120:].values
    y_test = y.iloc[120:].values

    print (X_train.shape, X_test.shape)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = DecisionTreeClassifier(algoritmo = ArbolID3)
    classifier.fit(X_train, y_train)

    # Evaluamos el modelo
    y_pred = classifier.predict(X_test)

    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()