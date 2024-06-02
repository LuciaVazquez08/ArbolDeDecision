import numpy as np
import pandas as pd
import os
from ArbolID3 import ArbolID3
#import matplotlib.pyplot as plt
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier as DTC, plot_tree

def main():

    directorio_actual = os.path.dirname(os.path.abspath(__file__))
    iris = 'datasets/IRIS.csv'
    titanic = "datasets/titanic.csv"

    ruta_iris = os.path.join(directorio_actual, iris)
    ruta_titanic = os.path.join(directorio_actual, titanic)

    df_iris = pd.read_csv(ruta_iris)
    df_titanic = pd.read_csv(ruta_titanic)

    #limpiamos el titanic de cosas que lo pueden sensibilidzar demasiado (columnas numeicas no correspondientes a una categoria) --> despues vemos el resto 
    df_titanic = df_titanic.drop(columns=["PassengerId", "Name", "Fare", "Cabin", "Ticket"], axis = 1)
    df_titanic['Age'] = df_titanic['Age'].apply(lambda x: int(str(x)[0]) if pd.notnull(x) else x)

    null_counts = df_titanic.isnull().sum() 
    for column, null_count in null_counts.items():
        if null_count > 0:
            print (f"{column}: {null_count}")

    df_titanic = df_titanic.dropna(subset=['Age'])

    balance = df_titanic['Survived'].value_counts()
    print(balance)

    X = df_titanic.drop(["Survived"], axis = 1)
    y = df_titanic[["Survived"]]

    X_train =  X.iloc[:265].values
    y_train = y.iloc[:265].values

    X_test =  X.iloc[265:].values
    y_test = y.iloc[265:].values

    classifier = DecisionTreeClassifier(algoritmo = ArbolID3)
    classifier.fit(X_train, y_train)
    
    y_pred_nuestro = classifier.predict(X_test)
    print(classifier.arbol)
    
    for n in range(len(y_pred_nuestro)):
        print(f"{y_pred_nuestro[n]} - {y_test[n]}")

    accuracy = np.mean(y_pred_nuestro == y_test)
    print("Accuracy:", accuracy)

    # ----------------------------------------------------------- TEST PRED ---------------------------------------------------------------------------------------------------
    arbol = classifier.arbol

    dato = arbol.dato
    print(dato)

    hijos = arbol._hijos
    print(hijos)

    for arbol in hijos.values():
        print(arbol.dato)
    #------------------------------------------------------------ IRIS - C4.5 --------------------------------------------------------------------------------------------------

    # # Vemos el balance del dataset (en el caso de estar desbalanceado tendríamos que manejarlo)
    # balance = df_iris['species'].value_counts()

    # null_counts = df_iris.isnull().sum() 
    # for column, null_count in null_counts.items():
    #     if null_count > 0:
    #         print (f"{column}: {null_count}")

    # X = df_iris.drop(['species'], axis=1)
    # y = df_iris[['species']]

    # X_array = X.values
    # y_array = y.values

    # X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42)
    # # print (X_train.shape, X_test.shape)

    # # Creamos y entrenamos el clasificador de árbol de decisión
    # classifier = DecisionTreeClassifier(algoritmo = ArbolID3)
    # classifier.fit(X_train, y_train)

    # # print(classifier.arbol)

    # # Evaluamos el modelo
    # y_pred = classifier.predict(X_test)

    # # Calculamos la precisión del modelo
    # accuracy = np.mean(y_pred == y_test)
    # # print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()
