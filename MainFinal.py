import pandas as pd
import os
from ArbolC4_5 import ArbolC4_5
from ArbolID3 import ArbolID3
from DecisionTreeClassifier import DecisionTreeClassifier
from RandomForest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def main_interactivo():
    print("Bienvenidos a nuestra magnífica DEMO sobre árboles de decisión.")
    input("¡Vamos a comenzar!")

    while True:
        print("\nSelecciona una opción:")
        print("1. Funcionamiento del ID3 con DecisionTreeClassifier")
        print("2. Funcionamiento del C4.5 con DecisionTreeClassifier")
        print("3. Funcionamiento del ID3 con RandomForest")
        print("4. Funcionamiento del C4.5 con RandomForest")
        print("5. Salir")

        opcion = input("Ingrese el número de la tarea que desea realizar: ")

        if opcion == "1":
            id3_DecisionTree()
        elif opcion == "2":
            c45_DecisionTree()
        elif opcion == "3":
            id3_RandomForest()
        elif opcion == "4":
            c45_RandomForest()
        elif opcion == "5":
            print("Gracias por su atención, ¡hasta luego!")
            break
        else:
            print("Opción no válida. Por favor, ingrese un número válido.")

        input("Presiona Enter para continuar...")


def id3_DecisionTree():
    print("Dataset elegido:")
    directorio_actual = os.getcwd()
    ruta_archivo = os.path.join(directorio_actual,"datasets/tratamiento.csv")
    df = pd.read_csv(ruta_archivo)
    print(df)

    # Vemos el balance del target 
    balance = df['AdministrarTratamiento'].value_counts() 
    print(balance)

    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = DecisionTreeClassifier(algoritmo="ID3")
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


def c45_DecisionTree():
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
    classifier = DecisionTreeClassifier(algoritmo = "C4.5")
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


def id3_RandomForest():
    print("Dataset elegido:")
    directorio_actual = os.getcwd()
    ruta_archivo = os.path.join(directorio_actual,"datasets/tratamiento.csv")
    df = pd.read_csv(ruta_archivo)

    # Vemos el balance del target 
    balance = df['AdministrarTratamiento'].value_counts() 
    print(balance)

    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = RandomForestClassifier(algoritmo="ID3")
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


def c45_RandomForest():
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
    classifier = RandomForestClassifier(algoritmo = "C4.5", tecnica_balanceo="RandomUnder")
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


if __name__ == "__main__":
    main_interactivo()
