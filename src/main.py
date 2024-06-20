import pandas as pd
from arbol_decision.ArbolC4_5 import ArbolC4_5
from arbol_decision.ArbolID3 import ArbolID3
from arbol_decision.DecisionTreeClassifier import DecisionTreeClassifier
from arbol_decision.RandomForestClassifier import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def main_interactivo():
    print("Bienvenidos a nuestra magnífica DEMO sobre árboles de decisión.")
    input("¡Vamos a comenzar!")

    while True:
        print("\nSelecciona una opción:")
        print("1. Funcionamiento del ID3 con DecisionTreeClassifier")
        print("2. Funcionamiento del C4.5 con DecisionTreeClassifier (solo atributos continuos)")
        print("3. Funcionamiento del ID3 con RandomForest")
        print("4. Funcionamiento del C4.5 con RandomForest (solo atributos continuos)")
        print("5. Funcionamiento del C4.5 con DecisionTreeClassifier")
        print("6. Funcionamiento del C4.5 con RandomForest")
        print("7. Salir")

        opcion = input("Ingrese el número de la tarea que desea realizar: ")

        if opcion == "1":
            id3_DecisionTree()
        elif opcion == "2":
            c45_DecisionTree_continuos()
        elif opcion == "3":
            id3_RandomForest()
        elif opcion == "4":
            c45_RandomForest_continuos()
        elif opcion == "5":
            c45_DecisionTree()
        elif opcion == "6":
            c45_RandomForest()
        elif opcion == "7":
            print("Gracias por su atención, ¡hasta luego!")
            break
        else:
            print("Opción no válida. Por favor, ingrese un número válido.")

        input("Presiona Enter para continuar...")


def id3_DecisionTree():
    print("Dataset elegido:")
    df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/tratamiento.csv")
    print(df)

    input("Presione Enter para continuar...")

    # Vemos el balance del target 
    balance = df['AdministrarTratamiento'].value_counts() 
    print(f'Balance del target: {balance}')
    input("Vamos a ver el resultado de nuestro árbol entrenado...")

    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = DecisionTreeClassifier(algoritmo=ArbolID3, minimas_obs_h=3)
    classifier.fit(X_train, y_train)
    print(classifier._arbol)

    input("Ahora vamos a ver las métricas:")

    # Evaluamos el modelo
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    matriz = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'{matriz}')


def c45_DecisionTree_continuos():
    print("Dataset elegido:")
    df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/star.csv")
    print(df)

    input("Presione Enter para continuar...")

    # Vemos el balance del target 
    balance = df['TargetClass'].value_counts() 
    print(f'Balance del target: {balance}')

    input("Vamos a ver el resultado de nuestro árbol entrenado...")

    X = df.drop(['TargetClass', 'SpType'], axis=1)
    y = df[['TargetClass']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = DecisionTreeClassifier(algoritmo = ArbolC4_5)
    classifier.fit(X_train, y_train)
    print(classifier._arbol)

    input("Este árbol es muy grande, así que vamos a probar cómo funcionan algunos criterios de parada...")
    input("Vamos a usar una profundidad máxima = 3 y mínimas observaciones por nodo = 35:")

    # Creamos y entrenamos el clasificador de árbol de decisión con hiperparámetros
    classifier = DecisionTreeClassifier(algoritmo = ArbolC4_5, profundidad_max=3, minimas_obs_n=35)
    classifier.fit(X_train, y_train)
    print(classifier._arbol)

    input("Ahora vamos a ver las métricas:")

    # Evaluamos el modelo
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    matriz = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'{matriz}')


def id3_RandomForest():
    print("Dataset elegido:")
    df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/tratamiento.csv")
    print(df)

    input("Presione Enter para continuar...")

    # Vemos el balance del target 
    balance = df['AdministrarTratamiento'].value_counts() 
    print(f'Balance del target: {balance}')

    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    forest = RandomForestClassifier(algoritmo=ArbolID3, feature_selection_method='sqrt')
    forest.fit(X_train, y_train)

    input("Ahora vamos a ver las métricas de nuestro RandomForest con 100 árboles:")

    # Evaluamos el modelo
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    matriz = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'{matriz}')


def c45_RandomForest_continuos():
    print("Dataset elegido:")
    df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/star.csv")
    print(df)

    input("Presione Enter para continuar...")

    # Vemos el balance del target 
    balance = df['TargetClass'].value_counts() 
    print(f'Balance del target: {balance}')

    X = df.drop(['TargetClass', 'SpType'], axis=1)
    y = df[['TargetClass']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    forest = RandomForestClassifier(algoritmo = ArbolC4_5, n_estimadores=5, feature_selection_method='sqrt')
    forest.fit(X_train, y_train)

    input("Ahora vamos a ver las métricas de nuestro RandomForest con 5 árboles:")

    # Evaluamos el modelo
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    matriz = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'{matriz}')

def c45_DecisionTree():
    print("Dataset elegido:")
    df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/heart.csv")
    print(df)

    input("Presione Enter para continuar...")

    # Vemos el balance del target 
    balance = df['output'].value_counts() 
    print(f'Balance del target: {balance}')

    input("Vamos a ver el resultado de nuestro árbol entrenado...")

    X = df.drop(['output'], axis=1)
    y = df[['output']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = DecisionTreeClassifier(algoritmo=ArbolC4_5)
    classifier.fit(X_train, y_train)
    print(classifier._arbol)

    input("Ahora vamos a ver las métricas:")

    # Evaluamos el modelo
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    matriz = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'{matriz}')

def c45_RandomForest():
    print("Dataset elegido:")
    df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/heart.csv")
    print(df)

    input("Presione Enter para continuar...")

    # Vemos el balance del target 
    balance = df['output'].value_counts() 
    print(f'Balance del target: {balance}')

    X = df.drop(['output'], axis=1)
    y = df[['output']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos y entrenamos el clasificador de árbol de decisión
    forest = RandomForestClassifier(algoritmo=ArbolC4_5, n_estimadores=20, feature_selection_method='none')
    forest.fit(X_train, y_train)

    input("Ahora vamos a ver las métricas de nuestro RandomForest con 20 árboles:")

    # Evaluamos el modelo
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    matriz = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'{matriz}')


if __name__ == "__main__":
    main_interactivo()
