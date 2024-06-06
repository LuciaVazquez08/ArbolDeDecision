import pandas as pd
from ArbolID3 import ArbolID3
from ArbolC4_5 import ArbolC4_5
from DecisionTreeClassifier import DecisionTreeClassifier
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def main():

    df = pd.read_csv("C:/Users/naiar/Downloads/tratamiento.csv")

    print(df)

    # Vemos el balance del target 
    balance = df['AdministrarTratamiento'].value_counts() 
    print(balance)


    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #print (X_train.shape, X_test.shape)


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


if __name__ == '__main__':
    main()