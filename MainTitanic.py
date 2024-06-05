import numpy as np
import pandas as pd
from ArbolID3 import ArbolID3
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def main():

    df = pd.read_csv("C:/Users/naiar/Downloads/titanic.csv")

    df_titanic = df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Age'], axis=1)

    null_counts = df_titanic.isnull().sum() 
    for column, null_count in null_counts.items():
        if null_count > 0:
            print (f"{column}: {null_count}")

    # Borramos las filas con valores faltantes de edad
    #df_titanic = df_titanic.dropna(subset=['Age'])
    #print(df_titanic)

    # Vemos el balance del dataset (en el caso de estar desbalanceado tendríamos que manejarlo)
    balance = df['Survived'].value_counts()
    print(balance)

    X = df_titanic.drop(["Survived"], axis = 1)
    y = df_titanic[["Survived"]]

    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    print (X_train.shape, X_test.shape)

    # Creamos y entrenamos el clasificador de árbol de decisión
    classifier = DecisionTreeClassifier(algoritmo = ArbolID3)
    classifier.fit(X_train, y_train)

    # Evaluamos el modelo
    y_pred = classifier.predict(X_test)

    #nombres_atributos = X.columns.tolist()
    #print(nombres_atributos)

    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

if __name__ == '__main__':
    main()
