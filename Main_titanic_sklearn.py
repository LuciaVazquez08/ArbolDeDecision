import numpy as np
import pandas as pd
import os
from ArbolID3 import ArbolID3
import matplotlib.pyplot as plt
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, f1_score

def main():

    directorio_actual = os.path.dirname(os.path.abspath(__file__))

    titanic = "datasets/titanic.csv"

    ruta_titanic = os.path.join(directorio_actual, titanic)

    df_titanic = pd.read_csv(ruta_titanic)

    #limpiamos el titanic de cosas que lo pueden sensibilidzar demasiado (columnas numeicas no correspondientes a una categoria) --> despues vemos el resto 
    df_titanic = df_titanic.drop(columns=["PassengerId", "Name", "Fare", "Cabin", "Ticket"], axis = 1)
    df_titanic = df_titanic.dropna(subset=['Age'])    
    df_titanic['Age'] = df_titanic['Age'].apply(lambda x: int(str(x)[0]))
    df_titanic['Sex'] = df_titanic['Sex'].replace({'male': 0, 'female': 1})
    df_titanic['Embarked'] = df_titanic['Embarked'].replace({'C': 0, 'S': 1, "Q":2})   

    X = df_titanic.drop(["Survived"], axis = 1)
    y = df_titanic[["Survived"]]

    X_train =  X.iloc[:265].values
    y_train = y.iloc[:265].values

    X_test =  X.iloc[265:].values
    y_test = y.iloc[265:].values

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plt.figure(figsize=(20,10))
    plot_tree(clf, filled=True, feature_names= X.columns)
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    precision = recall_score(y_test, y_pred, average='weighted')
    recall = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

if __name__ == '__main__':
    main()
