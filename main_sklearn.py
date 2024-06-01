import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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

    X_train =  X.iloc[:265]
    y_train = y.iloc[:265]

    X_test =  X.iloc[265:]
    y_test = y.iloc[265:]

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

    #------------------------------------------------------------ IRIS - C4.5 --------------------------------------------------------------------------------------------------
    iris = 'datasets/IRIS.csv'    
    
    ruta_iris = os.path.join(directorio_actual, iris)

    df_iris = pd.read_csv(ruta_iris)
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
    # print (X_train.shape, X_test.shape)

    # clf = DecisionTreeClassifier()
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # plt.figure(figsize=(20,10))
    # plot_tree(clf, filled=True, feature_names= df_iris.columns, class_names=df_iris["species"], rounded=True)
    # plt.show()

    # accuracy = accuracy_score(y_test, y_pred)
    # precision = recall_score(y_test, y_pred, average='weighted')
    # recall = f1_score(y_test, y_pred, average='weighted')
    # print(f'Accuracy: {accuracy}')
    # print(f'Precision: {precision}')
    # print(f'Recall: {recall}')

if __name__ == '__main__':
    main()