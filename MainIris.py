import numpy as np
import pandas as pd
from ArbolID3 import ArbolID3
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def main():

    df = pd.read_csv("C:/Users/naiar/Downloads/IRIS.csv")

    # Vemos el balance del dataset (en el caso de estar desbalanceado tendríamos que manejarlo)
    balance = df['species'].value_counts()

    null_counts = df.isnull().sum() 
    for column, null_count in null_counts.items():
        if null_count > 0:
            print (f"{column}: {null_count}")

    X = df.drop(['species'], axis=1)
    y = df[['species']]
    
    X_array = X.values
    y_array = y.values

    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42)
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