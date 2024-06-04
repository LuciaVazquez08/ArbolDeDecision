import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, f1_score

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

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    plt.figure(figsize=(20,10))
    plot_tree(clf, filled=True, feature_names= df.columns, class_names=df["species"], rounded=True)
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    precision = recall_score(y_test, y_pred, average='weighted')
    recall = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

if __name__ == '__main__':
    main()