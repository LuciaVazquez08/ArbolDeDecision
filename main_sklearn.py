import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

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

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = recall_score(y_test, y_pred, average='weighted')
    recall = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

if __name__ == '__main__':
    main()