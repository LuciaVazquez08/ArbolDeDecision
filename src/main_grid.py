import pandas as pd
from arbol_decision.ArbolC4_5 import ArbolC4_5
from arbol_decision.ArbolID3 import ArbolID3
from arbol_decision.DecisionTreeClassifier import DecisionTreeClassifier
from arbol_decision.RandomForestClassifier import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix


def main():
    print("Dataset elegido:")
    df = pd.read_csv("C:/Users/naiar/OneDrive/Documentos/Unsam/CIENCIA DE DATOS/CUATRIMESTRE_4/ALGORITMOS_2/trabajo_final/datasets/tratamiento.csv")
    print(df)

    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Hacemos listas de hiperparámetros para hallar la mejor combinación
    '''prof_max_list = [5, 10, 15, 20]
    minimas_obs_n_list = [3, 5, 10, 20, 30]
    minimas_obs_h_list = [3, 5, 10, 20, 30]
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    counter_tree = 0
    data_results_tree = pd.DataFrame()
    N_folds_tree = 5

    for profundidad_max in prof_max_list:
        for minimas_obs_n in minimas_obs_n_list:
            for minimas_obs_h in minimas_obs_h_list:
                tree = DecisionTreeClassifier(profundidad_max=profundidad_max, minimas_obs_n=minimas_obs_n, minimas_obs_h=minimas_obs_h)
                tree.fit(X_train, y_train)

                # Utilizamos cross_validate para la validación cruzada en conjunto de entrenamiento
                cv_scores = cross_validate(tree, X_train, y_train, cv=N_folds_tree, scoring=scoring)

                # Obtenemos los resultados promedio de las métricas durante la validación cruzada
                accuracy_train_fold = cv_scores['test_accuracy'].mean()
                precision_train_fold = cv_scores['test_precision'].mean()
                recall_train_fold = cv_scores['test_recall'].mean()
                f1_train_fold = cv_scores['test_f1'].mean()

                # Guardamos resultados
                data_results_tree.loc[counter_tree, 'profundidad_max_tree'] = profundidad_max
                data_results_tree.loc[counter_tree, 'minimas_obs_n_tree'] = minimas_obs_n
                data_results_tree.loc[counter_tree, 'minimas_obs_h_tree'] = minimas_obs_h
                data_results_tree.loc[counter_tree, 'accuracy_train_cv'] = accuracy_train_fold
                data_results_tree.loc[counter_tree, 'precision_cv'] = precision_train_fold
                data_results_tree.loc[counter_tree, 'recall_cv'] = recall_train_fold
                data_results_tree.loc[counter_tree, 'f1_score_cv'] = f1_train_fold

                counter_tree += 1

    print("Mejor combinación de hiperparámetros:")
    mejor_modelo_tree = data_results_tree.iloc[data_results_tree['f1_score_cv'].argmax()]
    print(mejor_modelo_tree)'''

    # Creamos y entrenamos el clasificador de árbol de decisión con los mejores hiperparámetros
    mejor_classifier = DecisionTreeClassifier(algoritmo=ArbolC4_5)
    mejor_classifier.fit(X_train, y_train)
    print(mejor_classifier._arbol)

    # Evaluamos el modelo
    y_pred = mejor_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average= 'weighted')
    recall = recall_score(y_test, y_pred, average= 'weighted')
    f1 = f1_score(y_test, y_pred, average= 'weighted')
    matriz = confusion_matrix(y_test, y_pred)
    print("Resultados en el conjunto de evaluación:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-score: {f1}')
    print(f'{matriz}')


if __name__ == '__main__':
    main()