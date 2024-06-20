import os
import numpy as np
import pandas as pd
from arbol_decision.ArbolC4_5 import ArbolC4_5
from arbol_decision.ArbolID3 import ArbolID3
from arbol_decision.DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

def search_id3():

    directorio_actual = os.getcwd()
    ruta_archivo = os.path.join(directorio_actual, "datasets/tratamiento.csv")
    df = pd.read_csv(ruta_archivo)
    X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
    y = df[['AdministrarTratamiento']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hacemos listas de hiperparámetros para hallar la mejor combinación
    profundidad_max_list = [None, 3, 5]
    minimas_obs_n_list = [None, 2, 4] 
    minimas_obs_h_list = [None, 3, 4] 
    ganancia_minima_list = [None, 0.1, 0.2] 

    counter_tree = 0
    resultados_id3 = pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for profundidad_max in profundidad_max_list:
        for minimas_obs_n in minimas_obs_n_list:
            for minimas_obs_h in minimas_obs_h_list:
                for ganancia_minima in ganancia_minima_list:
                
                    print(f"ID3-{profundidad_max}-{minimas_obs_n}-{minimas_obs_h}-{ganancia_minima}")
                    
                    accuracy_scores = []
                    precision_scores = []
                    recall_scores = []
                    f1_scores = []
                    
                    for train_index, val_index in kf.split(X):
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                        
                        forest = DecisionTreeClassifier(algoritmo=ArbolID3, profundidad_max=profundidad_max, minimas_obs_n=minimas_obs_n,
                                                        minimas_obs_h=minimas_obs_h, ganancia_minima=ganancia_minima)
                        forest.fit(X_train, y_train)
                        y_pred = forest.predict(X_val)
                        
                        accuracy_scores.append(accuracy_score(y_val, y_pred))
                        precision_scores.append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
                        recall_scores.append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
                        f1_scores.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
                    
                    resultados_id3.loc[counter_tree, 'profundidad_max'] = profundidad_max
                    resultados_id3.loc[counter_tree, 'minimas_obs_n'] = minimas_obs_n
                    resultados_id3.loc[counter_tree, 'minimas_obs_h'] = minimas_obs_h
                    resultados_id3.loc[counter_tree, 'ganancia_minima'] = ganancia_minima
                    resultados_id3.loc[counter_tree, 'accuracy_train_cv'] = np.mean(accuracy_scores)
                    resultados_id3.loc[counter_tree, 'precision_cv'] = np.mean(precision_scores)
                    resultados_id3.loc[counter_tree, 'recall_cv'] = np.mean(recall_scores)
                    resultados_id3.loc[counter_tree, 'f1_score_cv'] = np.mean(f1_scores)

                    counter_tree += 1

    print(resultados_id3)
    print("Mejor combinación de hiperparámetros:")
    mejor_modelo_tree = resultados_id3.iloc[resultados_id3['f1_score_cv'].argmax()]
    print(mejor_modelo_tree)

    # Creamos un nuevo modelo con los mejores hiperparámetros
    mejor_tree = DecisionTreeClassifier(algoritmo=ArbolID3)

    # Entrenamos el modelo en el conjunto de entrenamiento
    mejor_tree.fit(X_train, y_train)

    # Evaluamos el modelo
    y_pred = mejor_tree.predict(X_test)

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

    return resultados_id3

def search_c45():

    directorio_actual = os.getcwd()
    ruta_archivo = os.path.join(directorio_actual, "datasets/heart.csv")
    df = pd.read_csv(ruta_archivo)
    X = df.drop(['output'], axis=1)
    y = df[['output']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hacemos listas de hiperparámetros para hallar la mejor combinación
    profundidad_max_list = [None, 3, 5, 7]
    minimas_obs_n_list = [None, 5, 10, 15, 20] 
    minimas_obs_h_list = [None, 5, 10, 15, 20] 
    ganancia_minima_list = [None, 0.1, 0.2] 

    counter_tree = 0
    resultados_c45 = pd.DataFrame()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for profundidad_max in profundidad_max_list:
        for minimas_obs_n in minimas_obs_n_list:
            for minimas_obs_h in minimas_obs_h_list:
                for ganancia_minima in ganancia_minima_list:
                
                    print(f"C4.5-{profundidad_max}-{minimas_obs_n}-{minimas_obs_h}-{ganancia_minima}")
                    
                    accuracy_scores = []
                    precision_scores = []
                    recall_scores = []
                    f1_scores = []
                    
                    for train_index, val_index in kf.split(X):
                        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                        
                        forest = DecisionTreeClassifier(algoritmo=ArbolC4_5, profundidad_max=profundidad_max, minimas_obs_n=minimas_obs_n,
                                                        minimas_obs_h=minimas_obs_h, ganancia_minima=ganancia_minima)
                        forest.fit(X_train, y_train)
                        y_pred = forest.predict(X_val)
                        
                        accuracy_scores.append(accuracy_score(y_val, y_pred))
                        precision_scores.append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
                        recall_scores.append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
                        f1_scores.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
                    
                    resultados_c45.loc[counter_tree, 'profundidad_max'] = profundidad_max
                    resultados_c45.loc[counter_tree, 'minimas_obs_n'] = minimas_obs_n
                    resultados_c45.loc[counter_tree, 'minimas_obs_h'] = minimas_obs_h
                    resultados_c45.loc[counter_tree, 'ganancia_minima'] = ganancia_minima
                    resultados_c45.loc[counter_tree, 'accuracy_train_cv'] = np.mean(accuracy_scores)
                    resultados_c45.loc[counter_tree, 'precision_cv'] = np.mean(precision_scores)
                    resultados_c45.loc[counter_tree, 'recall_cv'] = np.mean(recall_scores)
                    resultados_c45.loc[counter_tree, 'f1_score_cv'] = np.mean(f1_scores)

                    counter_tree += 1

    print(resultados_c45)
    print("Mejor combinación de hiperparámetros:")
    mejor_modelo_tree = resultados_c45.iloc[resultados_c45['f1_score_cv'].argmax()]
    print(mejor_modelo_tree)

    # Creamos un nuevo modelo con los mejores hiperparámetros
    mejor_tree = DecisionTreeClassifier(algoritmo=ArbolC4_5, profundidad_max=7, ganancia_minima=0.2)

    # Entrenamos el modelo en el conjunto de entrenamiento
    mejor_tree.fit(X_train, y_train)

    # Evaluamos el modelo
    y_pred = mejor_tree.predict(X_test)

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

    return resultados_c45

if __name__ == '__main__':
    search_c45()