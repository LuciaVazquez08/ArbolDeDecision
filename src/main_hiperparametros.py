import os
import pandas as pd
from arbol_decision.ArbolC4_5 import ArbolC4_5
from arbol_decision.ArbolID3 import ArbolID3
from arbol_decision.RandomForestClassifier import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, f1_score, confusion_matrix

n_estimadores = [10]
profundidad_max = [None, 3,5]
minimas_obs_n = [None, 20, 40] 
minimas_obs_h = [None, 5,10] 
ganancia_minima = [None, 1.25, 3] 
bootstrap = [True, False]
feature_selection_method = ["sqrt", "log", "none"]
tecnica_balanceo = ["RandomUnder", "RandomOver", "NearMiss", "TomekLinks"]

#SEARCH ID3
directorio_actual = os.getcwd()
ruta_archivo = os.path.join(directorio_actual,"datasets/tratamiento.csv")
df = pd.read_csv(ruta_archivo)
X = df.drop(['AdministrarTratamiento', 'Paciente'], axis=1)
y = df[['AdministrarTratamiento']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for nro in n_estimadores:
    for profundidad in profundidad_max:
        for obs_n in minimas_obs_n:
            for obs_h in minimas_obs_h:
                for min_ganancia in ganancia_minima:
                    for mode in bootstrap:
                        for method in feature_selection_method:
                            for tecnica in tecnica_balanceo:
                                print(f"ID3-{nro}-{profundidad}-{obs_n}-{obs_h}-{min_ganancia}-{mode}-{method}-{tecnica}")
                                forest = RandomForestClassifier(algoritmo=ArbolID3,n_estimadores=nro, profundidad_max=profundidad ,minimas_obs_n=obs_n, minimas_obs_h=obs_h, ganancia_minima=min_ganancia, bootstrap=mode, feature_selection_method=method, tecnica_balanceo=tecnica)
                                forest.fit(X_train, y_train)

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

# #SEARCH C4.5
directorio_actual = os.getcwd()
ruta_archivo = os.path.join(directorio_actual,"datasets/star.csv")
df = pd.read_csv(ruta_archivo)
X = df.drop(['TargetClass', 'SpType'], axis=1)
y = df[['TargetClass']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for nro in n_estimadores:
    for profundidad in profundidad_max:
        for obs_n in minimas_obs_n:
            for obs_h in minimas_obs_h:
                for min_ganancia in ganancia_minima:
                    for mode in bootstrap:
                        for method in feature_selection_method:
                            for tecnica in tecnica_balanceo:
                                print(f"C4.5-{nro}-{profundidad}-{obs_n}-{obs_h}-{min_ganancia}-{mode}-{method}-{tecnica}")
                                forest = RandomForestClassifier(algoritmo=ArbolC4_5,n_estimadores=nro, profundidad_max=profundidad ,minimas_obs_n=obs_n, minimas_obs_h=obs_h, ganancia_minima=min_ganancia, bootstrap=mode, feature_selection_method=method, tecnica_balanceo=tecnica)
                                forest.fit(X_train, y_train)

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