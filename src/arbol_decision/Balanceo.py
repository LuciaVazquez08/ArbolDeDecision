import numpy as np
import os
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
class Balanceo:
    @staticmethod
    def calcular_distancia(x1, x2):
        return np.linalg.norm(x1 - x2)

    @staticmethod
    def random_undersample(X: np.ndarray, y: np.ndarray) -> (np.ndarray , np.ndarray):
        clases_target = np.unique(y)
        indices = []
        state = np.random.RandomState(42)

        tamaño = np.inf
        for clase in clases_target:
            indices_clase = np.where(y == clase)[0]
            tamaño_clase = len(indices_clase)
            if tamaño_clase < tamaño:
                tamaño = tamaño_clase
        
        for clase in clases_target:
            target_indices = np.where(y == clase)[0]
            indx = state.choice(target_indices, size= tamaño, replace=False)
            indices.extend(indx)
        
        X_filtrado = X[indices, :]
        y_filtrado = y[indices, :]
        
        return X_filtrado, y_filtrado

    def random_oversample(X: np.ndarray, y: np.ndarray) -> (np.ndarray , np.ndarray):
        clases_target = np.unique(y)
        indices = []
        state = np.random.RandomState(42)

        tamaño = -np.inf
        for clase in clases_target:
            indices_clase = np.where(y == clase)[0]
            tamaño_clase = len(indices_clase)
            if tamaño_clase > tamaño:
                tamaño = tamaño_clase
        
        for clase in clases_target:
            target_indices = np.where(y == clase)[0]
            indx = state.choice(target_indices, size=tamaño, replace=True)
            indices.extend(indx)
        
        X_filtrado = X[indices, :]
        y_filtrado = y[indices, :]
        
        return X_filtrado, y_filtrado
    
    @staticmethod
    def tomek_links(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        # Identificar los índices de las muestras de diferentes clases
        n_samples = X.shape[0]
        neighbors = NearestNeighbors(n_neighbors=2).fit(X)
        indices = neighbors.kneighbors(X, return_distance=False)
        
        toremove_indices = set()
        
        for i in range(n_samples):
            for j in range(1, 2):  # Solo el vecino más cercano
                if y[i] != y[indices[i][j]]:  # Verificar si son de clases diferentes
                    if indices[indices[i][j]][1] == i:  # Verificar enlace bidireccional
                        toremove_indices.add(i)
                        toremove_indices.add(indices[i][j])
        
        # Eliminar las muestras que forman enlaces de Tomek
        X_filtrado = np.delete(X, list(toremove_indices), axis=0)
        y_filtrado = np.delete(y, list(toremove_indices))
        
        return X_filtrado, y_filtrado

    @staticmethod
    def nearmiss(X, y):
        instancia_cercana = defaultdict(list)
        clases = np.unique(y)
        
        for clase in clases:
            index = np.where(y == clase)[0]
            instancias = X[index]
            
            for instancia in instancias:
                distancias = []
                for label in np.unique(y):
                    if label != clase:
                        indices_clase = np.where(y == label)[0]
                        instancia_clase = X[indices_clase]
                        for sample in instancia_clase:
                            distancia = Balanceo.calcular_distancia(instancia, sample)
                            distancias.append((distancia, label))
                
                distancias.sort()
                vecino_mas_cercano = distancias[:1]
                
                for distancia, nearest_class in vecino_mas_cercano:
                    instancia_cercana[clase].append((instancia, nearest_class))
        
        undersampled_X = []
        undersampled_y = []
        for clase, samples in instancia_cercana.items():
            for sample, nearest_class in samples:
                undersampled_X.append(sample)
                undersampled_y.append(clase)
        
        return np.array(undersampled_X), np.array(undersampled_y)
    
    @staticmethod
    def smote(X, y):
        clases = np.unique(y)
        
        n_clase_mayoritaria = 0

        for clase in clases:
            instancias = np.where(y== clase)[0]
            if len(instancias) > :

        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        for clase in clases:
            instancias = np.where(y == clase)[0]
            if len(instancias) < n_clase_mayoritaria:
                n_muestras_necesarias = n_clase_mayoritaria - len(instancias)
                X_clase = X[instancias]
                
                nn = NearestNeighbors(n_neighbors=6)
                nn.fit(X_clase)
                neighbors = nn.kneighbors(X_clase, return_distance=False)[:, 1:]
                
                synthetic_samples = []
                for _ in range(n_muestras_necesarias):
                    idx = np.random.choice(range(len(X_clase)))
                    neighbor_indices = neighbors[idx]
                    
                    synthetic_instance = np.array([
                        np.random.choice(X_clase[neighbor_indices, col]) for col in range(X_clase.shape[1])
                    ])
                    
                    synthetic_samples.append(synthetic_instance)
                
                synthetic_samples = np.array(synthetic_samples)  # Convertir a array NumPy
                
                X_resampled = np.vstack((X_resampled, synthetic_samples))
                y_resampled = np.hstack((y_resampled, np.array([clase] * n_muestras_necesarias)))
        
        return X_resampled, y_resampled

# if __name__ == "__main__":
#     data = {
#         'feature1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
#         'feature2': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z'],
#         'feature3': ['M', 'N', 'M', 'O', 'N', 'M', 'O', 'N', 'M', 'O'],
#         'feature4': ['P', 'Q', 'P', 'R', 'Q', 'P', 'R', 'Q', 'P', 'R'],
#         'feature5': ['S', 'T', 'S', 'U', 'T', 'S', 'U', 'T', 'S', 'U'],
#         'feature6': ['V', 'W', 'V', 'X', 'W', 'V', 'X', 'W', 'V', 'X'],
#         'target':  [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
#     }

#     df = pd.DataFrame(data)

#     X = df.drop('target', axis=1).values
#     y = df['target'].values

#     balance = df['TargetClass'].value_counts() 
#     print(balance)

#     X = df.drop(['TargetClass', 'SpType'], axis=1)
#     y = df[['TargetClass']]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     classifier = RandomForestClassifier(algoritmo = "C4.5", tecnica_balanceo="TomekLinks")
#     classifier.fit(X_train, y_train)

#     # Evaluamos el modelo
#     y_pred = classifier.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = recall_score(y_test, y_pred, average= 'weighted')
#     recall = f1_score(y_test, y_pred, average= 'weighted')
#     matriz = confusion_matrix(y_test, y_pred)
#     print(f'Accuracy: {accuracy}')
#     print(f'Precision: {precision}')
#     print(f'Recall: {recall}')
#     print(f'{matriz}')
