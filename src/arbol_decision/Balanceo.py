import numpy as np
import os
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
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
    
    # @staticmethod
    # def tomek_links(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
    #     n_samples = X.shape[0]

    #     encoder = OneHotEncoder()
    #     X_encoded = encoder.fit_transform(X)

    #     vecinos = NearestNeighbors(n_neighbors=2).fit(X_encoded)
    #     indices = vecinos.kneighbors(X_encoded, return_distance=False)

    #     indices = set()
        
    #     for i in range(n_samples):
    #         for j in range(1, 2):  
    #             if y[i] != y[indices[i][j]]:  
    #                 if indices[indices[i][j]][1] == i: 
    #                     indices.add(i)
    #                     indices.add(indices[i][j])

    #     X_filtrado = np.delete(X, list(indices), axis=0)
    #     y_filtrado = np.delete(y, list(indices))
        
    #     return X_filtrado, y_filtrado
    
    def tomek_links(X: np.ndarray, y: np.ndarray) -> (np.ndarray, np.ndarray):
        n_samples = X.shape[0]

        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)

        # Step 1: Find nearest neighbors
        vecinos = NearestNeighbors(n_neighbors=2).fit(X_encoded)
        indices = vecinos.kneighbors(X_encoded, return_distance=False)

        # Step 2: Identify Tomek links
        indices_to_keep = set()

        for i in range(n_samples):
            for j in range(1, 2):  # j starts from 1 because index 0 is the sample itself
                neighbor_index = indices[i][j]
                
                if y[i] != y[neighbor_index]:
                    if indices[neighbor_index][1] == i:  # Check if i is the nearest neighbor of neighbor_index
                        indices_to_keep.add(i)
                        indices_to_keep.add(neighbor_index)

        # Step 3: Filter X and y based on identified indices
        indices_to_keep = list(indices_to_keep)
        X_filtered = np.delete(X, indices_to_keep, axis=0)
        y_filtered = np.delete(y, indices_to_keep)

        return X_filtered, y_filtered

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

    def nearmiss_categorico(X, y):
        instancia_cercana = defaultdict(list)
        clases = np.unique(y)
        
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_encoded)
        
        distances, indices = nn.kneighbors(X_encoded)
        for i, clase in enumerate(clases):
            nearest_index = indices[i][0] 
            nearest_class = y[nearest_index]  
            
            if clase != nearest_class:
                instancia_cercana[clase].append((X[i], nearest_class))
        
        undersampled_X = []
        undersampled_y = []
        
        for clase, samples in instancia_cercana.items():
            for sample, nearest_class in samples:
                undersampled_X.append(sample)
                undersampled_y.append(clase)
        
        return np.array(undersampled_X), np.array(undersampled_y)

    
    