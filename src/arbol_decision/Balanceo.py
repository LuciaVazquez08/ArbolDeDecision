import numpy as np
import os
import pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
class Balanceo:
    @staticmethod
    def random_undersample(X: np.ndarray, y: np.ndarray) -> (np.ndarray , np.ndarray):
        """
        Elimina filas de datos de manera aleatoria para que la cantidad de instancias de cada clase sea la misma.

        Parámetros
        ----------
        X: np.ndarray
            Conjunto de entrada
        y: np.ndarray
            Etiquetas correspondientes a X

        Returns
        -------
        (np.ndarray , np.ndarray) : Devuelve un  par ordenado con las instanicas y sus target reducidos y balanceados.

        """
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
        """
        Repite filas de datos de manera aleatoria para que la cantidad de instancias de cada clase sea la misma.

        Parámetros
        ----------
        X: np.ndarray
            Conjunto de entrada
        y: np.ndarray
            Etiquetas correspondientes a X

        Returns
        -------
        (np.ndarray , np.ndarray) : Devuelve un  par ordenado con las instanicas y sus target aumentados y balanceados.

        """
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
        """
        Elimina filas de datos buscando dos vecinos cercanos cuyas clases sean distintas.

        Parámetros
        ----------
        X: np.ndarray
            Conjunto de entrada
        y: np.ndarray
            Etiquetas correspondientes a X

        Returns
        -------
        (np.ndarray , np.ndarray) : Devuelve un  par ordenado con las instanicas y sus target reducidos y balanceados.

        """
        n_samples = X.shape[0]

        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)

        vecinos = NearestNeighbors(n_neighbors=2).fit(X_encoded)
        indices = vecinos.kneighbors(X_encoded, return_distance=False)

        indices_eliminar = set()

        for i in range(n_samples):
            for j in range(1, 2): 
                indices_vecinos = indices[i][j]
                
                if y[i] != y[indices_vecinos]:
                    if indices[indices_vecinos][1] == i: 
                        indices_eliminar.add(i)
                        indices_eliminar.add(indices_vecinos)

        indices_eliminar = list(indices_eliminar)
        X_filtrado = np.delete(X, indices_eliminar, axis=0)
        y_filtrado = np.delete(y, indices_eliminar)

        return X_filtrado, y_filtrado

    @staticmethod
    def nearmiss(X: np.ndarray, y:np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Metodo de undersamppling que elimina filas de datos buscando instancias de la clase mayoritaria que esten cerca, en cuanto a similitud de features, a las de la clase minoritaria.
        Esta implementacion se usa junto al algoritmo C4.5

        Parámetros
        ----------
        X: np.ndarray
            Conjunto de datos de entrada.
        y: np.ndarray
            Etiquetas correspondientes a X.

        Returns
        -------
        (np.ndarray , np.ndarray) : Devuelve un  par ordenado con las instanicas y sus target reducidos y balanceados.

        """
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
                            distancia = np.linalg.norm(instancia, sample)
                            distancias.append((distancia, label))
                
                distancias.sort()
                vecino_mas_cercano = distancias[:1]
                
                for distancia, nearest_class in vecino_mas_cercano:
                    instancia_cercana[clase].append((instancia, nearest_class))
        
        X_reducido = []
        y_reducido = []
        for clase, samples in instancia_cercana.items():
            for sample, nearest_class in samples:
                X_reducido.append(sample)
                y_reducido.append(clase)
        
        return np.array(X_reducido), np.array(y_reducido)

    def nearmiss_categorico(X, y):
        """
        Metodo de undersamppling que elimina filas de datos buscando instancias de la clase mayoritaria que esten cerca, en cuanto a similitud de features, a las de la clase minoritaria.
        Esta implementacion se usa junto al algoritmo C4.5

        Parámetros
        ----------
        X: np.ndarray
            Conjunto de datos de entrada.
        y: np.ndarray
            Etiquetas correspondientes a X.

        Returns
        -------
        (np.ndarray , np.ndarray) : Devuelve un  par ordenado con las instanicas y sus target reducidos y balanceados.

        """
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
        
        X_reducido = []
        y_reducido = []
        
        for clase, samples in instancia_cercana.items():
            for sample, nearest_class in samples:
                X_reducido.append(sample)
                y_reducido.append(clase)
        
        return np.array(X_reducido), np.array(y_reducido)

    
    