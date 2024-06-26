from abc import ABC
import numpy as np

class Ganancia(ABC):

    """
    Clase abstracta que contiene métodos para calcular la ganancia de información, y otros cálculos relacionados, utilizados en algoritmos de construcción de árboles de decisión.
    """

    @staticmethod 
    def entropia(y: np.ndarray) -> float: 
        """
        Calcula la entropía de un conjunto de etiquetas.

        Parameters:
            y (np.ndarray): Array 1D de etiquetas.

        Returns:
            float: El valor de la entropía.
        """
        clase, frecuencia_clase = np.unique(y, return_counts=True) 
        probabilidad_clase = frecuencia_clase / len(y) 
        return -np.sum(probabilidad_clase * np.log2(probabilidad_clase))

    @staticmethod
    def ganancia_informacion_atributo(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float:
        """
        Calcula la ganancia de información de un atributo dado.

        Parameters:
            X (np.ndarray): Array 2D de características (atributos).
            y (np.ndarray): Array 1D de etiquetas de clase.
            indice_atributo (int): Índice del atributo para el cual se calculará la ganancia de información.

        Returns:
            float: El valor de la ganancia de información.
        """
        entropia_inicial = Ganancia.entropia(y)
        clases, frecuencia_clase = np.unique(X[:, indice_atributo], return_counts=True)
        entropia_ponderada = 0
        for clase, cantidad in zip(clases, frecuencia_clase):
            subconjunto_clase = y[X[:, indice_atributo] == clase]
            entropia_ponderada += (cantidad / len(y)) * Ganancia.entropia(subconjunto_clase)
        return entropia_inicial - entropia_ponderada
    
    @staticmethod
    def split_info(valores_del_atributo: np.ndarray) -> float:
        """
        Calcula el valor de información de división de un conjunto de valores de atributo.

        Parameters:
            valores_del_atributo (np.ndarray): Array 1D de valores de atributo.

        Returns:
            float: El valor de información de división.
        """
        _, counts = np.unique(valores_del_atributo, return_counts=True)
        probabilidades = counts / len(valores_del_atributo)
        return -np.sum(probabilidades * np.log2(probabilidades))

    @staticmethod
    def gain_ratio(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float:
        """
        Calcula el valor de información de división de un conjunto de valores de atributo.

        Parameters:
            valores_del_atributo (np.ndarray): Array 1D de valores de atributo.

        Returns:
            float: El valor de información de división.
        """
        ganancia_informacion = Ganancia.ganancia_informacion_atributo(X, y, indice_atributo)
        split_info = Ganancia.split_info(y)
        
        if split_info == 0:
            return 0  # Para evitar la división por cero
        
        gain_ratio = ganancia_informacion / split_info
        return gain_ratio

