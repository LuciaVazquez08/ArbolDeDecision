import numpy as np
import pandas as pd

class Entropia:

    @staticmethod 
    def entropia(y: np.ndarray) -> float: 
        clase, frecuencia_clase = np.unique(y, return_counts=True) 
        probabilidad_clase = frecuencia_clase / len(y) 
        return -np.sum(probabilidad_clase * np.log2(probabilidad_clase))

    @staticmethod
    def ganancia_informacion_atributo(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float:
        entropia_inicial = Entropia.entropia(y)
        clases, frecuencia_clase = np.unique(X[:, indice_atributo], return_counts=True)
        entropia_ponderada = 0
        for clase, cantidad in zip(clases, frecuencia_clase):
            subconjunto_clase = y[X[:, indice_atributo] == clase]
            entropia_ponderada += (cantidad / len(y)) * Entropia.entropia(subconjunto_clase)
        return entropia_inicial - entropia_ponderada
    
    @staticmethod
    def split_info(valores_del_atributo: np.ndarray) -> float:
        _, counts = np.unique(valores_del_atributo, return_counts=True)
        probabilidades = counts / len(valores_del_atributo)
        return -np.sum(probabilidades * np.log2(probabilidades))

    @staticmethod
    def gain_ratio(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float:
        ganancia_informacion = Entropia.ganancia_informacion_atributo(X, y, indice_atributo)
        split_info = Entropia.split_info(y)
        
        if split_info == 0:
            return 0  # Para evitar la división por cero
        
        gain_ratio = ganancia_informacion / split_info
        return gain_ratio

