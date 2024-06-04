import numpy as np
from Entropia import Entropia

class GainRatio:
    
    @staticmethod
    def split_info(valores_del_atributo: np.ndarray) -> float:
        _, counts = np.unique(valores_del_atributo, return_counts=True)
        probabilidades = counts / len(valores_del_atributo)
        return -np.sum(probabilidades * np.log2(probabilidades))

    @staticmethod
    def gain_ratio(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float:
        ganancia_informacion = Entropia.ganancia_informacion_atributo(X, y, indice_atributo)
        split_info = GainRatio.split_info(X[:, indice_atributo])
        
        if split_info == 0:
            return 0  # Para evitar la división por cero
        
        gain_ratio = ganancia_informacion / split_info
        return gain_ratio