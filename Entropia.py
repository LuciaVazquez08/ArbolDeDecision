import numpy as np

class Entropia:
    @staticmethod
    def entropia(y: np.ndarray) -> float: 
        clases, frecuencia_clase = np.unique(y, return_counts=True) 
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
