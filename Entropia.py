<<<<<<< HEAD
import numpy as np

#intento calcular la entropia de Shanon 
def entropia(y: np.ndarray) -> float: 
    clases_por_atributo, cantidad_cada_clase = np.unique(y, return_counts=True) 
    probabilidad_por_clase = cantidad_cada_clase / len(y) 
    return -np.sum(probabilidad_por_clase * np.log2(probabilidad_por_clase)) 

#calculo la ganancia 
def ganancia_informacion_atributo(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float: 
    entropia_inicial = entropia(y) 
    clases, cantidad_por_clase = np.unique(X[:, indice_atributo], return_counts=True) 
    entropia_ponderada = 0 
    for cla, cant in enumerate(clases): 
        cantidad = cantidad_por_clase[cla] 
        subconjunto_clase = y[X[:, indice_atributo] == clases[cla]] 
        entropia_ponderada += (cantidad / len(y)) * entropia(subconjunto_clase) 
    return entropia_inicial - entropia_ponderada
=======
import numpy as np

class Entropia:
    def entropia(y: np.ndarray) -> float: 
        clases_por_atributo, cantidad_cada_clase = np.unique(y, return_counts=True) 
        probabilidad_por_clase = cantidad_cada_clase / len(y) 
        return -np.sum(probabilidad_por_clase * np.log2(probabilidad_por_clase)) 

    def ganancia_informacion_atributo(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float: 
        entropia_inicial = Entropia.entropia(y) 
        clases, cantidad_por_clase = np.unique(X[:, indice_atributo], return_counts=True) 
        entropia_ponderada = 0 
        for cla, cant in enumerate(clases): 
            cantidad = cantidad_por_clase[cla] 
            subconjunto_clase = y[X[:, indice_atributo] == clases[cla]] 
            entropia_ponderada += (cantidad / len(y)) * Entropia.entropia(subconjunto_clase) 
        return entropia_inicial - entropia_ponderada
>>>>>>> 94e8d0dae1836c16a555d214c6ec724cf7225617
