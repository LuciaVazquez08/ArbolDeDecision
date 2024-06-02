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
    
    def split_info(valores_del_atributo: np.ndarray) -> float:
        _, counts = np.unique(valores_del_atributo, return_counts=True)
        probabilidades = counts / len(valores_del_atributo)
        return -np.sum(probabilidades * np.log2(probabilidades))
    
    def gain_ratio(X: np.ndarray, y: np.ndarray, indice_atributo: int) -> float:
        ganancia_informacion = Entropia.ganancia_informacion_atributo(X, y, indice_atributo)
        split_info = Entropia.split_info(X[:, indice_atributo])
        
        if split_info == 0:
            return 0  # Para evitar la división por cero
        
        gain_ratio = ganancia_informacion / split_info
        return gain_ratio
