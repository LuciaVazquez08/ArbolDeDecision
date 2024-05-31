import numpy as np
from Arbol import Arbol
from Entropia import Entropia

class ArbolID3(Arbol):
    
    def __init__(self, dato = None, es_hoja: bool = False) -> None:
        super().__init__(dato)
        self._es_hoja = es_hoja
        self._hijos: dict = {}

    def __str__(self):
        def mostrar(t: ArbolID3, nivel: int):
            tab = '.' * 4
            indent = tab * nivel
            out = indent + str(t.dato) + '\n'
            for valor, subarbol in t._hijos.items():
                out += indent + f"Valor: {valor}\n"
                out += mostrar(subarbol, nivel + 1)
            return out
        return mostrar(self, 0)
    
    @classmethod
    # X: dataset convertido en arrays sin la primer columna de atributos
    # y: columna con las clases
    def construir(cls, X: np.ndarray, y: np.ndarray, 
                  atributos: list[int],
                  profundidad_max: int = None, 
                  minimas_obs_n: int = None, 
                  minimas_obs_h: int = None, 
                  ganancia_minima: float = 0.0, 
                  profundidad_actual: int = 0
                  ) -> "ArbolID3":
        
        # Criterio de parada: Nodo puro (todos los elementos del nodo pertenecen a la misma clase)
        if len(np.unique(y)) == 1:
            return ArbolID3(y[0], es_hoja=True)
        
        # Criterio de parada: Maxima profundidad
        if profundidad_max is not None and profundidad_actual >= profundidad_max:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Criterio de parada: Mínimas observaciones por nodo
        if minimas_obs_n is not None and len(y) < minimas_obs_n:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Seleccionamos el mejor atributo en base a entropía y ganancia de información
        ganancias = [Entropia.ganancia_informacion_atributo(X, y, atributo) for atributo in atributos]
        mejor_atributo = atributos[np.argmax(ganancias)]
        
        # Criterio de parada: Ganancia mínima
        if ganancia_minima is not None and ganancias[np.argmax(ganancias)] < ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Criterio de parada: No quedan atributos
        if not atributos:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Creamos el árbol con el mejor atributo
        arbol = ArbolID3(mejor_atributo)
        
        atributos_restantes = atributos.copy()
        atributos_restantes.remove(mejor_atributo)
        
        # Creamos nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]):
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            sub_X = X[indices]
            sub_y = y[indices]

            # Criterio de parada: Mínimas observaciones por hoja
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y):
                clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                subarbol = ArbolID3(clase_mayoritaria, es_hoja=True)
            else:
                subarbol = cls.construir(sub_X, sub_y, atributos_restantes, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
            
            arbol._hijos[valor] = subarbol
            
        print(arbol)
        return arbol

    
    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]
    
        
        
    
