import numpy as np
from Arbol import Arbol
from Entropia import Entropia
from typing import TypeVar
T = TypeVar('T')

class ArbolID3(Arbol):
    
    def __init__(self, dato: T, atributo: str = None, es_hoja: bool = False) -> None:
        super().__init__(dato) 
        self._es_hoja = es_hoja
        self._hijos: dict = {}
        self._atributo_division = atributo
        self._num_samples = None

    def __str__(self, nivel=0) -> str:
        espacio_indentado = "    " * nivel
        if self._es_hoja:
            return f"{espacio_indentado}[Hoja: {self.dato}, Samples: {self._num_samples}]\n"
        else:
            nombre_atributo = self._atributo_division
            resultado = f"{espacio_indentado}[Atributo: {nombre_atributo}, Samples: {self._num_samples}]\n"
            for valor, hijo in self._hijos.items():
                resultado += f"{espacio_indentado}├── Valor: {valor}\n"
                resultado += hijo.__str__(nivel + 1)
            return resultado


    @classmethod
    def construir(cls, 
                  X: np.ndarray, 
                  y: np.ndarray, 
                  atributos: list[int],
                  nombres_atributos: list[str],
                  profundidad_max: int = None, 
                  minimas_obs_n: int = 0, 
                  minimas_obs_h: int = 0, 
                  ganancia_minima: float = 0.0, 
                  profundidad_actual: int = 0
                  ) -> "ArbolID3":
        
        # Criterio de parada: Nodo puro (todos los elementos del nodo pertenecen a la misma clase)
        if len(np.unique(y)) == 1:
            hoja = ArbolID3(y[0], None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
            
        # Criterio de parada: Maxima profundidad
        if profundidad_max is not None and profundidad_actual >= profundidad_max:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(y[0], atributo=nombres_atributos[atributos[0]], es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Mínimas observaciones por nodo
        if minimas_obs_n is not None and len(y) < minimas_obs_n:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(y[0], atributo=nombres_atributos[atributos[0]], es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        if not atributos:  # Sin atributos para dividir
            hoja = ArbolID3(y[0], None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Seleccionamos el mejor atributo en base a entropía y ganancia de información
        ganancias = [Entropia.ganancia_informacion_atributo(X, y, atributo) for atributo in atributos]

        # Criterio de parada: Ganancia mínima
        if ganancia_minima is not None and ganancias[np.argmax(ganancias)] < ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(y[0], atributo=nombres_atributos[atributos[0]], es_hoja=True)
            hoja._num_samples = len(y)
            return hoja

        # Creamos el árbol con el mejor atributo
        mejor_atributo = atributos[np.argmax(ganancias)]
        arbol = ArbolID3(mejor_atributo, atributo=nombres_atributos[mejor_atributo])
        arbol._num_samples = len(y)
        
        # Creamos nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]):
            
            atributos_restantes = atributos.copy()
            atributos_restantes.remove(mejor_atributo)
        
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            sub_X = X[indices]
            sub_y = y[indices]

            # Criterio de parada: Mínimas observaciones por hoja
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y):
                clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                subarbol = ArbolID3(valor=clase_mayoritaria, atributo=nombres_atributos[atributos[0]], es_hoja=True)
                subarbol._num_samples = len(sub_y)
            else:
                subarbol = cls.construir(sub_X, sub_y, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
                
            arbol._hijos[valor] = subarbol

        return arbol

    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]
    
    
        
        
    

    
        
        
    
