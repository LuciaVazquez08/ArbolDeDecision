import numpy as np
from Arbol import Arbol
from Ganancia import Ganancia
from typing import TypeVar
T = TypeVar('T')

class ArbolID3(Arbol):
    
    def __init__(self, dato: T, atributo: str = None, es_hoja: bool = False) -> None:
        super().__init__(dato) 
        self._es_hoja = es_hoja
        self._hijos: dict = {}
        self._atributo_division = atributo
        self._num_samples: int = None

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
                  indice_atributos: list[int],
                  nombres_atributos: list[str],
                  profundidad_max: int = None, 
                  minimas_obs_n: int = None, 
                  minimas_obs_h: int = None, 
                  ganancia_minima: float = None, 
                  profundidad_actual: int = 0
                  ) -> "ArbolID3":
        
        # Criterio de parada: Nodo puro 
        if len(np.unique(y)) == 1:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
            
        # Criterio de parada: Maxima profundidad
        if profundidad_max is not None and profundidad_actual >= profundidad_max:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Mínimas observaciones por nodo
        if minimas_obs_n is not None and len(y) < minimas_obs_n:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Sin atributos para dividir
        if not indice_atributos:  
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Calculamos la ganancia de información de cada atributo
        ganancias = [Ganancia.ganancia_informacion_atributo(X, y, atributo) for atributo in indice_atributos]

        # Criterio de parada: Ganancia mínima
        if ganancia_minima is not None and ganancias[np.argmax(ganancias)] < ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja

        # Seleccionamos el atributo con mayor ganancia y creamos un arbol con ese atributo
        mejor_atributo = indice_atributos[np.argmax(ganancias)]
        arbol = ArbolID3(mejor_atributo, atributo=nombres_atributos[mejor_atributo])
        arbol._num_samples = len(y)
        
        # Creamos nodos para cada valor del mejor atributo
        for valor in np.unique(X[:, mejor_atributo]):
            
            atributos_restantes = indice_atributos.copy()
            atributos_restantes.remove(mejor_atributo)
        
            indices = np.where(X[:, mejor_atributo] == valor)[0]
            sub_X = X[indices]
            sub_y = y[indices]

            # Criterio de parada: Mínimas observaciones por hoja
            if minimas_obs_h is not None and len(sub_y) < minimas_obs_h:
                clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                subarbol = ArbolID3(clase_mayoritaria, atributo=None, es_hoja=True)
                subarbol._num_samples = len(sub_y)
            else:
                subarbol = cls.construir(sub_X, sub_y, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
                
            arbol._hijos[valor] = subarbol

        return arbol

    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]
    
    
        
        
    

    
        
        
    
