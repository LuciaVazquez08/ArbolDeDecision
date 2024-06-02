import numpy as np
from Arbol import Arbol
from Entropia import Entropia

from typing import Generic, TypeVar
T = TypeVar('T')

class ArbolID3(Arbol):
    
    def __init__(self, dato, es_hoja: bool = False) -> None:
        super().__init__(dato) 
        self._es_hoja = es_hoja
        self._hijos: dict = {}

    def __str__(self, nivel=0) -> str:
        # Espacio de indentación basado en el nivel de profundidad
        espacio_indentado = "    " * nivel
        if self._es_hoja:
            return f"{espacio_indentado}[Hoja: {self.dato}]\n"
        else:
            resultado = f"{espacio_indentado}[Nodo: {self.dato}]\n"
            for valor, hijo in self._hijos.items():
                resultado += f"{espacio_indentado}├── Valor: {valor}\n"
                resultado += hijo.__str__(nivel + 1)
            return resultado

    @classmethod
    # X: dataset convertido en arrays sin la primer columna de atributos
    # y: columna con las clases

    def construir(cls, 
                  X: np.ndarray, 
                  y: np.ndarray, 
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
        
        # Criterio de parada: Sin atributos para dividir
        if not atributos:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        # Seleccionamos el mejor atributo en base a entropía y ganancia de información
        ganancias = [Entropia.ganancia_informacion_atributo(X, y, atributo) for atributo in atributos]
        mejor_atributo = atributos[np.argmax(ganancias)]
        
        #print(f"Ganancias en profundidad {profundidad_actual}: {ganancias}")
        #print(f"Mejor atributo en profundidad {profundidad_actual}: {mejor_atributo} con ganancia {ganancias[np.argmax(ganancias)]}")
        
        # Criterio de parada: Ganancia mínima
        if ganancia_minima is not None and ganancias[np.argmax(ganancias)] < ganancia_minima:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)
        
        if not atributos:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            return ArbolID3(clase_mayoritaria, es_hoja=True)

        # Creamos el árbol con el mejor atributo
        arbol = ArbolID3(mejor_atributo)
        
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
                subarbol = ArbolID3(valor=clase_mayoritaria, es_hoja=True)
            else:
                subarbol = cls.construir(sub_X, sub_y, atributos_restantes, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
            
            arbol._hijos[valor] = subarbol
        
        return arbol

    @staticmethod
    def clase_mayoritaria(y: np.ndarray) -> int:
        clases, conteo = np.unique(y, return_counts=True)
        return clases[np.argmax(conteo)]
    
        
        
    
