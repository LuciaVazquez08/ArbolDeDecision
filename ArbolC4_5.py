from Ganancia import Ganancia
import numpy as np
from Arbol import Arbol
from typing import TypeVar
T = TypeVar('T')

class ArbolC4_5(Arbol):
    
    def __init__(self, dato = T, label: T = None, atributo: str = None, es_hoja: bool = False):
        super().__init__(dato) 
        self._es_hoja = es_hoja
        self._hijos = {}
        self.label = label
        self._atributo_division = atributo
        self._num_samples = None

    def __str__(self, nivel=0) -> str:
        espacio_indentado = "    " * nivel
        if self._es_hoja:
            return f"{espacio_indentado}[Hoja: {self.label}, Samples: {self._num_samples}]\n"
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
                  minimas_obs_n: int = 0, 
                  minimas_obs_h: int = 0, 
                  ganancia_minima: float = 0.0, 
                  profundidad_actual: int = 0
                  ) -> "ArbolC4_5":

        # Criterio de parada: Nodo puro 
        if len(np.unique(y)) == 1:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(None, label = clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
            
        # Criterio de parada: Maxima profundidad
        if profundidad_max is not None and profundidad_actual >= profundidad_max:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(None, label = clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Mínimas observaciones por nodo
        if minimas_obs_n is not None and len(y) < minimas_obs_n:
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(None, label = clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja
        
        # Criterio de parada: Sin atributos para dividir
        if not indice_atributos:  
            clase_mayoritaria = cls.clase_mayoritaria(y)
            hoja = ArbolC4_5(None, label = clase_mayoritaria, atributo=None, es_hoja=True)
            hoja._num_samples = len(y)
            return hoja

        # Seleccionar el mejor atributo
        mejor_atributo, mejor_umbral = cls.seleccionar_mejor_atributo(X, y, indice_atributos)

        # Creamos el árbol con el mejor atributo
        arbol = ArbolC4_5(mejor_atributo, atributo=nombres_atributos[mejor_atributo])
        arbol._num_samples = len(y)

        # Caso 1: El mejor atributo es continuo
        if mejor_umbral is not None:  

            atributos_restantes = indice_atributos.copy()
            atributos_restantes.remove(mejor_atributo)

            indices_izq = np.where(X[:, mejor_atributo] <= mejor_umbral)[0]
            indices_der = np.where(X[:, mejor_atributo] > mejor_umbral)[0]

            sub_X_izq = X[indices_izq]
            sub_y_izq = y[indices_izq]
            sub_X_der = X[indices_der]
            sub_y_der = y[indices_der]

            # Criterio de parada: Mínimas observaciones por hoja (no anda)
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y_izq): 
                clase_mayoritaria = cls.clase_mayoritaria(sub_y_izq)
                subarbol = ArbolC4_5(None,label= clase_mayoritaria, atributo=None, es_hoja=True)
                subarbol._num_samples = len(sub_y_izq)
            else:
                sub_arbol_izq = cls.construir(sub_X_izq, sub_y_izq, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)

            # Criterio de parada: Mínimas observaciones por hoja (no anda)
            if minimas_obs_h is not None and minimas_obs_h > len(sub_y_der): 
                clase_mayoritaria = cls.clase_mayoritaria(sub_y_der)
                subarbol = ArbolC4_5(None, label= clase_mayoritaria, atributo=None, es_hoja=True)
                subarbol._num_samples = len(sub_y_der)
            else:
                sub_arbol_der = cls.construir(sub_X_der, sub_y_der, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)

            arbol._hijos[('<=', mejor_umbral)] = sub_arbol_izq
            arbol._hijos[('>', mejor_umbral)] = sub_arbol_der

        # Caso 2: El mejor atributo es categórico
        else:  
            for valor in np.unique(X[:, mejor_atributo]):
                
                atributos_restantes = indice_atributos.copy()
                atributos_restantes.remove(mejor_atributo)
            
                indices = np.where(X[:, mejor_atributo] == valor)[0]
                sub_X = X[indices]
                sub_y = y[indices]

                # Criterio de parada: Mínimas observaciones por hoja
                if minimas_obs_h is not None and minimas_obs_h > len(sub_y):
                    clase_mayoritaria = cls.clase_mayoritaria(sub_y)
                    subarbol = ArbolC4_5(None, label = clase_mayoritaria, atributo=None, es_hoja=True)
                    subarbol._num_samples = len(sub_y)
                else:
                    subarbol = cls.construir(sub_X, sub_y, atributos_restantes, nombres_atributos, profundidad_max, minimas_obs_n, minimas_obs_h, ganancia_minima, profundidad_actual + 1)
                    
                arbol._hijos[valor] = subarbol

        return arbol

    @staticmethod
    def determinar_tipo_atributo(X: np.ndarray, top_n: int = 3, umbral: float = 0.8) -> str:
        tipo_atributo = 'continuo'  

        valores_unicos, conteos = np.unique(X, return_counts=True)
        
        if len(conteos) > 0:
        
            top_n = min(top_n, len(conteos))
    
            indices_ordenados = np.argsort(conteos)[::-1]
            top_conteos = conteos[indices_ordenados][:top_n]

            proporcion = np.sum(top_conteos) / len(X)

            if proporcion >= umbral:
                tipo_atributo = 'categorico'

        return tipo_atributo

    @staticmethod
    def seleccionar_mejor_atributo(X, y, atributos):
        mejor_ganancia = -np.inf
        mejor_atributo = None
        mejor_umbral = None

        for atributo in atributos:
            valores_atributo = X[:, atributo]

            # Verificar el tipo de atributo 
            tipo_atributo = ArbolC4_5.determinar_tipo_atributo(valores_atributo)

            if tipo_atributo == 'continuo':
                umbral, ganancia = ArbolC4_5.obtener_umbral_y_gain_ratio(valores_atributo, y)
            else:
                ganancia = Ganancia.ganancia_informacion_atributo(X, y, atributo)
                umbral = None

            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_atributo = atributo
                mejor_umbral = umbral

        return mejor_atributo, mejor_umbral

    @staticmethod
    def obtener_umbral_y_gain_ratio(atributo_continuo, y):
        ganancia_maxima = -1
        umbral_optimo = None
        
        # Ordenar los valores únicos del atributo continuo
        valores_unicos = np.sort(np.unique(atributo_continuo))
        
        for i in range(len(valores_unicos) - 1):
            umbral = (valores_unicos[i] + valores_unicos[i + 1]) / 2
            
            grupo_1_y = y[atributo_continuo <= umbral]
            grupo_2_y = y[atributo_continuo > umbral]

            n_izquierda = len(grupo_1_y)
            n_derecha = len(grupo_2_y)
            
            if n_izquierda == 0 or n_derecha == 0:
                continue
            
            # Crear una matriz X para gain_ratio
            X_dividido = np.concatenate((np.zeros(n_izquierda), np.ones(n_derecha)))
            y_dividido = np.concatenate((grupo_1_y, grupo_2_y))
            
            gain_ratio = Ganancia.gain_ratio(X_dividido.reshape(-1, 1), y_dividido, 0)
            
            if gain_ratio > ganancia_maxima:
                ganancia_maxima = gain_ratio
                umbral_optimo = umbral
        
        return umbral_optimo, ganancia_maxima

    @staticmethod
    def clase_mayoritaria(y):
        valores, counts = np.unique(y, return_counts=True)
        return valores[np.argmax(counts)]



    
            

    
            


    
            
