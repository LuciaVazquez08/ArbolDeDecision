from typing import List

class DecissionTreeClassifier:
    def __init__(self):
        self.tree = None
        
    
    # Implementa el entrenamiento del árbol
    def fit(self, X: List[List[float]], y: List[str]):
        self._contruir_arbol(X, y)

    # Implementa construcción del árbol
    def _build_tree(self, X: List[List[float]], y: List[str]):
        pass

    # Implementa la predicción
    def predict(self, X: List[List[float]]) -> List[str]:
        predicciones = []
        for instancia in X:
            predicciones.append(self._predict_instancia(instancia, self.tree))
        return predicciones
    
    # Implementa la predicción para una instancia específica
    def _predict_instance(self, instance: List[float], tree_node) -> str:
        pass
    
    #transf es para trabajar con los encodings
    #predict predice
    #fit entrena el modelo