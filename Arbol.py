from typing import Generic, TypeVar

T = TypeVar('T')

class Arbol(Generic[T]):
    #ESTRUCTURA
    # Estructura con recursión directa múltiple
    def __init__(self, dato: T):
        self._dato: T = dato
        self._subarboles: list[Arbol[T]] = []
    
    # Decidimos mantener los atributos privados para convertirlos en propiedades
    @property
    def dato(self) -> T:
        return self._dato

    @dato.setter
    def dato(self, valor: T):
        self._dato = valor

    @property
    def subarboles(self) -> "list[Arbol[T]]":
        return self._subarboles
    
    @subarboles.setter
    def subarboles(self, subarboles: "list[Arbol[T]]"):
        self._subarboles = subarboles

    #METODOS ESPECIALES
    # Cantidad de nodos
    def __len__(self) -> int:
        if self.es_hoja():
            return 1
        else:
            return 1 + sum([len(subarbol) for subarbol in self.subarboles])

    def __eq__(self, otro: "Arbol[T]") -> bool:
        return isinstance(otro, Arbol) and self.dato == otro.dato and self.subarboles == otro.subarboles

    def __str__(self):
        def mostrar(t: Arbol[T], nivel: int):
            tab = '.' * 4
            indent = tab * nivel
            out = indent + str(t.dato) + '\n'
            for subarbol in t.subarboles:
                out += mostrar(subarbol, nivel + 1)
            return out
        return mostrar(self, 0)
    
    #METODOS

    # Extender árboles con nuevos descendientes
    def insertar_subarbol(self, subarbol: "Arbol[T]"):
        self.subarboles.append(subarbol)

    # Presenta si un nodo no tiene subárboles
    def es_hoja(self) -> bool:
        return self.subarboles == []
    
    def altura(self) -> int:
        def altura_n(bosque: list[Arbol[T]]) -> int:
            if not bosque:
                return 0
            else:
                return max(bosque[0].altura(), altura_n(bosque[1:]))
        return 1 + altura_n(self.subarboles)
    
    def preorder(self) -> list[T]:
        def preorder_n(bosque: list[Arbol[T]]) -> list[T]:
            return [] if not bosque else bosque[0].preorder() + preorder_n(bosque[1:])
        return [self.dato] + preorder_n(self.subarboles)

    def postorder(self) -> list[T]:
        def postorder_n(bosque: list[Arbol[T]]) -> list[T]:
            return [] if not bosque else bosque[0].postorder() + postorder_n(bosque[1:])
        return  postorder_n(self.subarboles) + [self.dato] 

    def bfs(self) -> list[T]:
        def recorrido(queue: list[Arbol], camino: list[T]) -> list[T]:
            if not queue:
                return camino
            nodo_actual = queue.pop(0)
            camino.append(nodo_actual.dato)
            if not nodo_actual.es_hoja():
                for subarbol in nodo_actual.subarboles:
                    queue.append(subarbol)
            return recorrido(queue, camino)
        return recorrido([self], [])

    def nivel(self, dato: T) -> int:
        return len(self.antecesores(dato)) + 1

    def pertenece(self, x: T) -> bool:
        if self.dato == x:
            return True
        for subarbol in self.subarboles:
            if subarbol.pertenece(x):
                return True
        return False
    
    def copy(self) -> "Arbol[T]":
        nuevo = Arbol(self.dato)
        for subarbol in self.subarboles:
            nuevo_subarbol = subarbol.copy()
            nuevo.insertar_subarbol(nuevo_subarbol)
        return nuevo
        
    def sin_hojas(self) -> "Arbol[T]":
        if self.es_hoja():
            return None
        nuevo = Arbol(self.dato)
        for subarbol in self.subarboles:
            a = subarbol.sin_hojas()
            if a is None:
                continue
            nuevo.insertar_subarbol(a)
        return nuevo

    # Version 4 -> Top-down, vamos llenando la lista y borrando a medida que encontremo y no encontremos
    def antecesores(self, dato: T) -> list[T]:
        def antecesor_interna(t: Arbol[T], antecesores: list[T]):
            if dato == t.dato:
                return antecesores
            elif t.es_hoja():
                return []
            else:
                i = 0
                antecesores.append(t.dato)
                resultado = []
                while i < len(t.subarboles) and not resultado:
                    resultado = antecesor_interna(t.subarboles[i], antecesores.copy())
                    i += 1
                return resultado
        if self.pertenece(dato):
            return antecesor_interna(self,[])
        else:
            raise ValueError("No existe ese dato")

    def recorrido_guiado(self, direcciones: list[int]) -> T:   # profundidad
        if not direcciones:
            return self.dato
        else:
            i = direcciones.pop(0)
            if i >= len(self.subarboles):
                raise Exception('No existe la dirección ingresada.')
            return self.subarboles[i].recorrido_guiado(direcciones)

    def podar(self, dato: T) -> "Arbol[T]":
        if self.dato == dato:
            return Arbol(self.dato)
        else:
            nuevo = Arbol(self.dato)
            for subarbol in self.subarboles:
                if subarbol.dato != dato:
                    nuevo.insertar_subarbol(subarbol.podar(dato))
            return nuevo
        
    def eliminar_nodo(self, dato: T):
<<<<<<< HEAD
        pass
=======
        pass
        
>>>>>>> 94e8d0dae1836c16a555d214c6ec724cf7225617
