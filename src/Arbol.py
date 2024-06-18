from typing import Generic, TypeVar

T = TypeVar('T')

class Arbol(Generic[T]):

    """
    Implementación genérica de un árbol n-ario.

    Parámetros
    ----------
    dato : T 
        El dato almacenado en el nodo del árbol.

    Atributos
    ---------
    _subarboles : list[Arbol[T]] 
        Lista de subárboles que parten de este nodo.
    """

    def __init__(self, dato: T):
        self._dato: T = dato
        self._subarboles: list[Arbol[T]] = []


    @property
    def dato(self) -> T:
        """
        Obtiene el dato almacenado en el nodo.

        Returns
        -------
        T : Devuelve el dato almacenado en el nodo.
        """
        return self._dato


    @dato.setter
    def dato(self, valor: T):
        """
        Establece el dato almacenado en el nodo.

        Parámetros
        ----------
        valor : T 
            El nuevo valor del dato del nodo.
        """
        self._dato = valor


    @property
    def subarboles(self) -> "list[Arbol[T]]":
        """
        Obtiene la lista de subárboles del nodo.

        Returns
        -------
        list[Arbol[T]] : Devuelve la lista de subárboles del nodo.
        """
        return self._subarboles
    

    @subarboles.setter
    def subarboles(self, subarboles: "list[Arbol[T]]"):
        """
        Establece la lista de subárboles del nodo.

        Parámetros
        ----------
        subarboles : list[Arbol[T]]
            La nueva lista de subárboles del nodo.
        """
        self._subarboles = subarboles


    def __len__(self) -> int:
        """
        Calcula la cantidad de nodos en el árbol.

        Returns
        -------
        int : Devuelve la cantidad de nodos en el árbol.
        """
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


    def insertar_subarbol(self, subarbol: "Arbol[T]"):
        """
        Inserta un subárbol en el nodo actual.

        Parámetros
        ----------
        subarbol : Arbol[T]
            El subárbol a insertar.
        """
        self.subarboles.append(subarbol)


    def es_hoja(self) -> bool:
        """
        Determina si el nodo actual es una hoja.

        Returns
        -------
        bool : Devuelve True si el nodo es una hoja, False de lo contrario.
        """
        return self.subarboles == []
    

    def altura(self) -> int:
        """
        Calcula la altura del árbol, es decir, la mayor longitud encontrada desde la raíz hasta una hoja.

        Returns
        -------
        int : Devuelve la altura del árbol.
        """
        def altura_n(bosque: list[Arbol[T]]) -> int:
            if not bosque:
                return 0
            else:
                return max(bosque[0].altura(), altura_n(bosque[1:]))
        return 1 + altura_n(self.subarboles)
    
    def preorder(self) -> list[T]:
        """
        Realiza un recorrido en profundidad preorder del árbol, comenzando desde la raíz y luego visitando los subárboles.

        Returns
        -------
        list[T] : Devuelve una lista con los datos de los nodos en recorrido preorder.
        """
        def preorder_n(bosque: list[Arbol[T]]) -> list[T]:
            return [] if not bosque else bosque[0].preorder() + preorder_n(bosque[1:])
        return [self.dato] + preorder_n(self.subarboles)


    def postorder(self) -> list[T]:
        """
        Realiza un recorrido en profundidad postorder del árbol, visitando primero los subárboles y luego la raíz.

        Returns
        -------
        list[T] : Devuelve una lista con los datos de los nodos en recorrido postorder.
        """
        def postorder_n(bosque: list[Arbol[T]]) -> list[T]:
            return [] if not bosque else bosque[0].postorder() + postorder_n(bosque[1:])
        return  postorder_n(self.subarboles) + [self.dato] 


    def bfs(self) -> list[T]:
        """
        Realiza un recorrido a lo ancho del árbol, visitando todos los nodos de un mismo nivel antes de pasar al siguiente nivel.

        Returns
        -------
        list[T] : Devuelve una lista con los datos de los nodos con recorrido BFS.
        """
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
        """
        Obtiene el nivel del nodo en el árbol. El nivel de la raíz es 1.

        Parámetros
        ----------
        dato : T
            El dato del nodo cuyo nivel se desea conocer.

        Returns
        -------
        int : Devuelve el nivel del nodo en el árbol.
        """
        return len(self.antecesores(dato)) + 1


    def pertenece(self, dato: T) -> bool:
        """
        Determina si un valor dado pertenece al árbol.

        Parámetros
        ----------
        dato : T 
            El dato que se desea verificar si está presente en el árbol.

        Returns
        -------
        bool : Devuelve True si el dato está presente en el árbol, False de lo contrario.
        """
        if self.dato == dato:
            return True
        for subarbol in self.subarboles:
            if subarbol.pertenece(dato):
                return True
        return False
    

    def copy(self) -> "Arbol[T]":
        """
        Crea una copia del árbol actual.

        Returns
        -------
        Arbol[T] : Una copia del árbol actual.
        """
        nuevo = Arbol(self.dato)
        for subarbol in self.subarboles:
            nuevo_subarbol = subarbol.copy()
            nuevo.insertar_subarbol(nuevo_subarbol)
        return nuevo
        

    def sin_hojas(self) -> "Arbol[T]":
        """
        Elimina todas las hojas del árbol, creando un nuevo árbol sin ellas.

        Returns
        -------
        Arbol[T] : Un nuevo árbol sin hojas.
        """
        if self.es_hoja():
            return None
        nuevo = Arbol(self.dato)
        for subarbol in self.subarboles:
            a = subarbol.sin_hojas()
            if a is None:
                continue
            nuevo.insertar_subarbol(a)
        return nuevo


    def antecesores(self, dato: T) -> list[T]:
        """
        Obtiene los antecesores de un nodo en el árbol.

        Parámetros
        ----------
        dato : T 
            El dato del nodo cuyos antecesores se desean obtener.

        Returns
        -------
        list[T] : Devuelve una lista con los datos de los antecesores del nodo, en orden desde el nodo actual hasta la raíz del árbol.
                     
        Raises
        ------
        ValueError : Si el dato no está presente en el árbol.
        """
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

    def recorrido_guiado(self, direcciones: list[int]) -> T: 
        """
        Realiza un recorrido guiado del árbol, siguiendo una lista de direcciones.

        Parámetros
        ----------
        direcciones : list[int] 
            Una lista de índices que indican el camino a seguir en el árbol. 
            Cada índice representa el hijo que se debe visitar desde el nodo actual.

        Returns
        -------
        T : Devuelve el dato del nodo alcanzado después de seguir las direcciones.

        Raises
        ------
        Exception : Si una dirección dada no es válida.
        """
        if not direcciones:
            return self.dato
        else:
            i = direcciones.pop(0)
            if i >= len(self.subarboles):
                raise Exception('No existe la dirección ingresada.')
            return self.subarboles[i].recorrido_guiado(direcciones)


    def podar(self, dato: T) -> "Arbol[T]":
        """
        Elimina un nodo y todos sus descendientes del árbol.

        Parámetros
        ----------
        dato : T 
            El dato del nodo que se desea eliminar.

        Returns
        -------
        Arbol[T] : Devuelve un nuevo árbol que resulta de podar el nodo especificado.
        """
        if self.dato == dato:
            return Arbol(self.dato)
        else:
            nuevo = Arbol(self.dato)
            for subarbol in self.subarboles:
                if subarbol.dato != dato:
                    nuevo.insertar_subarbol(subarbol.podar(dato))
            return nuevo
        
