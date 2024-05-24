from Arbol import Arbol

if __name__ == '__main__':
    t = Arbol(1)
    n2 = Arbol(2)
    n3 = Arbol(3)
    n4 = Arbol(4)
    n5 = Arbol(5)
    n6 = Arbol(6)
    n7 = Arbol(7)
    n8 = Arbol(8)
    n9 = Arbol(9)
    t.insertar_subarbol(n2)
    t.insertar_subarbol(n3)
    t.insertar_subarbol(n4)
    n2.insertar_subarbol(n5)
    n2.insertar_subarbol(n6)
    n4.insertar_subarbol(n7)
    n4.insertar_subarbol(n8)
    n7.insertar_subarbol(n9)

    print(t)

    # print(f'Altura: {t.altura()}')
    # print(f'Nodos: {len(t)}')
    print(f'Nivel de 9: {t.nivel(9)}')

    # print(f'BFS: {t.bfs()}')
    # print(f'DFS preorder : {t.preorder()}')
    # print(f'DFS postorder: {t.postorder()}')

    # print(f'Antecesores: {t.antecesores(9)}')
    # print(f'Recorrido guiado: {t.recorrido_guiado([2,0,0])}')

    t2 = t.podar(7)
    print(t2)

    # print(t.pertenece(10))

    # print(f'recorrido_guiado [2,0,0]: {t2.recorrido_guiado([2,0,0])}')
<<<<<<< HEAD
    
=======

>>>>>>> 94e8d0dae1836c16a555d214c6ec724cf7225617
