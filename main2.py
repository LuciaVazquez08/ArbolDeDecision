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


