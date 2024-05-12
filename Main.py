from Arbol import ArbolN

t = ArbolN(1)
n2 = ArbolN(2)
n3 = ArbolN(3)
n4 = ArbolN(4)
n5 = ArbolN(5)
n6 = ArbolN(6)
n7 = ArbolN(7)
n8 = ArbolN(8)
n9 = ArbolN(9)
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

# print(f'Antecesores: {t.antecesores_pred(9)}')
# print(f'Antecesores: {t.antecesores_fin(9)}')
# print(f'Antecesores: {t.antecesores_look(9)}')
# print(f'Antecesores: {t.antecesores(9)}')

# print(f'Recorrido guiado: {t.recorrido_guiado([2,0,0])}')

# print(f'BFS: {t.bfs()}')
# print(f'DFS preorder : {t.preorder()}')
# print(f'DFS posorder: {t.posorder()}')

# print(t.pertenece(10))

print(f'Nivel de 9: {t.nivel(9)}')
print(f'Nivel de 13: {t.nivel(14)}')

# t2 = t.copy()
# t3 = t2.sin_hojas()
# print(t)
# print(t2)
# print(t3)
# print(f't == t2 {t == t2}')

# print(f'recorrido_guiado [2,0,0]: {t2.recorrido_guiado([2,0,0])}')

