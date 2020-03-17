# -*- coding: utf-8 -*-

"""
Python First Test
Author: David Coelho
Last Change: 2018/08/24
"""

# Teste Python!

print("Hello World")

lista1 = [1,2,3,4,5]  # cria variavel do tipo lista
lista2 = lista1       # lista2 "aponta" para o mesmo espaço de memoria que lista1
lista1.append(7)      # append é feito no end de memoria de lista1 / lista2
print(lista2)         # mostra os valores de lista2 no console

x = 10                # cria variavel int
y = x + 2             # cria outra variavel (diferente do caso anterior)
print(x)              # mostra o valor de x no console
print(y)              # mostra o valor de y no console
