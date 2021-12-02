# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:36:23 2021

@author: Administrador (Andrés Felipe Escallón Portilla)
"""


import numpy as np

#REF: https://pythondiario.com/2019/01/matrices-en-python-y-numpy.html

P = np.array([[0.3,0.4,0.2,1/3,1/2],[1/2,0.35,0.2,0,0],[0.2,0,0.2,2/3,0],[0,0.05,0.2,0,0],[0,0.2,0.2,0,1/2]])

Pt = P.copy()

print('\n P= \n',P)
#print('\n P^50 = \n',P^50)

#Looping t from 2 to 50:
for t in range(2,50,1):
    #Pt = np.matmul(P, Pt)
    Pt = P.dot(Pt)    
    print(f'\P^{t}:\n{Pt}')

print(f'\LAST P^50:\n{Pt}') #Pt=P50

p01 = np.array([1,0,0,0,0]).transpose() # e
p02 = np.array([0,1,0,0,0]).transpose() # e2
p03 = np.array([0,0,1,0,0]).transpose() # e3
p04 = np.array([0,0,0,1,0]).transpose() # e4
p05 = np.array([0,0,0,0,1]).transpose() # e5
print('\n p0= \n',p01)


print('\n p(50)1=p01*P^50 =\n', p01.dot(Pt)[0]) #p(50)=p(0)*P^50
print('\n p(50)2=p02*P^50 =\n', p02.dot(Pt)[1]) #p(50)=p(0)*P^50
print('\n p(50)3=p03*P^50 =\n', p03.dot(Pt)[2]) #p(50)=p(0)*P^50
print('\n p(50)4=p04*P^50 =\n', p04.dot(Pt)[3]) #p(50)=p(0)*P^50
print('\n p(50)5=p05*P^50 =\n', p05.dot(Pt)[4]) #p(50)=p(0)*P^50

print('\n Da el mismo resultado anterior si extraigo los valores de la diagonal principal usando np.diag(Matriz)=\n', np.diag(Pt)) 

eigenvalue = 1

print(np.eye(5))


M = P - eigenvalue*np.eye(5) # M = A-lambda*I
print('\n M= \n',M)
print('\n det(M)= \n',np.linalg.det(M)) #como este det=0, no se puede calcular con la inversa, pero con la Matriz modificada si se puede porque el det!=0

Mmod = np.array([[0.3-eigenvalue,0.4,0.2,1/3,1/2],[1/2,0.35-eigenvalue,0.2,0,0],[0.2,0,0.2-eigenvalue,2/3,0],[0,0.05,0.2,0-eigenvalue,0],[1,1,1,1,1]])#-eigenvalue(-1) in the main diagonal, and last raw is replaced but the constraint (pi0+pi1+pi2+pi3+pi4=1)
print('\n Mmod= \n',Mmod)
print('\n det(Mmod)= \n',np.linalg.det(Mmod)) #**se puede con la inversa (det=!0) y también con np.alglin.solve()

print(np.array([0,0,0,0,1]))

b = np.array([0,0,0,0,1]).transpose()
print('\n b= \n',b)

#pi = b.dot(np.linalg.inv(Mmod))
#pi = np.matmul(b,np.linalg.inv(Mmod))
#pi = np.matmul(b,np.linalg.inv(M))
pi = np.matmul(np.linalg.inv(M),b.transpose())#**
print('\n pi= \n',pi)

#pimod = np.matmul(b,np.linalg.inv(Mmod))
pimod = np.matmul(np.linalg.inv(Mmod),b.transpose())#**
print('\n pimod= \n',pimod)


#pimod2 = (Mmod)\(b.transpose()) # division invertida (no funciona, en matlab si)
#pimod2 = np.divide(Mmod,b.transpose()) # division (es igua que /: divide elemento por elemento)
#print('\n pimod2= \n',pimod2)

x = np.linalg.solve(M, b.transpose())
print('\n x= \n',x)

suma_x = sum(x)
print('\n suma_x \n:', suma_x)

x_normalizado = (1/suma_x) * x

print('\n x_normalizado \n =', x_normalizado) #FUNCIONÓ!!!

Mdef = np.array([[x[0],0,0,0,0],[0,x[1],0,0,0],[0,0,x[2],0,0],[0,0,0,x[3],0],[1,1,1,1,1]])

xdef = np.linalg.solve(Mdef, b.transpose())
print('\n xdef= \n',xdef)


xmod = np.linalg.solve(Mmod, b.transpose())
print('\n xmod= \n',xmod)



# Nos devuelve un vector con los autovalores y una matriz con los autovectores
autovalores, autovectores = np.linalg.eig(P)
print('autovalores: ',autovalores)
print(f'Esta matriz tiene {len(autovalores)} autovalores')
print()

print('autovectores:',autovectores)
print('El autovalor 1 tiene asociado el siguiente autovector:')
print(autovectores[:,0])# está en la posición 0 de la lista

suma = sum(autovectores[:,0])# equivalente a autovectores[:,2][0]*5 porque son 5 iguales
print()
print('suma:',suma) # restricción para normalizar ya que la suma de las probabilidades debe ser 1*

probabilidades = (1/suma)*autovectores[:,0] #restricción para normalizar ya que la suma de las probabilidades debe ser 1*
print()
print('probabilidades:',probabilidades)
print()
print('restricción para normalizar ya que la suma de las probabilidades debe ser 1:')
print(sum(probabilidades))
