# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:23:32 2021

@author: Administrador (Andrés Felipe Escallón Portilla)
"""

##############REFERENCES################################
'''
https://claudiovz.github.io/scipy-lecture-notes-ES/packages/sympy.html
https://docs.sympy.org/latest/modules/vector/index.html
https://jorgedelossantos.github.io/apuntes-python/SymPy.html
https://www.matesfacil.com/matrices/metodo-matriz-inversa-resolver-sistemas-ecuaciones-lineales-ejemplos.html
https://ernestocrespo13.wordpress.com/2015/02/21/resolucion-de-sistemas-de-ecuaciones-con-sympy/
http://research.iac.es/sieinvens/python-course/sympy.html
https://relopezbriega.github.io/blog/2015/06/14/algebra-lineal-con-python/
https://www.superprof.es/diccionario/matematicas/algebralineal/regla-cramer.html
https://es.acervolima.com/2021/02/09/python-sympy-metodo-matrix-eigenvects/
'''
#######################################################

import sympy as sp
from sympy import *
from sympy import Matrix
from sympy import Symbol
#from sympy.matrices import Matrix

#*****************************************************
print('Testing first...')
print(Matrix([[1,0], [0,1]]))

x = Symbol('x')
y = Symbol('y')

A = Matrix([[1,x], [y,1]])

print(A)
print(A**2)
#*****************************************************

print()
print("birth_death_markov_chain:")
print()

p0 = Symbol('p0')
p1 = Symbol('p1')
p2 = Symbol('p2')
p3 = Symbol('p3')

q1 = Symbol('q1')
q2 = Symbol('q2')
q3 = Symbol('q3')


#pi0 = Symbol('pi0')
#pi1 = Symbol('pi1')
#pi2 = Symbol('pi2')
#pi3 = Symbol('pi3')
(pi0,pi1,pi2,pi3) = symbols("pi0,pi1,pi2,pi3")

P = Matrix( [ [1-p0, p0, 0, 0],  [q1, 1-p1-q1, p1, 0], [0, q2, 1-p2-q2, p2], [0, 0, q3, 1-q3]  ] )
print('\n P:\n', P)
print('\n det(P):\n', P.det())

Pi = Matrix( [pi0,pi1,pi2,pi3] ).T #row vector = column vector transposed 
print('\nPi:\n',Pi)

PiP = Pi * P
print('\nPiP:\n',PiP)

#res = Pi * (P.inv()) # this is not the solution because it does not involve the relationship A*x=lamba*x, with lambda=1 (it is considering as if it were A*x=b with b a vector of constants)
#res0 = Pi * (P.inv()) # this is not the solution because it does not involve the relationship A*x=lamba*x, with lambda=1 (it is considering as if it were A*x=b with b a vector of constants)

eigenvalue = 1
#Mmod = Matrix( [ [1-p0-eigenvalue, p0, 0, 0],  [q1, 1-p1-q1-eigenvalue, p1, 0], [0, q2, 1-p2-q2-eigenvalue, p2], [1, 1, 1, 1]  ] ).T#-eigenvalue(-1) in the main diagonal, and last raw is replaced but the constraint (pi0+pi1+pi2+pi3+pi4=1)
Mmod = Matrix([[-p0, q1, 0, 0], [p0, -p1 - q1, q2, 0], [0, p1, -p2 - q2, 1], [1, 1, 1, 1]]).T
print('\n Mmod= \n',Mmod)
print('\n det(Mmod)= \n',Mmod.det()) #**se puede con la inversa (det=!0) y también con solve()


#PiMmod = Pi * Mmod
PiMmod = Matrix([[-p0*pi0 + pi1*q1, p0*pi0 + pi1*(-p1 - q1) + pi2*q2, p1*pi1 + pi2*(-p2 - q2) + q3, pi0 + pi1 + pi2 + pi3]])

print('\nPiMmod:\n',PiMmod)

#b = Matrix([0,0,0,1]).T #row vector = column vector transposed 
#b = Matrix([0,0,0,pi3]).T #row vector = column vector transposed 
b = Matrix([0,0,0,q3]).T #row vector = column vector transposed 
print('\n b= \n',b)

Mmod_inv = Mmod.inv()
print('\n Mmod_inv= \n',Mmod_inv)
#pimod = Mmod.inv() * b.T #**
#pimod = b.T * Mmod_inv  #**
pimod = b * Mmod_inv  #**
print('\n pimod= \n',simplify(pimod))

#xmod = np.linalg.solve(Mmod, b.transpose())
#print('\n xmod= \n',xmod)

result0 = solve(PiMmod,[pi0,pi1,pi2,pi3]) #FUNCIONO!!!! (con inversa no se porque no funcionó aquí, tal vez por usar números en b en lugar de letras, no se pero con solve si funciona!)
#result = solve( [ pi0 - pi3*q1*q2*q3/(p0*p1*p2), pi1 - pi3*q2*q3/(p1*p2), pi2 - pi3*q3/p2, pi0 + pi1 + pi2 + pi3 -1 ], [pi0,pi1,pi2,pi3] ) 
print('\n result0 = \n', result0) #this solution involves the complementary equation where sum_j of pi_j = 1

#pi3 = 1 - (pi0 + pi1 + pi2)
result = solve( [ -p0*pi0 + q1*pi1, p0*pi0 - (p1-q1)*pi1 + q2*pi2, p1*pi1 - (p1-q1)*pi2 + q3*pi3, pi0+pi1+pi2+pi3], [pi0,pi1,pi2,pi3] ) 
print('\n result = \n', result) # this is the trivial solution (x=0)















####WORKING:

'''
P_minus_1I = P - sp.eye(4)
print('\n det(P_minus_1I):\n', P_minus_1I.det()) # det = 0 to get a non-trivial solution

Pi_by_P_minus_1I = Pi * P_minus_1I

print('\n Pi_by_P_minus_1I : \n', Pi_by_P_minus_1I)

#res = solve([3*x+9*y-10*z-24,x-6*y+4*z+4,10*x-2*y+8*z-20],[x,y,z])
res = solve(Pi_by_P_minus_1I,[pi0,pi1,pi2,pi3])
print('\n res = \n', res) # eigenvalues: this is the solution of (A-lambda*I)*x=0, with lambda=1


res2 = solve( [ pi0 - pi3*q1*q2*q3/(p0*p1*p2), pi1 - pi3*q2*q3/(p1*p2), pi2 - pi3*q3/p2, pi0 + pi1 + pi2 + pi3 -1 ], [pi0,pi1,pi2,pi3] ) 
print('\n res2 = \n', res2) #this solution involves the complementary equation where sum_j of pi_j = 1

pi3 = 1 - (pi0 + pi1 + pi2)
res3 = solve( [ -p0*pi0 + q1*pi1, p0*pi0 - (p1-q1)*pi1, p1*pi1 - (p1-q1)*pi2 + q3*pi3, p2*pi2 - q3*pi3], [pi0,pi1,pi2,pi3] ) 
print('\n res3 = \n', res3) # this is the trivial solution (x=0)


print('\n Testing manually \n:')

sum = (q1*q2*q3/(p0*p1*p2)) + (q2*q3/(p1*p2)) + (q3/p2) + (1)

stationary_vector = (1/sum) * Matrix([q1*q2*q3/(p0*p1*p2), q2*q3/(p1*p2), q3/p2,1])

print('\n stationary_vector \n:', stationary_vector)

print('\n stationary_vector simplified \n:', simplify(stationary_vector))

print("\n stationary_vector components must add to 1 (Probabilities): \n", simplify(stationary_vector[0]+stationary_vector[1]+stationary_vector[2]+stationary_vector[3]))


###########################################################################
print()
print("Now trying with eigenvectors automatically:")

M = Matrix([[4,2],[3,3]])
print("Matrix : {} ".format(M)) 
M_eigenvects = M.eigenvects()  
print("Eigenvects of a matrix : {}".format(M_eigenvects))   

print()

#print("Matrix : {} ".format(P)) 
#P_eigenvects = P.eigenvects()  
#print("Eigenvects of a matrix : {}".format(P_eigenvects))   

'''