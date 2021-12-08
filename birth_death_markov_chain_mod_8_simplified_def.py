# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 07:57:00 2021

@author: Administrador (Andrés Felipe Escallón Portilla)
"""
##############REFERENCE################################

#http://blog.espol.edu.ec/estg1003/tag/cadenas-markov/

#######################################################
import sympy as sp
from sympy import *
from sympy import Matrix
from sympy import Symbol
############################################################################################
print("birth_death_markov_chain:")

#working symbollically with sympy
(p0,p1,p2,q1,q2,q3) = symbols("p0,p1,p2,q1,q2,q3")
(pi0,pi1,pi2,pi3) = symbols("pi0,pi1,pi2,pi3")

#transition matrix
P = Matrix( [ [1-p0, p0, 0, 0],  [q1, 1-p1-q1, p1, 0], [0, q2, 1-p2-q2, p2], [0, 0, q3, 1-q3]  ] )
print('\n Qual a matriz de transição P? \n')
print('\n Transition matrix: \n')
print('\n P:\n', P)

#pi (row vector)
Pi = Matrix( [pi0,pi1,pi2,pi3] ).T #row vector = column vector transposed 
print('\nPi:\n',Pi)

eigenvalue = 1

############################################################################################
print("\n Another way to do it more automatically:\n")

tam = int(len(P)**0.5) #number of rows/columns

#arranging and solving the linear system to get the stationary propabilities
Pt_minus_1I = P.T - eigenvalue*sp.eye(tam)
print('\n Pt_minus_1I:\n', Pt_minus_1I) 

Pt_minus_1I[-1,:] = sp.ones(tam)[-1,:] #replacing the last row with ones (probability constraint)

Pt_minus_1I_replaced = Pt_minus_1I

b = sp.zeros(tam)[-1,:] #independent row vector (zeros)
b[-1] = 1  #replacing the last row with one (probability constraint)
b_replaced = b

Pt_minus_1I_replaced_by_Pit = Pt_minus_1I_replaced * Pi.T
print('\n Pt_minus_1I_replaced_by_Pit : \n', Pt_minus_1I_replaced_by_Pit)

Pn_calc = solve(Pt_minus_1I_replaced_by_Pit - b_replaced.T , Pi.T) #solving the resulting linear system (passing b_replaced to the left and leaving the right part as 0)

print('\n Quais as probabilidades estacionárias?:\n') 
print('\n Stationary probabilities: \n')
print(' Pn_calc =', pretty(Pn_calc)) #pretty used to show it more beautifully

     
print('\n Eu sei que eu saí do estado 1 no instante n, qual a probabilidade de eu ter ido para estado 2?: \n')
resposta="P(X_{n+1} = 2 | X_{n+1} ≠ 1 , X_{n} = 1 )} = P(X_{n+1} = 2 , X_{n+1}  ≠ 1 | X_{n} = 1 ) / P(X_{n+1} ≠ 1 , X_{n} = 1 ) = P(X_{n+1} = 2 | X_{n} = 1 ) / P(X_{n+1} ≠ 1 , X_{n} = 1 ) = p_{12} / 1-p_{11} = p_1 / 1-[1-(q_{1}+p_{1})] = p_{1} / 1-1+(q_{1}+p_{1}) = p_{1} / q_{1}+p_{1}"
print(pretty(resposta))