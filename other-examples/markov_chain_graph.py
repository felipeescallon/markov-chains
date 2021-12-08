from __future__ import division  # Only for how I'm writing the transition matrix
import networkx as nx  # For the magic
import matplotlib.pyplot as plt  # For plotting


import sympy as sp
from sympy import *
from sympy import Matrix
from sympy import Symbol


# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 09:57:44 2021

@author: Administrador (Andrés Felipe Escallón Portilla)
"""

'''
REFERENCES:
    
    https://vknight.org/unpeudemath/code/2015/11/15/Visualising-markov-chains.html
    
'''


states = [(0, 0),
          (1, 0),
          (2, 0),
          (3, 0)]

#(p0,p1,p2,q1,q2,q3) = symbols("p0,p1,p2,q1,q2,q3")
#transition matrix Q:
'''    
Q=Matrix([[1-p0,      p0,        0,     0],
          [q1,   1-p1-q1,       p1,     0],
          [0,         q2,  1-p2-q2,    p2],
          [0,          0,       q3,  1-q3]]) 
'''
Q=[["1-p0",      "p0",        "0",     "0"],
  ["q1",   "1-p1-q1",       "p1",     "0"],
  ["0",         "q2",  "1-p2-q2",    "p2"],
  ["0",          "0",       "q3",  "1-q3"]]
'''
   Node    N0       N1       N2    N3
0   N0  1-p0       p0        0     0
1   N1    q1  1-p1-q1       p1     0
2   N2     0       q2  1-p2-q2    p2
3   N3     0        0       q3  1-q3

'''

'''
To build the networkx graph we will use our states as nodes and have edges labeled by the corresponding values of Q (ignoring edges that would correspond to a value of 0). The neat thing about networkx is that it allows you to have any Python instance as a node:
'''

G = nx.MultiDiGraph()
labels={}
edge_labels={}

for i, origin_state in enumerate(states):
    for j, destination_state in enumerate(states):
        rate = Q[i][j]
        #if rate > "0":
        if rate != "0":
            G.add_edge(origin_state,
                       destination_state,
                       weight=rate,
                       #label="{:.02f}".format(rate))
                       label=rate)
            edge_labels[(origin_state, destination_state)] = label=rate#"{:.02f}".format(rate)
            

#Now we can draw the chain:
plt.figure(figsize=(14,7))
node_size = 200
pos = {state:list(state) for state in states}
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
nx.draw_networkx_labels(G, pos, font_weight=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.axis('off');    
#plt.savefig("../images/mc-matplotlib.svg", bbox_inches='tight')
plt.savefig("./images/markov_chain.png", bbox_inches='tight')
#plt.savefig("markov_chain.png", bbox_inches='tight')
#nx.write_dot(G, 'mc.dot')