#REFERENCE: https://github.com/NaysanSaran/markov-chain

import numpy as np
import matplotlib.pyplot as plt

# modules from this repository
from markovchain import MarkovChain

def main():
    
    #--------------------------------------------------------------------------
    # 2-state Markov chain
    #--------------------------------------------------------------------------
    P = np.array([[0.8, 0.2], [0.1, 0.9]]) # Transition matrix
    mc = MarkovChain(P, ['1', '2'])
    mc.draw("../img/markov-chain-two-states.png")
    
    #--------------------------------------------------------------------------
    # 3-state Markov chain
    #--------------------------------------------------------------------------
    P = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.7, 0.2],
    ])
    mc = MarkovChain(P, ['A', 'B', 'C'])
    mc.draw("../img/markov-chain-three-states.png")
 
    #--------------------------------------------------------------------------
    # 4-state Markov chain
    #--------------------------------------------------------------------------
    P = np.array([
        [0.8, 0.1, 0.1, 0.0], 
        [0.1, 0.7, 0.0, 0.2],
        [0.1, 0.0, 0.7, 0.2],
        [0.1, 0.0, 0.7, 0.2]
    ])
    mc = MarkovChain(P, ['1', '2', '3', '4'])
    mc.draw("../img/markov-chain-four-states.png")
    
    #--------------------------------------------------------------------------
    # 4-state Markov chain (my_example: symbolically)
    #--------------------------------------------------------------------------
    '''
    P = np.array([
        [0.8, 0.1, 0.1, 0.0], 
        [0.1, 0.7, 0.0, 0.2],
        [0.1, 0.0, 0.7, 0.2],
        [0.1, 0.0, 0.7, 0.2]
    ])
    '''    
    P = np.array([
       ["1-p0",      "p0",        "0",     "0"],
       ["q1",   "1-p1-q1",       "p1",     "0"],
       ["0",         "q2",  "1-p2-q2",    "p2"],
       ["0",          "0",       "q3",  "1-q3"]
    ])    
    mc = MarkovChain(P, ['0', '1', '2', '3'])
    mc.draw("../img/markov-chain-four-states-my_example.png")
 

if __name__ == "__main__":
    main()

