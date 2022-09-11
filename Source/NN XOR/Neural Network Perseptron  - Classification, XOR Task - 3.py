#################################
# ---- Made By Skult78911 ---- ##  
# ---- All RIght Reserved ---- ##
# ---- Writted In VS Code ---- ## 
# ---- Writted In Python3 ---- ## 
#################################

import numpy as np # Importing Numpy Libary

# -- Activation Function -- #
def act(x):
    return 0 if x <= 0 else 1 # Returns 0 if x is equal to or less than zero and one if x is greater than zero

# -- go Function -- #
def go(C): # go Function - A Function That Launches a NN (Neural Network)
           # In a neural network, we transmit Class C

    x = np.array([C[0], C[1], 1]) # Class C Will Have A Value Either 0 Or 1
                                  # By These Values We Will Determine Which Class Our Input Signal Belongs To Either Class C1 Or Class C2
                                  # Last Unit This Is 3 Entry
    w1 = [1, 1, -1.5] # Forming Weight For 1 Neuron
    w2 = [1, 1, -0.5] # Forming Weight For 2 Neuron
    w_hidden = np.array([w1, w2]) # Combine All Weights into a Matrix
    w_out = np.array([-1, 1, -0.5]) # Defining Weights for the Output Neuron

    sum = np.dot(w_hidden, x) # We calculate the sum on each 
    out = [act(x) for x in sum] # Passing the Sum Through the Activation Function
    out.append(1) # Adding a Unit to the out Variable
    out = np.array(out) # We turn everything received into a vector

    sum = np.dot(w_out, out) # The resulting vector is multiplied by the weight coefficients for the output neuron and we get the sum
    y = act(sum) # We pass the received amount through the activation function and get the output value of NN (Neural Network) 
    return y # Returning y Using the go Function

C1 = [(1,0), (0,1)] # C1 Class 
C2 = [(0,0), (1,1)] # C2 Class

print( go(C1[0]), go(C1[1]) ) # We Pass Class C1 Through NN (Neural Network) And Display The Result In The Console
print( go(C2[0]), go(C2[1]) ) # We Pass Class C2 Through NN (Neural Network) And Display The Result In The Console

# --- If We Did Everything Correctly Then The Console Should Display 1 1 And 0 0 --- #