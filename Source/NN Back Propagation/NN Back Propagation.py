#################################
# ---- Made By Skult78911 ---- ##  
# ---- All RIght Reserved ---- ##
# ---- Writted In VS Code ---- ## 
# ---- Writted In Python3 ---- ## 
#################################

import numpy as np # Importing Numpy Libary

# -- f Function - To Calculate Hyperbolic Dunbance -- #
def f(x):  
    return 2/(1 + np.exp(-x)) - 1

# -- df Function - To Calculate Dunbans Derivative -- #
def df(x): 
    return 0.5*(1 + x)*(1 - x)

# -- Weight Layers -- #
W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]]) # Weight For 1 Layer
W2 = np.array([0.2, 0.3]) # Weight For 2 Layer

# -- Function To Pass An Observation Vector Through A Neural Network -- #
def go_forward(inp): 
    sum = np.dot(W1, inp) # 1 Sum For Neuron
    out = np.array([f(x) for x in sum]) # out Variable - Output Value For Hidden Layer

    sum = np.dot(W2, out) # 2 Sum For Neuron
    y = f(sum) # y Variavle - Output Value For Entire Neural Network
    return (y, out) # Return These Values As This Carthage

# -- A Function That Trains a Neural Network -- #
def train(epoch):
    global W2, W1
    lmd = 0.01  # Learning Step
    N = 10000  # Number Of Training İterations
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # Random Selection Of The Input Signal From The Training Sample
        y, out = go_forward(x[0:3])  # Direct Pass Through The NN And Calculation Of Output Values Of Neurons
        e = y - x[-1] # Error
        delta = e*df(y) # Local Gradient
        W2[0] = W2[0] - lmd * delta * out[0] # First Link Weight Adjustment
        W2[1] = W2[1] - lmd * delta * out[1] # Adjusting The Weight Of The Second Link
        delta2 = W2*delta*df(out) # Vector Of 2 Local Gradients

        # Adjustment Of Links Of The First Layer
        W1[0, :] = W1[0, :] - np.array(x[0:3]) * delta2[0] * lmd
        W1[1, :] = W1[1, :] - np.array(x[0:3]) * delta2[1] * lmd

# Training Sample (Aka Full Sample)
epoch = [(-1, -1, -1, -1), 
         (-1, -1, 1, 1),
         (-1, 1, -1, -1),
         (-1, 1, 1, 1),
         (1, -1, -1, -1),
         (1, -1, 1, 1),
         (1, 1, -1, -1),
         (1, 1, 1, -1)]

train(epoch)  # Start Learning Step

# -- Verification Of The Results -- #
for x in epoch:
    y, out = go_forward(x[0:3])
    print(f"Output value NN: {y} => {x[-1]}") # Printing Values To Console