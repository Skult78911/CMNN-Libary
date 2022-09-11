#################################
# ---- Made By Skult78911 ---- ##  
# ---- All RIght Reserved ---- ##
# ---- Writted In VS Code ---- ## 
# ---- Writted In Python3 ---- ## 
#################################

import numpy as np # Importing Numpy Libary

def act(x): # Activation Function
	return 0 if x < 0.5 else 1 # Return Zero If x Is 0.5 And One In All Other Cases

# -- Go Function - Passes Through the Neural Network Input Signals (This is house, rock And attr) -- #
def go(house, rock, attr):
	x = np.array([house, rock, attr]) # Variable x - Forming a Vector Based on These Three Parameters(This is house, rock And attr)
	w11 = [0.3, 0.3, 0] # Weight For First Neuron
	w12 = [0.4, -0.5, 1] # Weight For Second Neuron
	weight1 = np.array([w11, w12]) # Combine Weights In 2x3 Matrix
	weight2 = np.array([-1, 1]) # Forming a Connection Vector for the Output Neuron With 1x2 Vector

	sum_hidden = np.dot(weight1, x) # Calculate The Sum Of The Inputs Of Neurons In The Hidden Layer
	print("The value of the sums on the neurons of the hidden layer: "+str(sum_hidden)) # Displaying Computed Hidden Layer Neuron

	out_hidden = np.array([act(x) for x in sum_hidden]) # Using the sum_hidden Vector We Pass It Through the Activation Function And We Get Accordingly From Each Neuron Of The Hidden Layer
	print("The value at the outputs of neurons of the hidden layer: "+str(out_hidden)) # Displaying Computed Out Hidden Layer Neuron

	sum_end = np.dot(weight2, out_hidden) # Calculate The Sum On The Output Neuron According To The Last Layer
										  # We Take The Weights (That's weight2 Variable With Values -1 And 1) And Multiply It By The Sum Of The Hidden Output Value

	y = act(sum_end) # Passing the Sum Through the Activation Function
	                 # What We Get At The Output This Is And The Output Of Our Neural Network
	print("Output Value LV: "+str(y)) # 

	return y # Return Go Function

# You Need To Know That! 'If 1 - Yes, If 0 - No' :)

house = 1 # house Variable 
rock = 0 # rock Variable 
attr = 1 # rock Variable 

res = go(house, rock, attr) # Passing These Variables (This Is A house, rock and attr Variables) Through The Neural Network
if res == 1: # If At The Output Sympathy Is Formed That Is 1 - Yes Then
	print("I like you!") # Print "I Like You!" :)
else: # Else It's 0 - No
	print("Let's Call") # Print "Let's Call" :(