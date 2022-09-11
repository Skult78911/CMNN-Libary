#################################
# ---- Made By Skult78911 ---- ##  
# ---- All RIght Reserved ---- ##
# ---- Writted In VS Code ---- ## 
# ---- Writted In Python3 ---- ## 
#################################

import numpy as np # Importing Numpy Libary
import matplotlib.pyplot as plt # # Importing Matplotlib Libary

N = 5 # Number of Looks for 1st and 2nd Classes
b = 3 # Offset Variable b

# -- Modeling 1 Image -- #
x1 = np.random.random(N) # Modeling a Random Value Along One Axis x1
x2 = x1 + [np.random.randint(10)/10 for i in range(N)] + b # x2 Modeled as x1 And Plus Random Deviation And at the end we add the variable b
C1 = [x1, x2] # Forming a Double List C1 From the Set of These Points (These Points x1 And x2)

# -- Modeling 1 Image -- #
x1 = np.random.random(N) # Modeling a Random Value Along One Axis x1
x2 = x1 - [np.random.randint(10)/10 for i in range(N)] - 0.1 + b # x2 Modeled as x1 And Minus Random Deviation And at the end we add the variable b
                                                            # And Additionally We Make Minus 0.1 So That This Point x2 Is Below Our Line

C2 = [x1, x2] # Forming a Double List C2 From the Set of These Points (These Points x1 And x2)

f = [0+b, 1+b] # We form a straight line under 45 degrees to see how the dividing line goes But Also Adding b Variable

# -- Determine 2 Coefficients -- #
w2 = 0.5 # Let Omega 2 Coefficient Be 0.5
w3 = -b*w2 # After that We Automatically Calculate Omega 3 Equals Minus b to Omega 2

w = np.array([-w2, w2, w3]) # Well, All Weight Coefficients Will Be What's Inside (It's -w2, w2 and w3 )
for i in range(N):
    x = np.array([C1[0][i], C1[1][i], 1]) # Passing All Images Through Class C1
    y = np.dot(w, x) # Calculate This Output Value y
    if y >= 0: # We look If the value of y is greater than or equal to zero, then this is class C1
        print("Класс C1") # And output it to the console
    else: # Otherwise It's Class C2 
        print("Класс C2") # And output it to the console

plt.scatter(C1[0][:], C1[1][:], s=10, c='red') # Display All Points C1
plt.scatter(C2[0][:], C2[1][:], s=10, c='blue') # Display All Points C2
plt.plot(f) # Splitting the Program With the plot Function
plt.grid(True) # Gridding The Box Where All Neurons
plt.show() # Showing Window With All Neurons (Like Windows Application)