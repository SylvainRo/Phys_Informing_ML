import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

import tensorflow as tf
from tensorflow.keras import layers, models

# True function
start = 0.0
stop = 4.0
N = 2000
t = np.linspace(start,stop,N)
y = np.exp(t)
delta = t[1] - t[0]

# Fake data
y_data = np.exp(t) + np.random.normal(0, 0.9, N)

# Define a simple neural network architecture
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(t, y_data, epochs=10);

# Use the trained model
y_pred = model.predict(t)
y_pred = np.array(y_pred).reshape((len(t),))

# First derivative
def der(x):
  der1 = np.zeros(len(x))
  der1[0] = (x[1] - x[0])/delta
  der1[-1] = (x[-1] - x[-2])/delta
  for i in range(len(x)-2):
    der1[i+1] = (x[i+2] - x[i]) / (2 * delta)
  return der1

# From the plots, der() seems to behave weirdly... It should have far less constant values in my opinion... 
# Since I use exp... Unless the exp function seems very linear to the der() operator???
# OR ! RELU being partially linear, the differentialtion yields piecewise constant derivatives!!!! Try a more smooth activation function like tanh!

# First derivative of the data at given datapoints
y_1 = der(y_pred)


# Plots
plt.figure(figsize=(10, 6))
plt.plot(t,y_data,label='Data',color='blue')
plt.plot(t,y_pred,label='Predicted Solution',color='red')
plt.plot(t,y_1,label='Derivative Predicted Solution',color='y',linestyle='--')
#plt.scatter(t,y_data,label='Data',color='green')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.legend()
plt.title('True Solution vs Predicted Solution')
plt.show()


# Differential Equation Coefficient Prediction (Using prediction and its derivative)
Null = np.zeros((len(t),2))


for i in range(len(t)):
  Null[i] = null_space([[y_pred[i],y_1[i]]]).reshape(2,)


print(Null[1:-1,0].mean() + Null[1:-1,1].mean())
# Ideally we obtain 0.0

coef = np.array([Null[1:-1,0].mean(),Null[1:-1,1].mean()]) * (1/Null[1:-1,1].mean())
# Real (analytic) answer is the vector [-1,1] since the differential equation satisfied is
# -1 * f(y) + 1 * d/dy f(y) = 0  

print(coef)

# Make ODE Solver with the coef above and then compare to exact solution
# Doesn't work, but something like that
from scipy.integrate import odeint

def ode_first(y, t):
    dydt = -coef[0]*y
    return dydt


y0 = y[0] 

#y_data[0] 
# what if the first datapoint is an offlier??? maybe I need a better method... e.g. use y[0]?
# yes... just replaced y_data[0] by y[0]: there is the issue of the dependence on initial values... but since I have many data points, I have many initial
# values at different times, so I might still be able to average different solutions of different init val and get something correct.
# ISSUE ODE Int: takes initial value as [0] and evolves from then, hence evolve both forward and backward in time at different data points and average solutions. 
# This should solve the dependence on init cond. (Should do a statistical analysis of the expecation values) 

sol = odeint(ode_first, y0, t)

plt.figure(figsize=(10, 6))
plt.plot(t, y_data, 'g', label='Data')
plt.plot(t, sol[:, 0], 'b', label='ODE Pred')
plt.legend(loc='best')
plt.xlabel('Time (t)')
plt.legend()
plt.show()
