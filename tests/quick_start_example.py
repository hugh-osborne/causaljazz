#!/usr/bin/env python
# coding: utf-8

# # **Scenario 3 : Data Generation from a Probability Distribution**
# 
# As before, we train ANNs or define our own functions to calculate each variable in the DAG. However, instead of a Monte Carlo (agent based) approach, we use Causal Jazz to build a discretised probability distribution.

# 
# Import the usual suspects and the pmf module from causaljazz.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import csv

from causaljazz.visualiser import Visualiser
from causaljazz.inference import TEDAG_FUNCTION
from causaljazz.inference import TEDAG
import causaljazz.data as data

from scipy.stats import norm


# # **Helper Functions**

# In[2]:


# Return the approximate discretised probability mass function for a normal distribution with x_mean and x_sd. The discretisation goes from x_min to x_max with res bins.
def generateGaussianNoisePmf(x_min, x_max, x_mean, x_sd, res):
    x_space = np.linspace(x_min, x_max, res)
    x_dist = [a * ((x_max-x_min)/res) for a in norm.pdf(x_space, x_mean, x_sd)]
    x_pmf = [a / sum(x_dist) for a in x_dist]
    return x_pmf


# Load the original data into an ND-array structure.
# The original data is made up of up to 25 sets of 1000 data points. Setting number_of_experiments higher improves the functions.

# In[3]:


csv_names = ['X1', 'X2', 'LC', 'X3', 'SF']

# Load the data into an array
ground = [] # The ground truth array
number_of_experiments = 25
with open('ground.csv') as csvfile:
    ground_reader = csv.DictReader(csvfile)
    for row in ground_reader:
        if int(row['Sim']) > number_of_experiments:
            break
        d = [float(a) for a in [row[k] for k in csv_names]]
        ground += [d]

# Normalise it otherwise training doesn't work!
max_vals = np.max(ground, axis=0)
min_vals = np.min(ground, axis=0)
ground = ((np.array(ground)-min_vals)/(max_vals-min_vals))


# Set grid resolution variables and flags.

# In[4]:


generate_models = False
number_agents = 500
input_res = 10
output_res = 30
total_output_size = 2*output_res


# # **Latent Function C**
# 
# Before learning the ANN functions, let's design our function for C.<br>
# 
# *func_c* takes two inputs, X<sub>1</sub> and X<sub>2</sub>, and returns a distribution across two values, 0 and 1.
# 
# To achieve the expected path coefficients, we first define the expected value of a non-dichotomised (non-binary) C based on X<sub>1</sub> and X<sub>2</sub>.
# <br><br>
# E[C] <- 0.3X<sub>1</sub> + 0.3X<sub>2</sub>
# <br><br>
# Next, we must define the variance of C for each value of X<sub>1</sub> and X<sub>2</sub>. For simplicity we will assume that the variance is normally distributed around the expected value with a standard deviation of 1. The conditional distributions could be dependent on the inputs and considerably more complicated than a normal distribution. Note that any skew or bias in the distribution will affect the resulting covariance.<br><br>
# If the latent variable is not to be processed further (for example dichotomised), the function is simple and can return a normal distribution around the expected value. However, this imparts no new information from the latent variable beyond some additional variance - which may be all that is required (enigmatic variation). In this case, though, we also wish to capture a 30/70 split between high and low risk groups. <br><br>
# To get a distribution across two values, 1 and 0, we need to dichotamise the joint distribution so that 30% falls into the high-risk group (C=1). This step has to be performed on the full joint distribution of X<sub>1</sub>, X<sub>2</sub>, and C so that lower values of X<sub>1</sub> and X<sub>2</sub> are more likely to appear in the low-risk group.
# 
# 

# In[5]:


def func_c_exp(y):
  x1 = np.array(y)[:,0]
  x2 = np.array(y)[:,1]

  out = np.reshape((0.3*x1 + 0.3*x2), (np.array(y).shape[0]))
  return out

def func_c_noise(y):
  x1 = np.array(y)[:,0]
  x2 = np.array(y)[:,1]

  return (np.random.normal(0.0, 0.1, size=np.array(y).shape[0])-min_vals[0])/(max_vals[0] - min_vals[0])

def func_c_sampled(y):
  x1 = y[0]
  x2 = y[1]

  exp_c = 0.3*x1 + 0.3*x2
  cont_c = np.random.normal(loc=exp_c, scale=0.1)
  return cont_c
  #if cont_c > 0.2:
  #  return 1
  #else:
  #  return 0


# Learn the functions for X<sub>2</sub>, X<sub>3</sub>, and S. In this example dataset, X<sub>1</sub> is normally distributed around 0.0 with a standard deviation of 1.0.

# In[6]:


# X2 <- X1
data_points = np.stack([ground[:,0], ground[:,1]])
func_e_x2, func_x2_noise = data.trainANN('x2_given_x1', generate_models, data_points, [input_res], output_res)

# C <- X1,X2
generated_c = np.array([func_c_sampled(x) for x in ground[:,:2]])

# X3 <- X1,X2,C
data_points = np.stack([ground[:,0], ground[:,1], generated_c, ground[:,3]])
func_e_x3, func_x3_noise = data.trainANN('x3_given_x1x2c', generate_models, data_points, [input_res,input_res,input_res], output_res)

# S <- X1,X2,C,X3
data_points = np.stack([ground[:,0], ground[:,1], generated_c, ground[:,3], ground[:,4]])
func_e_s, func_s_noise  = data.trainANN('s_given_x1x2cx3', generate_models, data_points, [input_res,input_res,input_res,input_res], output_res)


# # **Causal Jazz Simulation**

# In[7]:

# Make a function to sum things
# Helper function to sum the inputs (in this case, sum the expected value and the noise)
def func_sum(y):
  out = np.reshape(np.sum(y, axis=-1), (np.array(y).shape[0]))
  return out

# Define the variable names for each function in TEDAG
tedag_func_x2_e = TEDAG_FUNCTION(['X1'], 'X2E', 0, func_e_x2)
tedag_func_x2_noise = TEDAG_FUNCTION(['X1'], 'X2N', 0, func_x2_noise)
tedag_func_x2 = TEDAG_FUNCTION(['X2E', 'X2N'], 'X2', 0, func_sum)
tedag_func_lc_e = TEDAG_FUNCTION(['X1', 'X2'], 'LCE', 0, func_c_exp)
tedag_func_lc_noise = TEDAG_FUNCTION(['X1', 'X2'], 'LCN', 0, func_c_noise)
tedag_func_lc = TEDAG_FUNCTION(['LCE', 'LCN'], 'LC', 0, func_sum)
tedag_func_x3_e = TEDAG_FUNCTION(['X1', 'X2', 'LC'], 'X3E', 0, func_e_x3)
tedag_func_x3_noise = TEDAG_FUNCTION(['X1', 'X2', 'LC'], 'X3N', 0, func_x3_noise)
tedag_func_x3 = TEDAG_FUNCTION(['X3E', 'X3N'], 'X3', 0, func_sum)
tedag_func_s_e  = TEDAG_FUNCTION(['X1', 'X2', 'LC', 'X3'], 'SE', 0, func_e_s)
tedag_func_s_noise  = TEDAG_FUNCTION(['X1', 'X2', 'LC', 'X3'], 'SN', 0, func_s_noise)
tedag_func_s = TEDAG_FUNCTION(['SE', 'SN'], 'S', 0, func_sum)

# Initialise the TEDAG
tedag = TEDAG(1, [tedag_func_x2_e,tedag_func_x2_noise,tedag_func_lc_e,tedag_func_lc_noise,tedag_func_x2,tedag_func_lc,tedag_func_x3_e,tedag_func_x3_noise,tedag_func_x3,tedag_func_s_e,tedag_func_s_noise,tedag_func_s], observables=['X1', 'X2', 'LC', 'X3', 'S'], verbose=True)

# Add a single intervention to set X1
# Set as a normal distribution normalised according to the original X1 data
tedag.addIntervention(['X1'], [(np.random.normal(0.0, 1.0, size=number_agents)-min_vals[0])/(max_vals[0]-min_vals[0])], 0)

# Forward calculate the distributions
while tedag.findNextFunctionAndApply(0):
    continue

# Let's plot the points and compare to the original data.

# In[8]:

space = [0.0, 1.0, 0.0, 1.0]
var_names = ['X1', 'S']

state = tedag.getSubState(var_names, 0)

fig = plt.figure(1, dpi=100)
plt.xlim([space[0],space[1]])
plt.xlabel(var_names[0])
plt.ylabel(var_names[1])
plt.ylim([space[2],space[3]])
plt.scatter(ground[:1000,0],ground[:1000,4],s=1.0,color='#FF00FF')
plt.scatter(state[0,:number_agents],state[1,:number_agents],s=15.0,marker='x')
plt.show(block=True)
#plt.close()

