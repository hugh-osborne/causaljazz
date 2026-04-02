import numpy as np
import matplotlib.pyplot as plt
import csv

from causaljazz.visualiser import Visualiser
from causaljazz.cpu import pmf
from causaljazz.cpu import CausalFunction
from causaljazz.inference import TEDAG_PD
import causaljazz.data as data

from scipy.stats import norm

import tensorflow as tf
from tensorflow.keras import layers

# Return the approximate discretised probability mass function for a normal distribution with x_mean and x_sd. The discretisation goes from x_min to x_max with res bins.
def generateGaussianNoisePmf(x_min, x_max, x_mean, x_sd, res):
    x_space = np.linspace(x_min, x_max, res)
    x_dist = [a * ((x_max-x_min)/res) for a in norm.pdf(x_space, x_mean, x_sd)]
    x_pmf = [a / sum(x_dist) for a in x_dist]
    return x_pmf


"""Load the original data into an ND-array structure.
The original data is made up of up to 50 sets of 1000 data points. Setting number_of_experiments higher improves the functions.
"""

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

"""Set grid resolution variables and flags."""

generate_models = True
input_res = 10
output_res = 30
output_buffer = 5
total_output_size = 2*(output_res+output_buffer)

"""# **Latent Function C**

Before learning the ANN functions, let's design our function for C.<br>

*func_c* takes two inputs, X<sub>1</sub> and X<sub>2</sub>, and returns a distribution across two values, 0 and 1.

To achieve the expected path coefficients, we first define the expected value of a non-dichotomised (non-binary) C based on X<sub>1</sub> and X<sub>2</sub>.
<br><br>
E[C] <- 0.3X<sub>1</sub> + 0.3X<sub>2</sub>
<br><br>
Next, we must define the variance of C for each value of X<sub>1</sub> and X<sub>2</sub>. For simplicity we will assume that the variance is normally distributed around the expected value with a standard deviation of 1. The conditional distributions could be dependent on the inputs and considerably more complicated than a normal distribution. Note that any skew or bias in the distribution will affect the resulting covariance.<br><br>
If the latent variable is not to be processed further (for example dichotomised), the function is simple and can return a normal distribution around the expected value. However, this imparts no new information from the latent variable beyond some additional variance - which may be all that is required (enigmatic variation). In this case, though, we also wish to capture a 30/70 split between high and low risk groups. <br><br>
To get a distribution across two values, 1 and 0, we need to dichotamise the joint distribution so that 30% falls into the high-risk group (C=1). This step has to be performed on the full joint distribution of X<sub>1</sub>, X<sub>2</sub>, and C so that lower values of X<sub>1</sub> and X<sub>2</sub> are more likely to appear in the low-risk group.


"""

def func_c_exp(y):
  x1 = np.array(y)[:,0]
  x2 = np.array(y)[:,1]

  out = np.reshape((0.3*x1 + 0.3*x2), (np.array(y).shape[0],1))
  print(out.shape)
  return out

def func_c_noise(y):
  x1 = np.array(y)[:,0]
  x2 = np.array(y)[:,1]

  out = np.array([generateGaussianNoisePmf(-0.5, 0.5, 0, 0.1, total_output_size) for a in range(np.array(y).shape[0])]).T
  print(out.shape)
  return out

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

"""Learn the functions for X<sub>2</sub>, X<sub>3</sub>, and S. In this example dataset, X<sub>1</sub> is normally distributed around 0.0 with a standard deviation of 1.0."""

# X2 <- X1
data_points = np.stack([ground[:,0], ground[:,1]])
func_e_x2, func_x2_noise = data.trainANNForPD('x2_given_x1', generate_models, data_points, [input_res], output_res, output_buffer)

# C <- X1,X2
generated_c = np.array([func_c_sampled(x) for x in ground[:,:2]])
print(np.sum(generated_c), generated_c.shape)

# X3 <- X1,X2,C
data_points = np.stack([ground[:,0], ground[:,1], generated_c, ground[:,3]])
func_e_x3, func_x3_noise = data.trainANNForPD('x3_given_x1x2c', generate_models, data_points, [input_res,input_res,input_res], output_res, output_buffer)

# S <- X1,X2,C,X3
data_points = np.stack([ground[:,0], ground[:,1], generated_c, ground[:,3], ground[:,4]])
func_e_s, func_s_noise  = data.trainANNForPD('s_given_x1x2cx3', generate_models, data_points, [input_res,input_res,input_res,input_res], output_res, output_buffer)

"""# **Causal Jazz Simulation**"""

# Define the Causal Jazz transition function
# It takes the name of the python function, the number of inputs, and a flag to say this is a function
# that returns a discretised distribution (as opposed to a single value)
trans_x2_e = CausalFunction(func_e_x2, 1)
trans_x2_noise = CausalFunction(func_x2_noise, 1, transition_function=True)
trans_lc_e = CausalFunction(func_c_exp, 2)
trans_lc_noise = CausalFunction(func_c_noise, 2, transition_function=True)
trans_x3_e = CausalFunction(func_e_x3, 3)
trans_x3_noise = CausalFunction(func_x3_noise, 3, transition_function=True)
trans_s_e  = CausalFunction(func_e_s, 4)
trans_s_noise  = CausalFunction(func_s_noise, 4, transition_function=True)

# Make a function to sum things
# Helper function to sum the inputs (in this case, sum the expected value and the noise)
def func_sum(y):
  out = np.reshape(np.sum(y, axis=-1), (np.array(y).shape[0],1))
  print(out.shape)
  return out

trans_sum = CausalFunction(func_sum, 2)

# Template distribution space for each variable
# The templates must have the same cell widths as the pmfs used to train the functions
x2_template = pmf(np.array([]), np.array([0.0]), np.array([1.0 / output_res]), 0.000001) # The minimum mass should obviously be 0
lc_template = pmf(np.array([]), np.array([0.0]), np.array([1.0 / output_res]), 0.000001)
x3_template = pmf(np.array([]), np.array([0.0]), np.array([1.0 / output_res]), 0.000001)
s_template  = pmf(np.array([]), np.array([0.0]), np.array([1.0 / output_res]), 0.000001)

# Define the variable names for each function in TEDAG
tedag_func_x2_e = TEDAG_PD.FUNCTION(['X1'], 'X2E', 0, trans_x2_e, x2_template)
tedag_func_x2_noise = TEDAG_PD.FUNCTION(['X1'], 'X2N', 0, trans_x2_noise, x2_template)
tedag_func_x2 = TEDAG_PD.FUNCTION(['X2E', 'X2N'], 'X2', 0, trans_sum, x2_template)
tedag_func_lc_e = TEDAG_PD.FUNCTION(['X1', 'X2'], 'LCE', 0, trans_lc_e, lc_template)
tedag_func_lc_noise = TEDAG_PD.FUNCTION(['X1', 'X2'], 'LCN', 0, trans_lc_noise, lc_template)
tedag_func_lc = TEDAG_PD.FUNCTION(['LCE', 'LCN'], 'LC', 0, trans_sum, lc_template)
tedag_func_x3_e = TEDAG_PD.FUNCTION(['X1', 'X2', 'LC'], 'X3E', 0, trans_x3_e, x3_template)
tedag_func_x3_noise = TEDAG_PD.FUNCTION(['X1', 'X2', 'LC'], 'X3N', 0, trans_x3_noise, x3_template)
tedag_func_x3 = TEDAG_PD.FUNCTION(['X3E', 'X3N'], 'X3', 0, trans_sum, x3_template)
tedag_func_s_e  = TEDAG_PD.FUNCTION(['X1', 'X2', 'LC', 'X3'], 'SE', 0, trans_s_e, s_template)
tedag_func_s_noise  = TEDAG_PD.FUNCTION(['X1', 'X2', 'LC', 'X3'], 'SN', 0, trans_s_noise, s_template)
tedag_func_s = TEDAG_PD.FUNCTION(['SE', 'SN'], 'S', 0, trans_sum, s_template)

# Initialise the TEDAG
tedag = TEDAG_PD(1, [tedag_func_x2_e,tedag_func_x2_noise,tedag_func_lc_e,tedag_func_lc_noise,tedag_func_x2,tedag_func_lc,tedag_func_x3_e,tedag_func_x3_noise,tedag_func_x3,tedag_func_s_e,tedag_func_s_noise,tedag_func_s], observables=['X1', 'X2', 'LC', 'X3', 'S'], verbose=True)

# Add a single intervention to set X1
x1 = generateGaussianNoisePmf(-3.0,3.0,0.0,1.0, output_res)
x1 /= max_vals[0]
x1_pmf = pmf(np.array(x1), np.array([0.0]), np.array([1.0/output_res]), 0.000001)
tedag.addIntervention(['X1'], 0, x1_pmf)

# Forward calculate the distributions
while tedag.findNextFunctionAndApply(0):
    continue

"""Let's plot the points and compare to the original data."""

space = [0.0, 1.0, 0.0, 1.0]
var_names = ['X1', 'S']

fig = plt.figure(1, dpi=100)

tedag_pmf = tedag.getPmfForIteration(var_names, 0)
if tedag_pmf is not None:
    node_indices = [[n.key for n in tedag_pmf.nodes].index(a+str(0)) for a in var_names]
    coords, centroids, vals = tedag_pmf.pmf.calcMarginal(node_indices)

    vals = np.array(vals)
    coords = np.array(coords)
    grid = np.zeros((output_res,output_res))
    for c in range(len(vals)):
        ws = [tedag_pmf.pmf.cell_widths[node_indices[0]],tedag_pmf.pmf.cell_widths[node_indices[1]]]
        os = [tedag_pmf.pmf.origin[node_indices[0]], tedag_pmf.pmf.origin[node_indices[1]]]
        cs = [coords[c][0]+(int((os[0]-space[0])/ws[0])), coords[c][1]+(int((os[1]-space[2])/ws[1]))]
        cs = [c if c < output_res else output_res-1 for c in cs]
        cs = [c if c >= 0 else 0 for c in cs]
        grid[int(cs[0]),int(cs[1])] = vals[c]
    grid = np.transpose(grid)

    plt.xlim([space[0],space[1]])
    plt.xlabel(var_names[0])
    plt.ylabel(var_names[1])
    plt.ylim([space[2],space[3]])
    plt.imshow(grid, cmap='hot', origin='lower', extent=(space[0],space[1],space[2],space[3]), aspect='auto')

    plt.scatter(ground[:1000,0],ground[:1000,4],s=1.0,color='#FF00FF')
    #plt.scatter(ground[:1000,0],ground[:1000,1],s=0.2)
plt.show(block=False)