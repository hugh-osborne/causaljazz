# Causal Jazz
Causal Jazz is a pure Python tool for building partially data-driven models with respect to a directed acyclic graph (DAG). The DAG is a fundamental requirement for causal inference that imbues models with the ability to describe the causal effect of policy changes and simulate counterfactual scenarios. This is in stark contrast to black-box/deep learning models that can only predict an outcome based on a full set of inputs.
Compared to other causal inference tools, Causal Jazz can use data to train simple ANNs that sit between nodes in the DAG and estimate the full distribution of possible values without prior knowledge of the shape of the data. It is also designed to handle time-dependent data in a well-defined and natural way improving analysis of datasets from longitudinal studies that often involve timed and repeated measures. 
___
"What's more important, the data or the jazz? Sure, sure, 'Information should be free' and all that - but anyone can set information free. The jazz is in how you do it, what you do it to, and in almost getting caught without getting caught. The data is 1's and 0's. Life is the jazz."

~ Datatech Sinder Roze on Infobop
___

## Quick Start Guide 
The source code for this guide is available at https://github.com/hugh-osborne/causaljazz/blob/e98059b3cad02dc48e26e4f3ba086a1889d10050/tests/quick_start_example.ipynb.
 
### Installation
Causal Jazz can be installed by cloning this repository and calling `pip install .` from the root directory.
### Learning Existing Data Distributions
The first task is to set up Causal Jazz with a dataset and associated DAG. `tests/ground.csv` holds a set of simulated data points for five variables, X1, X2, X3, LC, and S.
The variables are causally related according to the DAG below.
![The DAG](https://github.com/hugh-osborne/causaljazz/blob/e98059b3cad02dc48e26e4f3ba086a1889d10050/docs/quick_start_dag.png)

First import the relevant classes and modules from Causal Jazz.

    from causaljazz.cpu import pmf
    from causaljazz.cpu import CausalFunction
    from causaljazz.inference import TEDAG_FUNCTION
    from causaljazz.inference import TEDAG
    import causaljazz.data as data
Set up a helper function to generate normal/Gaussian probability mass function discretised between two values.

    def generateGaussianNoisePmf(x_min, x_max, x_mean, x_sd, res):
        x_space = np.linspace(x_min, x_max, res)
        x_dist = [a * ((x_max-x_min)/res) for a in norm.pdf(x_space, x_mean, x_sd)]
        x_pmf = [a / sum(x_dist) for a in x_dist]
        return x_pmf

Now load the data from `ground.csv`. The data are split into "experiments" made up of 1000 rows each. More experiments/rows means better but slower training. Of course, values are normalised to between 0 and 1.

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
Now set some global variables:

 - **generate_models** indicates that the ANNs should be generated and saved to file. When this is False, the expectation is that the files have already been generated and can be retrieved.
 - **input_res** - when training the ANNs, the input joint distribution of the function is discretised into this many bins for the input variables.
 - **output_res** - the joint distribution is discretised into this many bins for the output variable.
 - **output_buffer** - gives an additional width to the output variable so that estimates can be made beyond the distribution of the dataset.

generate_models = True
input_res = 10
output_res = 30
output_buffer = 5
total_output_size = 2*(output_res+output_buffer)

To calculate the variable LC, instead of an ANN, the function is user-defined. This demonstrates the ability of Causal Jazz to mix both data-driven and non-data-driven processes during simulation. `func_c_exp()` takes a set of points each with two values (X1 and X2) that are used to estimate the expected value of C. `func_c_noise()` takes the same input but returns a discretised Gaussian pmf to represent variance in C from the expected value. Finally, `func_c_sampled()` takes a set of X1s and X2s and returns a set of C samples.

    def func_c_exp(y):
      x1 = np.array(y)[:,0]
      x2 = np.array(y)[:,1]
    
      out = np.reshape((0.3*x1 + 0.3*x2), (np.array(y).shape[0],1))
      return out
    
    def func_c_noise(y):
      x1 = np.array(y)[:,0]
      x2 = np.array(y)[:,1]
    
      out = np.array([generateGaussianNoisePmf(-0.5, 0.5, 0, 0.1, total_output_size) for a in range(np.array(y).shape[0])]).T
      return out
    
    def func_c_sampled(y):
      x1 = y[0]
      x2 = y[1]
    
      exp_c = 0.3*x1 + 0.3*x2
      cont_c = np.random.normal(loc=exp_c, scale=0.1)
      return cont_c
Next, the data module is used to call `TrainANN()` on the remaining variables in the DAG. The output from func_c_sampled() is used to produce new training data values for C.

    # X2 <- X1
    data_points = np.stack([ground[:,0], ground[:,1]])
    func_e_x2, func_x2_noise = data.trainANN('x2_given_x1', generate_models, data_points, [input_res], output_res, output_buffer)
    
    # C <- X1,X2
    generated_c = np.array([func_c_sampled(x) for x in ground[:,:2]])
    
    # X3 <- X1,X2,C
    data_points = np.stack([ground[:,0], ground[:,1], generated_c, ground[:,3]])
    func_e_x3, func_x3_noise = data.trainANN('x3_given_x1x2c', generate_models, data_points, [input_res,input_res,input_res], output_res, output_buffer)
    
    # S <- X1,X2,C,X3
    data_points = np.stack([ground[:,0], ground[:,1], generated_c, ground[:,3], ground[:,4]])
    func_e_s, func_s_noise  = data.trainANN('s_given_x1x2cx3', generate_models, data_points, [input_res,input_res,input_res,input_res], output_res, output_buffer)

## Building the Time-Explicit DAG (TEDAG)
Now all functions have been generated, the DAG from above must be defined in Causal Jazz. Each function is associated with a TEDAG_FUNCTION that names the input and output variables so that they can be matched across the full DAG. Following the motif of a separate function for the expected value and one for the variance, two additional variables are defined for each node in the DAG appended with either 'E' (for expected) or 'N' (for noise). The actual function for the node is the summation of these two associated variables.

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
    tedag_func_x2_e = TEDAG_FUNCTION(['X1'], 'X2E', 0, trans_x2_e, x2_template)
    tedag_func_x2_noise = TEDAG_FUNCTION(['X1'], 'X2N', 0, trans_x2_noise, x2_template)
    tedag_func_x2 = TEDAG_FUNCTION(['X2E', 'X2N'], 'X2', 0, trans_sum, x2_template)
    tedag_func_lc_e = TEDAG_FUNCTION(['X1', 'X2'], 'LCE', 0, trans_lc_e, lc_template)
    tedag_func_lc_noise = TEDAG_FUNCTION(['X1', 'X2'], 'LCN', 0, trans_lc_noise, lc_template)
    tedag_func_lc = TEDAG_FUNCTION(['LCE', 'LCN'], 'LC', 0, trans_sum, lc_template)
    tedag_func_x3_e = TEDAG_FUNCTION(['X1', 'X2', 'LC'], 'X3E', 0, trans_x3_e, x3_template)
    tedag_func_x3_noise = TEDAG_FUNCTION(['X1', 'X2', 'LC'], 'X3N', 0, trans_x3_noise, x3_template)
    tedag_func_x3 = TEDAG_FUNCTION(['X3E', 'X3N'], 'X3', 0, trans_sum, x3_template)
    tedag_func_s_e  = TEDAG_FUNCTION(['X1', 'X2', 'LC', 'X3'], 'SE', 0, trans_s_e, s_template)
    tedag_func_s_noise  = TEDAG_FUNCTION(['X1', 'X2', 'LC', 'X3'], 'SN', 0, trans_s_noise, s_template)
    tedag_func_s = TEDAG_FUNCTION(['SE', 'SN'], 'S', 0, trans_sum, s_template)
    
    # Initialise the TEDAG
    tedag = TEDAG(1, [tedag_func_x2_e,tedag_func_x2_noise,tedag_func_lc_e,tedag_func_lc_noise,tedag_func_x2,tedag_func_lc,tedag_func_x3_e,tedag_func_x3_noise,tedag_func_x3,tedag_func_s_e,tedag_func_s_noise,tedag_func_s], observables=['X1', 'X2', 'LC', 'X3', 'S'], verbose=True)
An initial distribution must be defined for X1 because it has no parent node. To do this, we defined an "intervention" at time=0.

    # Add a single intervention to set X1
    x1 = generateGaussianNoisePmf(-3.0,3.0,0.0,1.0, output_res)
    x1 /= max_vals[0]
    x1_pmf = pmf(np.array(x1), np.array([0.0]), np.array([1.0/output_res]), 0.000001)
    tedag.addIntervention(['X1'], 0, x1_pmf)

Finally, to run the simulation, we make repeated calls to `findNextFunctionAndApply()` until all nodes have been calculated. For simulations that change in time (see other examples), this process is applied for each time step.

    # Forward calculate the distributions
    while tedag.findNextFunctionAndApply(0):
        continue
The result of the simulation is a discretised joint distribution of all variables in the DAG. Marginal distributions can be plotted.

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
    plt.close()
![Marginal distribution of X1 and S](https://github.com/hugh-osborne/causaljazz/blob/e98059b3cad02dc48e26e4f3ba086a1889d10050/docs/quick_start_output.png)