from .cpu import pmf
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

# Given a list of values representing a discretised pmf, return the cumulative list
def build_cmf(pmf):
    num_intervals = len(pmf)
    # build cmfs for sampling
    cmf = np.zeros(num_intervals)
    mass = 0.0
    for i in range(num_intervals):
        mass += pmf[i]
        cmf[i] = mass

    return cmf

# Given a list representing a discretised cmf, sample num_points points given that
# the range of the cmf is from vmin to vmax.
def sample_from_cmf(cmf, num_points, vmin, vmax):
    num_intervals = len(cmf)
    interval_width = (vmax-vmin)/num_intervals

    noise_rand = np.array([np.random.uniform() for a in range(num_points)])

    noise_sampled = np.zeros(num_points)
    for et in range(num_points):
        for i in range(num_intervals):
            if noise_rand[et] < cmf[i]:
                noise_sampled[et] = vmin + (i * interval_width) + (interval_width * np.random.uniform())
                break

    return noise_sampled

def generateTrainingSets(data_points, input_res, output_res):

    # Build a histogram of the data_points, normalised so that it resembles a discretised probability mass function
    # pmf takes
    # An initial distribution (empty as we will populate using the data points),
    # An origin value for each variable (zeros)
    # The cell (bin) widths for each dimension: 1.0/input_res for all input variables then 1.0/output_res for the final dimension
    # An epsilon value below which to ignore very small probability mass values - in this case we want as accurate as possible so set to 0
    data_pmf = pmf(np.array([]), np.zeros([data_points.shape[0]]), np.concatenate([np.ones([data_points.shape[0]-1])/input_res, np.array([1.0/output_res])], axis=0), _mass_epsilon=0.0)
    data_points = np.transpose(data_points)
    data_pmf.generateInitialDistribtionFromSample(data_points)

    training_input = []
    training_output = []
    for p in data_points:
        # Find the discretised coordinates of p
        coords = data_pmf.calcCellCoordFromPoint(p)
        # Causal Jazz retrieves the conditional distribution for the given discretised coordinates
        strip = data_pmf.calcConditionalSliceFromCoords([a for a in range(len(input_res))], coords[:-1])
        # if the strip is empty, ignore it
        if len(strip.keys()) == 0:
            continue

        # Shift the distribution to the correct position in the double sized ANN output array
        t_output = np.zeros(output_res*2)
        mass_sum = 0.0
        for k,v in strip.items():
            t_output[int(k[0]) + output_res] = v
            mass_sum += v

        if mass_sum != 0:
            t_output /= mass_sum
        # Add this training pair
        training_input += [p[:-1]]
        training_output += [t_output]

    return np.array(training_input), np.array(training_output)

def trainANNForMC(model_name, generate_model, data_points, input_res, output_res):

    # Forst generate a model for the expected value
    x_input = tf.keras.Input(shape=(data_points.shape[0]-1,))
    z = layers.Dense(200, activation='relu')(x_input)
    z = layers.Dense(200, activation='relu')(z)
    z_out = layers.Dense(1)(z)
    model_exp = tf.keras.Model(inputs=x_input, outputs=[z_out], name=model_name)

    if generate_model:
        # Helper callback function for the ANN to stop training early if we reach a minimum loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    restore_best_weights=True)
        # Train the model and save the weights in a file for quick retrieval later
        model_exp.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model_exp.fit(x=np.array(data_points[:-1,:].T), y=np.array(data_points[-1,:]), epochs=2000, batch_size=100, callbacks=[callback], verbose=1, validation_split=0.3)
        model_exp.save_weights(model_name + '_exp.weights.h5')

    else:
        model_exp.load_weights(model_name +'_exp.weights.h5')

    def func_expected(y):
        return model_exp.predict(np.array(y), verbose=False).T[0]

    # For technical reasons, we set the full output length to 2 times the output_res + buffer.
    # Causal Jazz stores only the relative coordinates of each distribution from an origin.
    # but the ANN only returns a list of values without reference.
    # By doubling the output size, we can set the central value of the ANN output to be the origin coord
    # of the distribution and still ensure we capture everything.
    strip_length = 2*output_res

    # Define a simple ANN to estimate the discretised conditional distribution P(X2|X1) for each input point, X1
    x_input = tf.keras.Input(shape=(data_points.shape[0]-1,))
    z = layers.Dense(200, activation='relu')(x_input)
    z = layers.Dense(200, activation='relu')(z)
    z_out = layers.Dense(strip_length, activation="softmax")(z)
    model = tf.keras.Model(inputs=x_input, outputs=[z_out], name=model_name)
    #model.summary()

    if generate_model:
        # Calculate the difference between the output variable and the expected value
        expected_values = func_expected(data_points[:-1,:].T)
        diff = np.concatenate([np.zeros(data_points[:-1,:].shape), np.reshape(expected_values, (1,data_points.shape[1]))], axis=0)
        data_points -= diff

        # Build the training data
        training_input, training_output = generateTrainingSets(data_points, input_res, output_res)

        # Helper callback function for the ANN to stop training early if we reach a minimum loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    restore_best_weights=True)

        # Train the model and save the weights in a file for quick retrieval later
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model.fit(x=np.array(training_input), y=np.array(training_output), epochs=2000, batch_size=100, callbacks=[callback], verbose=1, validation_split=0.3)
        model.save_weights(model_name + '.weights.h5')

    else:
        model.load_weights(model_name +'.weights.h5')

    # Build a function to take a set of points and return a matching set of estimated conditional distributions
    # This will be used by Causal Jazz to build the final joint distribution
    def func_noise(y):
        dist = model.predict(np.array(y), verbose=False)
        # rescale the output to ensure the probability sums to 1
        # this isn't guaranteed by the ANN but it should be close if its trained correctly
        dist /= np.stack([np.sum(dist, axis=1) for a in range(dist.shape[1])]).T
        sample = [pmf(a, np.array([-1.0]), np.array([1.0/output_res]), _mass_epsilon=0.0).sample(1)[0] for a in dist.tolist()]
        return np.array(sample).T[0]

    return func_expected, func_noise

def trainANNForPD(model_name, generate_model, data_points, input_res, output_res, output_buffer):

    # Forst generate a model for the expected value
    x_input = tf.keras.Input(shape=(data_points.shape[0]-1,))
    z = layers.Dense(200, activation='relu')(x_input)
    z = layers.Dense(200, activation='relu')(z)
    z_out = layers.Dense(1)(z)
    model_exp = tf.keras.Model(inputs=x_input, outputs=[z_out], name=model_name)

    if generate_model:
        # Helper callback function for the ANN to stop training early if we reach a minimum loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    restore_best_weights=True)
        # Train the model and save the weights in a file for quick retrieval later
        model_exp.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model_exp.fit(x=np.array(data_points[:-1,:].T), y=np.array(data_points[-1,:]), epochs=2000, batch_size=100, callbacks=[callback], verbose=1, validation_split=0.3)
        model_exp.save_weights(model_name + '_exp.weights.h5')

    else:
        model_exp.load_weights(model_name +'_exp.weights.h5')

    def func_expected(y):
        return model_exp.predict(np.array(y), verbose=False)

    # For technical reasons, we set the full output length to 2 times the output_res + buffer.
    # Causal Jazz stores only the relative coordinates of each distribution from an origin.
    # but the ANN only returns a list of values without reference.
    # By doubling the output size, we can set the central value of the ANN output to be the origin coord
    # of the distribution and still ensure we capture everything.
    strip_length = 2*(output_res + output_buffer)

    # Define a simple ANN to estimate the discretised conditional distribution P(X2|X1) for each input point, X1
    x_input = tf.keras.Input(shape=(data_points.shape[0]-1,))
    z = layers.Dense(200, activation='relu')(x_input)
    z = layers.Dense(200, activation='relu')(z)
    z_out = layers.Dense(strip_length, activation="sigmoid")(z)
    model = tf.keras.Model(inputs=x_input, outputs=[z_out], name=model_name)
    #model.summary()

    if generate_model:
        # Calculate the difference between the output variable and the expected value
        expected_values = func_expected(data_points[:-1,:].T)
        diff = np.concatenate([np.zeros(data_points[:-1,:].shape), np.reshape(expected_values, (1,data_points.shape[1]))], axis=0)
        data_points -= diff

        # Build the training data
        training_input, training_output = generateTrainingSets(data_points, input_res, output_res, output_buffer)

        # Helper callback function for the ANN to stop training early if we reach a minimum loss
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10,
                                                    restore_best_weights=True)

        # Train the model and save the weights in a file for quick retrieval later
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
        model.fit(x=np.array(training_input), y=np.array(training_output), epochs=2000, batch_size=100, callbacks=[callback], verbose=1, validation_split=0.3)
        model.save_weights(model_name + '.weights.h5')

    else:
        model.load_weights(model_name +'.weights.h5')

    # Build a function to take a set of points and return a matching set of estimated conditional distributions
    # This will be used by Causal Jazz to build the final joint distribution
    def func_noise(y):
        test = model.predict(np.array(y), verbose=False)
        # rescale the output to ensure the probability sums to 1
        # this isn't guaranteed by the ANN but it should be close if its trained correctly
        test /= np.stack([np.sum(test, axis=1) for a in range(test.shape[1])]).T
        return test.T

    return func_expected, func_noise