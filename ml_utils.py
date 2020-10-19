#!/usr/bin/env python3
"""
AUTHOR:
Ryan Burns

DESCRIPTION:
This module contains functions for the visualization of a linear feedback
shift register (LFSR) being estimated by a feedforward binary neural network.
The main focus of this module is the plotting functionality, assuming much of
theactual  math of the shift register and neural net is handled externally.

FUNCTIONS IN MODULE:
- feedforward_lfsr_predictor()
"""
###############################################################################
#                            Import dependencies                              #
###############################################################################

# Tensorflow/Keras functionality
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import TruePositives, FalsePositives
from tensorflow.keras.metrics import TrueNegatives, FalseNegatives

###############################################################################
# Feedforward binary neural network for prediction of LFSR state n + 1 from n #
###############################################################################

def feedforward_lfsr_predictor(deg, learning_rate, print_summary=True):
    """
    DESCRIPTION:
    Use Keras to generate dense (fully-connected) feedforward neural network
    for prediction of the (n + 1)'st LFSR state from the n'th LFSR state. This
    function receives the degree of the feedback polynomial defining the LFSR
    taps, the learning rate of the RMSprop optimizer, and a boolean switch
    controlling whether a model digest/summary is printed. The model has three
    layers, an input layer, a hidden sigmoidal layer, and an output sigmoidal
    layer. Metrics tracked: {TP,FP,TN,FN}. Loss: binary cross-entropy error.

    INPUTS & OUTPUTS:
    :param deg: degree of the feedback polynomial defining LFSR
    :type deg: int
    :returns: connected feedforward network approximating LFSR
    :rtype: tensorflow.keras.models.Model
    """
    ##################################################
    # Define feedforward binary network architecture #
    ##################################################

    # Input layer (LFSR @epoch n)
    x = Input(shape=(deg,), name='input')

    # Single hidden layer, sigmoid activations
    h = Dense(2 * deg, activation='sigmoid',
              use_bias=False, name='hidden')(x)

    # Output layer (LFSR @epoch n + 1), sigmoid activations
    y = Dense(deg, activation='sigmoid',
              use_bias=False, name='output')(h)

    # Neural network model
    model = Model(x, y)

    ###################################
    # Compile model, set up optimizer #
    ###################################

    # Use RMSprop for speed of convergence
    opt = RMSprop(learning_rate=learning_rate, epsilon=1e-07)

    # Model compilation using binary cross-entropy error
    model.compile(loss='binary_crossentropy', optimizer=opt,
                  metrics=[TruePositives(), TrueNegatives(),
                           FalsePositives(), FalseNegatives()])

    ##################################
    # Print model summary (optional) #
    ##################################

    # Print summary?....
    if print_summary:

        # Print model digest
        model.summary()

    # Return the compiled model
    return model
