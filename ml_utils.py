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
- tapped_register_observations()
- register_state_observations()
"""
###############################################################################
#                            Import dependencies                              #
###############################################################################

# Numpy
from numpy import zeros, roll, array, vstack, arange

# Tensorflow/Keras functionality
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import TruePositives, FalsePositives
from tensorflow.keras.metrics import TrueNegatives, FalseNegatives

# Plotting functionality
from matplotlib import pyplot as plt

# Galois tools
from galois_tools import str2vec, int2bin

###############################################################################
# Feedforward binary neural network for prediction of LFSR state n + 1 from n #
###############################################################################

def feedforward_lfsr_predictor(deg, num_hidden, learning_rate,
                               print_summary=True):
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
    :param num_hidden: number of hidden layer nodes
    :type num_hidden: int
    :param learning_rate: positive learning rate for Keras RMSprop optimizer
    :type learning_rate: float
    :param print_summary: print the digest output by model.summary() if True
    :type print_summary: bool
    :returns: connected feedforward network approximating LFSR
    :rtype: tensorflow.keras.models.Model
    """
    ##################################################
    # Define feedforward binary network architecture #
    ##################################################

    # Input layer (LFSR @epoch n)
    x = Input(shape=(deg,), name='input')

    # Single hidden layer, sigmoid activations
    h = Dense(num_hidden, activation='sigmoid',
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

###############################################################################
# Tap some p'th LFSR from a list thereof to form state vector obs. sequences  #
###############################################################################

def tapped_register_observations(shift_registers, p, deg, tap=0x1):
    """
    DESCRIPTION:
    This function accepts a list of LFSR() instances, the index p of a partic-
    ular register choice for training set generation, and the order of the re-
    cursion, deg (degree of the defining primitive polynomials over GF(2)).
    The p'th register is tapped at some tap set specified at input (logical &
    used to tap register state), and the resulting bit stream is sourced from
    the tap set specified. The bits are then buffered into rows of quasi-LFSR
    states--but not actual LFSR states! These are simply deg-length binary
    observations sourced from a tapped source register (via logical &). The
    n'th state vector for n = 0,1,2,... is stored sequentially in the rows of
    returned array X; the corresponding true future (n + 1)'st state per row
    of X are stored, 1:1, in the rows of the returned binary array Y.

    INPUTS & OUTPUTS:
    :param shift_registers: list of LFSR() class instances of order deg
    :type shift_registers: list, dtype=galois_tools.LFSR
    :param p: index of the register in the list to form observations from
    :type p: int
    :param deg: degree of primitive polynomial for LFSR recurrence
    :type deg: int
    :param tap: tap choice for reading off register state bits
    :type tap: hex or int (< 2^deg)
    :returns: total 2^deg - 1 observations of current & future state vectors
    :rtype: numpy.ndarray (x2 outputs)
    """
    #########################################################
    # Buffer in b[n] a total of 2^deg + deg m-sequence bits # - yes, inefficient
    #########################################################

    # Maximimum length binary sequences of length M = 2^deg - 1 via LFSRs
    b = vstack(tuple(register.stream(2**deg + deg, tap=tap)
                     for register in shift_registers))

    ########################################################
    # n'th LFSR register state observations, n = 0,1,2,... #
    ########################################################

    # Input LFSR windows (deg-bit observation vectors)
    X = array([b[p, n:n + deg]
               for n in range(b.shape[1] - deg - 1)])

    ############################################################
    # (n+1)'st LFSR register state observations, n = 0,1,2,... #
    ############################################################

    # Target LFSR windows, 1 epoch into future from X
    Y = array([b[p, n + 1:1 + n + deg]
               for n in range(b.shape[1] - deg - 1)])

    # Return current & future observations
    return X, Y

###############################################################################
# Tap some p'th LFSR from a list thereof to form state vector obs. sequences  #
###############################################################################

def register_state_observations(lfsr):
    """
    DESCRIPTION:
    This function accepts a list of LFSR() instances, the index p of a partic-
    ular register choice for training set generation, and the order of the re-
    cursion, deg (degree of the defining primitive polynomials over GF(2)).
    The binary lfsr.N-order register state vector is saved for each time epoch
    in a length 2^lfsr.N - 1 maximum length LFSR recursion. To simulate present
    n'th and future (n + 1)'st observation vectors for ML applications, this
    function circularly shifts the buffered LFSR state vector progression,
    producing a time-shifted copy of the same period of data. This can be used,
    for example, to predict state (n + 1) given some arbitrary n'th state. The
    present and future LFSR state progressions are returned as numpy arrays.

    INPUTS & OUTPUTS:
    :param shift_registers: list of LFSR() class instances of order deg
    :type shift_registers: list, dtype=galois_tools.LFSR
    :param p: index of the register in the list to form observations from
    :type p: int
    :param deg: degree of primitive polynomial for LFSR recurrence
    :type deg: int
    :returns: total 2^deg - 1 observations of current & future state vectors
    :rtype: numpy.ndarray (x2 outputs)
    """
    # Period of max-length LFSR recursion
    M = 2**lfsr.N - 1

    # Shift register state vector observations
    X = zeros([M, lfsr.N]) # Current n'th obs.

    # For each n'th epoch...
    for n in range(M):
        X[n, :] = str2vec(int2bin(lfsr.state, lfsr.N))
        lfsr.recurse()

    # Future (n + 1)'st observations (circular shift)
    Y = roll(X, -1, axis=0)

    # Return LFSR state sequences
    return X, Y

###############################################################################
#              Plot {TP,TN,FP,FN} metrics versus training epoch               #
###############################################################################

def plot_metrics_vs_epoch(model_history, N_epoch):
    """
    DESCRIPTION:
    This function plots true positives (TP), true negatives (TN), false posi-
    tives (FP), and false negatives (FN) versus training epoch, given a total
    training epoch count N_epoch and Keras model_history provided at input.

    INPUTS & OUTPUTS:
    :param model_history: feedforward binary/sigmoidal net training history
    :type model_history: tensorflow.python.keras.callbacks.History
    :param N_epoch: number of training epochs (x-axis grid length)
    :type N_epoch: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # New figure
    plt.figure(figsize=(9.9, 8))

    #####################################
    # True positives vs. training epoch #
    #####################################

    # New axes
    ax = plt.subplot(4, 1, 1)

    # Grid current axes
    plt.grid(c='k', alpha=0.25)

    # Plot true positives (TP) vs training epoch
    plt.plot(arange(1, N_epoch + 1),
             model_history.history['true_positives'], c='k')

    # Horizontal axis limits
    plt.xlim([1, N_epoch])

    # Horizontal axis label
    plt.xlabel(r'Training Epoch $\varepsilon$')

    # Vertical axis label
    plt.ylabel(r'TP${}_\varepsilon$')

    # Vertical axis label
    plt.title(r'True Positives (TP) vs. Training Epoch', weight='bold')

    ######################################
    # False positives vs. training epoch #
    ######################################

    # New axes
    plt.subplot(4, 1, 2, sharex=ax)

    # Grid current axes
    plt.grid(c='k', alpha=0.25)

    # Plot false positives (FP) vs training epoch
    plt.plot(arange(1, N_epoch + 1),
             model_history.history['false_positives'], c='k')

    # Horizontal axis limits
    plt.xlim([1, N_epoch])

    # Horizontal axis label
    plt.xlabel(r'Training Epoch $\varepsilon$')

    # Vertical axis label
    plt.ylabel(r'FP${}_\varepsilon$')

    # Vertical axis label
    plt.title(r'False Positives (FP) vs. Training Epoch', weight='bold')

    #####################################
    # True negatives vs. training epoch #
    #####################################

    # New axes
    plt.subplot(4, 1, 3, sharex=ax)

    # Grid current axes
    plt.grid(c='k', alpha=0.25)

    # Plot false positives (FP) vs training epoch
    plt.plot(arange(1, N_epoch + 1),
             model_history.history['true_negatives'], c='k')

    # Horizontal axis limits
    plt.xlim([1, N_epoch])

    # Horizontal axis label
    plt.xlabel(r'Training Epoch $\varepsilon$')

    # Vertical axis label
    plt.ylabel(r'TN${}_\varepsilon$')

    # Vertical axis label
    plt.title(r'True Negatives (TN) vs. Training Epoch', weight='bold')

    ######################################
    # False negatives vs. training epoch #
    ######################################

    # New axes
    plt.subplot(4, 1, 4, sharex=ax)

    # Grid current axes
    plt.grid(c='k', alpha=0.25)

    # Plot false positives (FP) vs training epoch
    plt.plot(arange(1, N_epoch + 1),
             model_history.history['false_negatives'], c='k')

    # Horizontal axis limits
    plt.xlim([1, N_epoch])

    # Horizontal axis label
    plt.xlabel(r'Training Epoch $\varepsilon$')

    # Vertical axis label
    plt.ylabel(r'FN${}_\varepsilon$')

    # Vertical axis label
    plt.title(r'False Negatives (TN) vs. Training Epoch', weight='bold')

    # Optimize subplot layout
    plt.tight_layout()
