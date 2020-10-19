#!/usr/bin/env python3
"""
AUTHOR:
Ryan Burns

DESCRIPTION:
This module contains functions for the visualization of m-sequences and
their associated transforms, distributions, properties, etc. Also within
scope are plots relating to the training of a feedforward binary neural net
to predict m-sequences, representing the underlying LFSR's finite field
recursion/orbit via latent (weighted) sigmoidal layer connections. While
it is assumed that much of the math computations happen in other modules,
this module will be devoted to visualization of the resulting sequences,
properties, statistics/metrics, etc.

FUNCTIONS IN MODULE:
- correlation_example()
- true_vs_predicted_masks()
"""
###############################################################################
#                            Import dependencies                              #
###############################################################################

# Numpy functions
from numpy import arange, correlate

# Import pyplot
from matplotlib import pyplot as plt

###############################################################################
#  Plot normalized autocorrelations and cross-correlation of two m-sequences  #
###############################################################################

def correlation_example(m_sequence0, m_sequence1, deg):
    """
    DESCRIPTION:
    This function accepts two (mutually orthogonal) maximal length LFSR
    sequences, or m-sequences. Full periods of each sequence (length 2^deg-1)
    are assumed. The (normalized) autocorrelation of each m-sequence is compu-
    ted, along with their normalized cross-correlation. The autocorrelation of
    m-sequence 0 is plotted in the top subplot; the autocorrelation of sequence
    1 us plotted in the middle subplot; and the cross-correlation of the two
    sequences is plotted in the bottom subplot. No values are returned.

    INPUTS & OUTPUTS:
    :param m_sequence0: m-sequence indexed 0 (binary, maximially random)
    :type m_sequence0: numpy.ndarray
    :param m_sequence1: m-sequence indexed 1 (binary, maximially random)
    :type m_sequence1: numpy.ndarray
    :param deg: degree of the feedback polynomial which produced m-sequence
    :type deg: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    #########################################
    # Pre-plotting correlation computations #
    #########################################
    # m-sequence length
    sequence_length = len(m_sequence0)

    # Correlation delays (bits/samples)
    delays = arange(-sequence_length + 1, sequence_length)

    # Normalized autocorrelation of m-sequence 0
    autocorr0 = correlate(m_sequence0, m_sequence0, 'full') / sequence_length

    # Normalized autocorrelation of m-sequence 1
    autocorr1 = correlate(m_sequence1, m_sequence1, 'full') / sequence_length

    # Normalized cross-correlation b/w m-sequences 0 and 1
    crosscorr = correlate(m_sequence0, m_sequence1, 'full') / sequence_length

    ######################################
    # PLOT: m-sequence 0 autocorrelation #
    ######################################

    # New figure
    plt.figure(figsize=(9.9, 6))

    # Save these axes as ax
    top_axis = plt.subplot(3, 1, 1)

    # Add grid
    plt.grid(c='k', alpha=0.25)

    # Plot autocorrelation of m-sequence 0 of collection
    plt.plot(delays, autocorr0, c='k', label=(
        r'$m$-sequence length $M=2^m-1 = '
        + str(int(2**deg - 1))
        + r',\quad m = ' + str(deg) + r'$'))

    # Set axis limits
    plt.xlim([-sequence_length + 1, sequence_length])
    plt.ylim([-0.1, 1])

    # Title
    plt.title(r'Autocorrelation of m-sequence 0', weight='bold')

    # Label axes
    plt.ylabel(r'Normalized')
    plt.xlabel(r'Delay (bits)')

    # Legend
    plt.legend()

    ######################################
    # PLOT: m-sequence 1 autocorrelation #
    ######################################

    # 2nd subplot on common axis
    plt.subplot(3, 1, 2, sharex=top_axis)

    # Add grid
    plt.grid(c='k', alpha=0.25)

    # Plot autocorrelation of m-sequence 1 of collection
    plt.plot(delays, autocorr1, c='k', label=(
        r'$m$-sequence length $M=2^m-1 = '
        + str(int(2**deg - 1))
        + r',\quad m = ' + str(deg) + r'$'))

    # Set axis limits
    plt.xlim([-sequence_length + 1, sequence_length])
    plt.ylim([-0.1, 1])

    # Title
    plt.title(r'Autocorrelation of m-sequence 1', weight='bold')

    # Label axes
    plt.ylabel(r'Normalized')
    plt.xlabel(r'Delay (bits)')

    # Legend
    plt.legend()

    ############################################
    # PLOT: m-sequence 0 & 1 cross-correlation #
    ############################################

    # 3rd subplot on common axis
    plt.subplot(3, 1, 3, sharex=top_axis)

    # Add grid
    plt.grid(c='k', alpha=0.25)

    # Plot cross-correlation of m-sequences 0 and 1
    plt.plot(delays, crosscorr, c='k', label=(
        r'$m$-sequence length $M=2^m-1 = '
        + str(int(2**deg - 1))
        + r',\quad m = ' + str(deg) + r'$'))

    # Set axis limits
    plt.xlim([-sequence_length + 1, sequence_length])
    plt.ylim([-0.1, 1])

    # Title
    plt.title(r'Cross-correlation of m-sequences 0 and 1', weight='bold')

    # Label axes
    plt.ylabel(r'Normalized')
    plt.xlabel(r'Delay (bits)')

    # Legend
    plt.legend()

    # Optimize subplot layout
    plt.tight_layout()

###############################################################################
#   True LFSR state mask at indices n vs predicted masks at indices (n + 1)   #
###############################################################################

def true_vs_predicted_masks(idx0, idx1, input_obs, output_activations):
    """
    DESCRIPTION:
    This function plots the true n'th LFSR mask vs time index n (in subplot 1)
    parallel to the predicted (n + 1)'st LFSR mask (in subplot 2). Technically
    what are plotted (using a binary colormap) in the later mask image are
    sigmoidal binary class activations output by a binary feedforward, bias-
    free 2-layer network for LFSR state prediction. Since the range of the
    sigmoidal function is the unit interval [0,1], these values are interpret-
    able as the probability of each bit being 1. Thresholding these values at
    0.5 would form a decision process on these binary bit predictions / acti-
    vations, but this function opts to simply plot the activations. The net
    is assumed to have 100% accuracy in LFSR future state prediction, so its
    sigmoidal output bit probabilities will extremely close to 0 and 1, resp.
    This function simply plots the masks in parallel and returns no data. The
    masks are plotted from input indices idx0 to idx1, with idx0 < idx1.

    INPUTS & OUTPUTS:
    :param idx0: leading index of the interval plotted of true & predicted
    :type idx0: int
    :param idx1: trailing index of the interval plotted of true & predicted
    :type idx1: int
    :param input_obs: LFSR state observation sequence (as rows of the array)
    :type input_obs: numpy.ndarray, dtype=int
    :param output_activations: neural net's future LFSR state predictions
    :type output_activations: numpy.ndarray, dtype=float
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # Set up shop in new figure
    plt.figure(figsize=(5.5, 9.9))

    #################################
    # Left subplot: true LFSR state #
    #################################

    # New subplot
    plt.subplot(1, 2, 1)

    # Plot true LFSR bits/mask vs index n
    plt.imshow(input_obs[idx0:idx1, :],
               aspect='auto',
               cmap='binary',
               interpolation='none')

    # Observation index axis label
    plt.ylabel(r'Observation Index / Epoch $n$',
               weight='bold')

    # Current LFSR state footer / label
    plt.xlabel(r'Observed LFSR State $n$', weight='bold')

    #######################################
    # Right subplot: predicted LFSR state #
    #######################################

    # New subplot
    plt.subplot(1, 2, 2)

    # Image of sigmoidal output activations in [0,1]
    # (NOTE: Elements will likely be close to 0 and 1.)
    plt.imshow(output_activations[idx0:idx1, :],
               aspect='auto', cmap='binary',
               interpolation='none')

    # Remove right plot's y-ticks
    plt.yticks([])

    # Predicted state footer / label
    plt.xlabel(r'Neural Net Predicted State $n + 1$',
               weight='bold')

    # Optimize subplot layout
    plt.tight_layout()
