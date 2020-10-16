#!/usr/bin/env python3
"""
AUTHOR:
Ryan Burns

DESCRIPTION:
This module contains functions for the visualization of m-sequences and
their associated transforms, distributions, properties, etc.

FUNCTIONS IN MODULE:
- correlation_example()
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
