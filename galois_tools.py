#!/usr/bin/env python3
"""
DESCRIPTION:
This module contains functions for the manipulation of matrix and scalar
variables over the Galois field GF(2)--that is, integers modulo 2: {0,1}.

FUNCTIONS IN MODULE:
    - companion_matrix()
    - matmul_gf2()
    - matmul_gf2_recursion()
    - matrix_power_gf2()
"""

###############################################################################
#                            Import dependencies                              #
###############################################################################
from numpy import matmul, transpose, identity, hstack, vstack, shape, zeros
from numpy import expand_dims

###############################################################################
#  Form a companion matrix from a vector of polynomial feedback coefficients  #
###############################################################################

def companion_matrix(c, config='galois'):
    """
    DESCRIPTION:
    This function computes a companion matrix corresponding to the linear
    recurrence relation with polynomial feedback tap coefficients defined by
    vector c = [c0,c1,c2,...,cN]^T. For general information on companion mat-
    rices, visit: https://en.wikipedia.org/wiki/Companion_matrix. For more
    information pertaining to this particular implementation, see: https://...
    ...en.wikipedia.org/wiki/Linear-feedback_shift_register#Matrix_forms. In
    addition to a set of (N + 1) polynomial coefficients in vector c, this
    function accepts a config parameter ('galois' by default) which specifies
    whether the Galois or Fibonacci form of companion matrix / shift register
    is desired. This function accommodates both behaviors, and in each case,
    constructs a companion matrix as a block matrix comprised of 3 blocks.
    Block 1 is an N x N identity matrix. Block 2 is a length-N zero vector.
    Block 3 is the input vector, c, of polynomial coefficients defining the
    linear recurrence relation. This function will accept c as either a list
    or numpy array; the latter is allowed to have either shape (N + 1,) or
    shape (N + 1,1). Upon construction, the companion matrix is returned.

    INPUTS & OUTPUTS:
    :param c: polynomial feedback coeff. vector of shape (N + 1,) or (N + 1,1)
    :type c: array-like (e.g., numpy.ndarray or list)
    :param config: specify recurrence format; options: {'galois','fibonacci'}
    :type config: str
    :returns: (N + 1) x (N + 1) companion matrix defining a linear recurrence
    :rtype: numpy.ndarray
    """
    ###################################################
    # Compute input coefficient vector dimensionality #
    ###################################################
    # Length of input vector c
    N = len(c) - 1

    # Check if singleton dim. needed...
    if len(shape(c)) == 1:

        # Make c a proper (N + 1) x 1 vector
        c = expand_dims(c, axis=1)

    ################################
    # Define identity matrix block #
    ################################

    # Block 1: an N x N identity matrix
    I = identity(N)

    ###############################################
    # Build companion matrix using form specified #
    ###############################################

    # Choice 1: Galois form...
    if config == 'galois':

        # Block 2: 1 x N zero vector
        zero_vector = zeros([1, N])

        # Return companion matrix (Galois configuration)
        return hstack((c, vstack((I, zero_vector))))

    # Choice 2: Fibonacci form...
    if config == 'fibonacci':

        # Block 2: N x 1 zero vector
        zero_vector = zeros([N, 1])

        # Return companion matrix (Fibonacci configuration)
        return vstack((hstack((zero_vector, I)), transpose(c)))

    # Invalid config argument notification & return
    print('Please provide valid config argument to companion_matrix().')
    print('Valid options:')
    print('    - "galois" (default)')
    print('    - "fibonacci"')
    return None

###############################################################################
#         Modulo-2 matrix multiplication over the Galois field GF(2)          #
###############################################################################

def matmul_gf2(A, B):
    """
    DESCRIPTION:
    This function computes the [binary] matrix product AB over Galois field
    GF(2); that is, AB mod 2. It is assumed that input arguments A and B are
    both square binary matrices (i.e., with elements 0 or 1 in GF(2)). For
    more information on Galois fields, navigate to the following Wikipedia
    page: https://en.wikipedia.org/wiki/Finite_field (interesting topic). The
    output matrix is binary and of identical dimensionality to inputs A and B.

    INPUTS & OUTPUTS:
    :param A: left-hand matrix in product AB (over field GF(2))
    :type A: numpy.ndarray
    :param B: right-hand matrix in product AB (over field GF(2))
    :type B: numpy.ndarray
    :returns: product of matrices A and B modulo 2 (over Galois field GF(2))
    :rtype: numpy.ndarray
    """
    # Return matrix product AB (mod 2)
    return matmul(A, B) % 2

###############################################################################
#          Modulo-2 matrix matrix power over the Galois field GF(2)           #
###############################################################################

def matmul_gf2_recursion(A, B, N, n):
    """
    DESCRIPTION:
    This function executes a recurrence relation A <-- AB (mod 2) over indices
    ranging from lower bound n to upper bound N. That is, for indices k = n,
    n + 1, ..., N - 2, N - 1, we overwrite the binary square matrix A as the
    matrix product AB modulo 2 (i.e., AB over Galois field GF(2)). The closed
    form solution eventually output by this function is AB^(N - 1 - n), for
    some lower & upper bounds [n,N]. So if initial index n = 0, the result of
    the recursion could be written in closed form as AB^(N - 1). This function
    is implemented by calling itself on matrices A <-- AB (mod 2) and B <-- B,
    for an incremented/updated index n <-- n + 1 (and fixed upper bound N).
    The stopping condition for the recursive call is index n equaling N - 1, at
    which point this function computes a final product AB^{N - 2 - n}B (mod 2)
    to arrive at the final result. It is assumed that matrices A and B are
    binary (i.e., with elements in GF(2)). For matrix multiplication over
    GF(2), this function makes use of function matmul_gf2() in this module.
    The returned matrix AB^(N - 1 - n) is also binary and square like A and B.

    INPUTS & OUTPUTS:
    :param A: left-hand matrix in product AB (over field GF(2))
    :type A: numpy.ndarray
    :param B: right-hand matrix in product AB (over field GF(2))
    :type B: numpy.ndarray
    :param N: this function outputs AB^(N-1-n) mod 2, for upper bound N
    :type N: int
    :param n: lower bound on index tracking the latest matrix power computed
    :type n: int
    :returns: if complete, outputs AB^(N-1-n) mod 2; self-call, if incomplete
    :rtype: numpy.ndarray or recursive call (depending on value of n)
    """
    # Terminate recursion @ AB^(N-1)...
    if n == N - 1:

        # Final matrix product over GF(2)
        return matmul_gf2(A, B)

    # Call recursive matrix product A*B*B*B... = AB^k over GF(2)
    return matmul_gf2_recursion(matmul_gf2(A, B), B, N, n + 1)

###############################################################################
#   Modulo-2 k'th matrix power over the Galois field GF(2) (aka F_2 or Z_2)   #
###############################################################################

def matrix_power_gf2(A, k):
    """
    DESCRIPTION:
    This function leverages the recursive function matmul_gf2_recursion() for
    computation of the binary matrix power A^k (mod 2), i.e., A^k over Galois
    field GF(2). There are 2 edge cases which this function handles manually.
    First, the zero'th power of any elligible N x N matrix, A, is the N x N
    identity matrix. The first power is, trivially, the original matrix A.
    This holds true over GF(2). All other cases are handled via recursive mat-
    rix multiplication: ((((A x A mod 2) x A mod 2) x A mod 2)... x A mod 2).
    On completion of the recursion, the N x N binary result is returned.

    INPUTS & OUTPUTS:
    :param A: binary matrix of which we are taking the k'th power (modulo 2)
    :type A: numpy.ndarray
    :param k: power of binary matrix A this function computes (via recursion)
    :type k: int
    :returns: binary matrix power A^k (mod 2), taken over Galois field GF(2)
    :rtype: numpy.ndarray
    """
    ###############################
    # 0'th power: identity matrix #
    ###############################

    # Base case: A^0 = I
    if k == 0:

        # Return identity matrix I
        return identity(A.shape[0])

    ################################
    # 1st power: original matrix A #
    ################################

    # Trival case: A^1 = A
    if k == 1:

        # Return original matrix A
        return A

    #########################################################
    # k'th power, k > 1: compute matrix power via recursion #
    #########################################################

    # Call recursion matrix multiplication over GF(2)
    return matmul_gf2_recursion(A, A, k, 0)
