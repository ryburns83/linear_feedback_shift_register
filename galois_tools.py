#!/usr/bin/env python3
"""
AUTHOR:
Ryan Burns

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

# Random helpful tools
from time import time

# Numpy
from numpy import matmul, transpose, identity, hstack, vstack, shape, zeros
from numpy import expand_dims, array

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
#             Modulo-2 k'th matrix power over Galois field GF(2)              #
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

###############################################################################
#   Load primitive polynomial coefficients [over GF(2)] of specified degree   #
###############################################################################

def read_polynomials_from_file(deg=32, path='./'):
    """
    DESCRIPTION:
    Load a list of primitive polynomial coefficients from file, stored as hex
    strings (using a lowercase convention for integers a-f)

    INPUTS & OUTPUTS:
    :param deg: specify polynomial degree (aka recursion order)
    :type deg: int
    :param path: specify path at which coefficient .txt file is stored
    :type path: str
    :returns: list of hexadecimal (str) primitive polynomial coefficients
    :rtype: list, dtype=str
    """
    f = open(path + str(deg) + '.txt', 'r')
    coefficients = f.read()
    f.close()
    return [
        ('0x' + coeff.lower()) for coeff in coefficients.split('\n')
        if len(coeff) > 0]

###############################################################################
#        Convert hexadecimal integer to binary string representation          #
###############################################################################

def hex2bin(hex_string, nbits=None):
    """
    DESCRIPTION:
    This function accepts a hexadecimal string (beginning with characters '0x'
    and having lowercase formatting for integers a-e) and converts it to a
    binary string representation. Optionally, a fixed number of bits can be
    specified, enforcing that the output bit vector have a certain fixed
    length. The default behavior is to set this argument to None, opting for
    the shortest possible binary string representation to be returned.

    INPUTS & OUTPUTS:
    :param hex_string: hexadecimal integer specified as a string
    :type hex_string: str
    :param nbits: specify the length of the bit vector to be returned
    :type nbits: int or None
    :returns: binary string representation of hexadecimal (str) integer
    :rtype: str
    """
    # Check that hex_string is valid...
    if hex_string[0:2] != '0x':

        # Print invalid string warning
        print('Input hex_string must be a hex string beginning with 0x.')
        return None

    # If no specified bit vector length...
    if nbits is None:

        # Return minimum possible total bits
        return bin(int(hex_string, 16))

    # Return the bit vector of specified length nbits
    return bin(int(hex_string, 16)).zfill(nbits)

###############################################################################
#     Convert binary string representation to binary numpy array (vector)     #
###############################################################################

def str2vec(b):
    """
    DESCRIPTION:
    This function takes a binary number b (string representation with leading
    '0b' characters) and casts it as a binary numpy array (bit vector). The
    array has dtype='int64' and length equal to that of input bit string b.

    INPUTS & OUTPUTS:
    :param b: string representation of bit vector (or polynomial over GF(2))
    :type b: str
    :returns: a binary numpy array representation of the input bit string
    :rtype: numpy.ndarray, dtype='int64'
    """
    return array([int(bit) for bit in b.replace('0b', '')])

###############################################################################
#    Convert binary vector representation to binary string representation     #
###############################################################################

def vec2str(b):
    """
    DESCRIPTION:
    This function accepts a 1 x L binary vector, for an arbitrary integer
    length L, and returns a length-L string representation of the binary
    input vector b (assumed to be a numpy array).

    INPUTS & OUTPUTS:
    :param b: binary numpy array for conversion to str
    :type b: numpy.ndarray
    :returns: string of '0' and '1' chars via input vector b
    :rtype: str
    """
    # Return a string-cast binary vector from numpy array b
    return ''.join([str(int(bit)) for bit in b])

###############################################################################
# Linear feedback shift register (LFSR) class defining recurrence over GF(2)  #
###############################################################################

class LinearAlgebraicLFSR():
    """
    DESCRIPTION:
    A linear algebraic, object-oriented implementation of a linear feedback
    shift register (LFSR) over Galois field GF(2). Provided a set of feedback
    polynomial coefficents, an initial (seed) state vector, and an optional
    register ID, this class seeks the emulate the behavior of a true LFSR. In
    lieu of integer or binary arithmetic, this class favors a linear algebraic
    implementation of the finite field arithmetic, using matrices modulo 2.

    METHODS:
    - __init__()
    - recurse()
    - summary()
    - cycle()
    - stream()
    """
    #############################
    # LFSR constructor function #
    #############################

    def __init__(self, c, seed, register_id=None):
        """
        DESCRIPTION:
        This is the constructor/initializer function for the LFSR class,
        defining all of the initial member variables for the class. No values
        are returned from this method; it's sole purpose is designation of the
        following member variables:

        - self.feedback_polynomial: linear feedback relation coefficients
        - self.seed: seed register state vector (initial recursion conditions)
        - self.state: the actual LFSR bit storage, initialized with seed
        - self.C: companion matrix defining the linear feedback relation
        - self.T: transition matrix defining the cumulative linear mapping
        - self.ID: a unique str identifier for a given LFSR() class instance
        - self.N: integer order of feedback polynomial & recurrence relation

        INPUTS & OUTPUTS:
        :param self: this particular LinearAlgebraicLFSR() class instance
        :type self: __main__.LinearAlgebraicLFSR
        :param c: feedback polynomial coefficients (register tap weights)
        :type c: array-like (e.g., list or numpy.ndarray)
        :param seed: initial register state as an [base-10] integer
        :type seed: int
        :param register_id: a unique register identifier (string)
        :type register_id: str or None
        :returns: nothing is returned by this method
        :rtype: None
        """
        # Use default register ID?...
        if register_id is None:

            # Define current UNIX/POSIX timestamp as register ID
            self.ID = str(time()).replace('.', '')

        # Define feedback taps using polynomial coefficients provided
        self.feedback_polynomial = c

        # Initialize shift register state vector with seed value provided
        self.seed = seed # Fixed, initial conditions of linear recurrence
        self.state = seed # Initialize LFSR state vector with seed value

        # Define companion matrix for linear recurrence relation
        self.C = companion_matrix(self.feedback_polynomial)

        # Initialize linear transition matrix
        self.T = self.C # (companion matrix)

        # Define the order of the feedback polynomial (state length)
        self.N = len(self.feedback_polynomial) # (units = bits)

        # Epoch index initialization @ zero
        self.epoch = 0

    ######################################################
    # Iterate LFSR recurrence for fixed number of epochs #
    ######################################################

    def recurse(self, num_epoch=1):
        """
        DESCRIPTION:
        Iterate register some fixed num_epoch number of cycles/epochs.
        There are two mechanisms available for implementation:
        (i) multiplication of the state vector by the (num_epoch)'th
            power of the companion matrix self.C;
        (ii) multiply the initial state vector (i.e., self.seed) by a
            net/aggregated transition matrix self.T, which is depend-
            ent on the current epoch/cycle index.
        Of these two choices, we opt for (ii), tracking a represen-
        tation of the net transition matrix self.T across all LFSR
        epochs or iterations. This is effected by multiplying the state
        vector self.seed by the matrix power of a transition matrix
        which is updated as self.C to the (self.epoch)'th matrix power.
        The state vector at the current epoch, self.epoch, is thus
        calculated as lefthand product of net transition matrix self.T
        and the initial state vector self.seed. No values are returned.

        INPUTS & OUTPUTS:
        :param self: this particular LinearAlgebraicLFSR() class instance
        :type self: __main__.LinearAlgebraicLFSR
        :param num_epoch: number of epochs/cycles to recurse LFSR by
        :type num_epoch: int
        :returns: nothing is returned by this method
        :rtype: None
        """
        # Increment total epoch count/track
        self.epoch += num_epoch

        # Update current transition matrix (fixed # of epochs/cycles)
        self.T = matmul_gf2(self.T, matrix_power_gf2(self.C, num_epoch))

        # Apply num_epoch'th power of companion matrix to state
        self.state = matmul_gf2(self.T, self.state)

    #######################################
    # Print LFSR variables summary/digest #
    #######################################

    def summary(self):
        """
        DESCRIPTION:
        This method pretty-prints a summary/digest of all of the member
        variables of this LFSR class instance (nothing is returned).

        INPUTS & OUTPUTS:
        :param self: this particular LinearAlgebraicLFSR() class instance
        :type self: __main__.LinearAlgebraicLFSR
        :returns: nothing is returned by this method
        :rtype: None
        """
        # Print header/banner above LFSR summary
        print('##################################################')
        print('# LINEAR FEEDBACK SHIFT REGISTER (LFSR) SUMMARY: #')
        print('##################################################')

        # Separate header/banner from recurrence order
        print('__________________________________________________')

        # Recurrence order
        print('ORDER:', self.N)

        # Separate recurrence order from epoch
        print('__________________________________________________')

        # Current recurrence epoch/index
        print('EPOCH:', self.epoch)

        # Separate epoch from register ID
        print('__________________________________________________')

        # Print register ID
        print('ID:', self.ID)

        # Separate register ID from register state
        print('__________________________________________________')

        # Print shift register state vector
        print('STATE:', vec2str(self.state))

        # Separate register state from register seed
        print('__________________________________________________')

        # Print seed state vector (i.e., initial conditions)
        print('SEED: ', vec2str(self.seed))

        # Separate register seed from recurrence companion matrix
        print('__________________________________________________')

        # Print companion matrix of recurrence relation
        print('COMPANION MATRIX:')

        # For each row in the companion matrix...
        for row in self.C:

            # Print row (indented)
            print('      ', vec2str(row))

        # Separate recurrence companion matrix from transition matrix
        print('__________________________________________________')

        # Print net transition matrix from seed to current state
        print('TRANSITION MATRIX:')

        # For each row in the transition matrix...
        for row in self.T:

            # Print row (indented)
            print('      ', vec2str(row))

        # Separation of LFSR summary from anything printed after
        print('__________________________________________________')
        print('')

    ############################################
    # Cycle the shift register by single epoch #
    ############################################

    def cycle(self, num_epoch=1, verbose=False):
        """
        DESCRIPTION:
        Recurse the linear feedback shift register state vector 1
        cycle (1 epoch). A single cycle corresponds mathematically
        to multiplication by a companion matrix over Galois field
        GF(2). In verbose mode, this method prints self.state.

        INPUTS & OUTPUTS:
        :param self: this particular LinearAlgebraicLFSR() class instance
        :type self: __main__.LinearAlgebraicLFSR
        :param num_epoch: number of epochs/cycles to recurse LFSR by
        :type num_epoch: int
        :param verbose: controls the printing of self.state
        :type verbose: bool
        :returns: nothing is returned by this method
        :rtype: None
        """
        # For each _'th epoch/cycle...
        for _ in range(num_epoch):

            # Cycle LFSR 1x (i.e., single epoch)
            self.recurse() # (default: num_epoch=1)

            # Verbose mode...
            if verbose:

                # Print state vector as string...
                print(vec2str(self.state))

    #####################################################
    # Stream bit(s) from linear feedback shift register #
    #####################################################

    def stream(self, num_bits=1, tap=-1):
        """
        DESCRIPTION:
        Stream 1 or more bits from the linear feedback shift register
        (LFSR) by tapping a single index of the register and emit-
        ting the bit at that index over a fixed number of epochs. The
        number of epochs is 1:1 with the specified num_bits to be
        buffered and returned by this function. If an invalid number
        of bits is specified or an invalid tap index is specified,
        the function prints a warning message and returns None. If a
        single bit is to be emitted, this function returns an integer.
        Otherwise, this function returns a numpy array of integers.

        INPUTS & OUTPUTS:
        :param self: this particular LinearAlgebraicLFSR() class instance
        :type self: __main__.LinearAlgebraicLFSR
        :param num_bits: # of epochs <==> # of bits output
        :type num_bits: int
        :param tap: state index tapped to produce stream
        :type tap: int
        :returns: nothing is returned by this method
        :rtype: None
        """
        ###################
        # Validate inputs #
        ###################

        # Invalid tap index
        if tap >= self.N:

            # Print a warning about invalid tap
            print('Invalid tap index: must be <= self.N.')
            return None

        # Invalid number of bits?...
        if num_bits < 1:

            # Print a warning about invalid number of bits
            print('Invalid number of bits: must be >= 1.')
            return None

        #############################################
        # Emit single bit from LFSR (cycle 1 epoch) #
        #############################################

        # Emit single bit?...
        if num_bits == 1:

            # Cycle LFSR 1x (i.e., single epoch)
            self.recurse() # (default: num_epoch=1)

            # Return tapped state vector bit (as int)
            return int(self.state[tap]) # (0 or 1)

        # Binary stream storage
        buffer = zeros([num_bits,], dtype='uint64')

        ###########################################
        # Stream & buffer multiple bits from LFSR #
        ###########################################

        # For each n'th bit required...
        for n in range(num_bits):

            # Cycle LFSR 1x (i.e., single epoch)
            self.recurse() # (default: num_epoch=1)

            # Return tapped state vector bit (as int)
            buffer[n] = int(self.state[tap]) # (0 or 1)

        # Return buffered bit stream
        return buffer
