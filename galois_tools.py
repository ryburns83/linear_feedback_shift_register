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

# Numpy functions
from numpy import array, zeros

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
#     Convert a non-negative integer to its binary string representation      #
###############################################################################

def int2bin(integer, num_bits=None):
    """
    DESCRIPTION:
    This function accepts a non-negative integer and returns its binary string
    representation, optionally requiring that the bitstring returned be of
    specified length num_bits. The returned binary integer has prefix '0b'.

    INPUTS & OUTPUTS:
    :param integer: non-negative integer to be converted to a bitstring
    :type integer: int
    :param num_bits: (optional) required length of the output bitstring
    :type num_bits: int
    :returns: binary string representation of integer
    :rtype: str
    """
    # Non-negative integer?...
    if integer < 0:

        # Print negative integer warning and return
        print('Invalid integer < 0 provided to int2bin().')
        return

    # No fixed bitstring length specified...
    if num_bits is None:

        # Default: wrapper for bin()
        return bin(integer)

    # Return binary representation of integer @ w/ bitstring length
    return '0b' + bin(integer).replace('0b', '').zfill(num_bits)

###############################################################################
# Linear feedback shift register (LFSR) class defining recurrence over GF(2)  #
###############################################################################

class LFSR():
    """
    DESCRIPTION:
    A object-oriented implementation of a linear feedback shift register (LFSR)
    over Galois field GF(2). Provided a set of feedback polynomial coefficents,
    as an N-bit integer (for some fixed bit length / order N) an initial
    length-N (seed) state vector, this class seeks the emulate the behavior of
    a true LFSR. In lieu of linear algebraic compsanion-matrix-based implemen-
    tation, this class favors a number theoretic implementation, leveraging
    integer (decimal), binary, & hexadecimal representation of N-bit integers.

    METHODS:
    - __init__()
    - pretty_print()
    - recurse()
    - summary()
    - cycle()
    - stream()
    """
    #############################
    # LFSR constructor function #
    #############################

    def __init__(self, mask, seed, order, register_id=None):
        """
        DESCRIPTION:
        This is the constructor/initializer function for the LFSR class,
        defining all of the initial member variables for the class. No values
        are returned from this method; it's sole purpose is designation of the
        following member variables:

        - self.feedback_polynomial: linear feedback relation coefficients
        - self.seed: seed register state vector (initial recursion conditions)
        - self.state: the actual LFSR bit storage, initialized with seed
        - self.ID: a unique str identifier for a given LFSR() class instance
        - self.N: integer order of feedback polynomial & recurrence relation

        INPUTS & OUTPUTS:
        :param self: this particular LFSR() class instance
        :type self: __main__.LFSR
        :param mask: feedback polynomial coefficients (register tap weights)
        :type mask: int
        :param seed: initial register state as an [base-10] integer
        :type seed: int
        :param order: register length / feedback polynomial deg. (in bits)
        :type order: int
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
        self.feedback_polynomial = mask # (toggled feedback mask)

        # Initialize shift register state vector with seed value provided
        self.seed = seed # Fixed, initial conditions of linear recurrence
        self.state = seed # Initialize LFSR state vector with seed value

        # Define the order of the feedback polynomial (state length)
        self.N = order # (units = bits)

        # Epoch index initialization @ zero
        self.epoch = 0

    ##################################################
    # Pretty-print bitstring of assumed order self.N #
    ##################################################

    def pretty_print(self, b):
        """
        DESCRIPTION:
        This method pretty-prints the binary string representation of the
        binary vector, removing the '0b' prefix and padding to bit vector
        length / order self.N. This method has 1 arg & returns 1 value.

        INPUTS & OUTPUTS:
        :param self: this particular LFSR() class instance
        :type self: __main__.LFSR
        :param b: binary string representation of format '0b...'
        :type b: str
        :returns: binary string lacking '0b' prefix (e.g., '11010100')
        :rtype: str
        """
        # Return length-self.N bitstring (assumes printing external)
        return bin(b).replace('0b', '').zfill(self.N)

    ######################################################
    # Iterate LFSR recurrence for fixed number of epochs #
    ######################################################

    def recurse(self, num_epoch=1):
        """
        DESCRIPTION:
        This function executes the linear feedback shift register (LFSR)
        recursion for a fixed number of time epochs (default, 1 epoch).
        A global self.epoch counter is incremented, the register stored
        in self.state is updated via recursion, and nothing is returned.

        INPUTS & OUTPUTS:
        :param self: this particular LFSR() class instance
        :type self: __main__.LFSR
        :param num_epoch: number of epochs/cycles to recurse LFSR by
        :type num_epoch: int
        :returns: nothing is returned by this method
        :rtype: None
        """
        # Increment total epoch count/track
        self.epoch += num_epoch

        # For each epoch...
        for _ in range(num_epoch):

            # Least significant bit
            LSB = self.state & 0x1

            # Shift register to right
            self.state >>= 1

            # If LSB is 1...
            if LSB:

                # X0R register with feedback polynomial mask
                self.state ^= self.feedback_polynomial

    #######################################
    # Print LFSR variables summary/digest #
    #######################################

    def summary(self):
        """
        DESCRIPTION:
        This method pretty-prints a summary/digest of all of the member
        variables of this LFSR class instance (nothing is returned).

        INPUTS & OUTPUTS:
        :param self: this particular LFSR() class instance
        :type self: __main__.LFSR
        :returns: nothing is returned by this method
        :rtype: None
        """
        # Print header/banner above LFSR summary
        print('##################################################')
        print('# LINEAR FEEDBACK SHIFT REGISTER (LFSR) SUMMARY: #')
        print('##################################################')

        # Separator
        print('__________________________________________________')

        # Print register ID
        print('ID:', self.ID)

        # Separator
        print('__________________________________________________')

        # Current recurrence epoch/index
        print('EPOCH:', self.epoch)

        # Separate register ID from register order
        print('__________________________________________________')

        # Recurrence order
        print('ORDER:', self.N)

        # Separator
        print('__________________________________________________')

        # Print shift register tap polynomial (binary)
        print('TAPS (BINARY):', self.pretty_print(self.feedback_polynomial))

        # Print shift register tap polynomial (int format)
        print('TAPS (DECIMAL):', self.feedback_polynomial)

        # Print shift register state polynomial (hexadecimal)
        print('TAPS (HEXADECIMAL):', hex(self.feedback_polynomial))

        # Separator
        print('__________________________________________________')

        # Print shift register seed vector (binary)
        print('SEED (BINARY):', self.pretty_print(self.seed))

        # Print seed state vector (as integer)
        print('SEED (DECIMAL): ', self.seed)

        # Print shift register seed vector (hexadecimal)
        print('SEED (HEXADECIMAL):', hex(self.seed))

        # Separator
        print('__________________________________________________')

        # Print shift register state vector (binary)
        print('STATE (BINARY):', self.pretty_print(self.state))

        # Print shift register state vector (int format)
        print('STATE (DECIMAL):', self.state)

        # Print shift register state vector (hexadecimal)
        print('STATE (HEXADECIMAL):', hex(self.state))

        # Separator
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
        to the shifting of the register bits to the right, followed
        by the XOR'ing of the shifted register atate with the feed-
        back polynomial mask IF the least significant bit (LSB) is 1.
        This is repeated for each epoch, with state-printing optional.

        INPUTS & OUTPUTS:
        :param self: this particular LFSR() class instance
        :type self: __main__.LFSR
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
                print(self.pretty_print(self))

    #####################################################
    # Stream bit(s) from linear feedback shift register #
    #####################################################

    def stream(self, num_bits=1, stream_type='array', tap=0x1):
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
        :param self: this particular LFSR() class instance
        :type self: __main__.LFSR
        :param num_bits: # of epochs <==> # of bits output
        :type num_bits: int
        :param stream_type: specify output as either 'str' or 'array'
        :type num_bits: str
        :param tap: mask to logical AND w/ to tap register for stream
        :type tap: int
        :returns: nothing is returned by this method
        :rtype: None
        """
        ###################
        # Validate inputs #
        ###################

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

            # Return bit string...
            if stream_type == 'str':

                # Return tapped state vector bit (as str)
                return str(int((self.state & tap) > 0))

            # Return bit array...
            if stream_type == 'array':

                # Return tapped state vector bit (as int)
                return int((self.state & tap) > 0)

        ###########################################
        # Stream & buffer multiple bits from LFSR #
        ###########################################

        # Return binary string...
        if stream_type == 'str':

            # Binary stream storage
            buffer = ''

            # For each n'th bit required...
            for n in range(num_bits):

                # Cycle LFSR 1x (i.e., single epoch)
                self.recurse() # (default: num_epoch=1)

                # Return tapped state vector bit (as int)
                buffer += self.pretty_print(self.state)

            # Return buffered bit stream
            return buffer

        # Return binary array...
        if stream_type == 'array':

            # Binary stream storage
            buffer = zeros([num_bits,], dtype='uint64')

            # For each n'th bit required...
            for n in range(num_bits):

                # Cycle LFSR 1x (i.e., single epoch)
                self.recurse() # (default: num_epoch=1)

                # Return tapped state vector bit (as int)
                buffer[n] = int((self.state & tap) > 0) # (0 or 1)

            # Return buffered bit stream
            return buffer
