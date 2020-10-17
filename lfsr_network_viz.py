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
- num2cmap()
- lfsr_xor_connection()
- lfsr_shift_connection()
- init_input_nodes()
- init_hidden_nodes()
- init_decision_nodes()
- network_weight_coloration()
- lfsr_polynomial_wiring()
- lfsr_feedback_loop_wiring()
- draw_lfsr()
- draw_prediction_link()
- draw_layer0_linkages()
- draw_layer1_linkages()
- draw_network_wiring()
- init_network_diagram()
"""
###############################################################################
#                            Import dependencies                              #
###############################################################################

# Numpy functions
from numpy import max as np_max

# Import pyplot
from matplotlib import pyplot as plt

###############################################################################
#  Map numeric neural network weights to normalized colormap values in [0,1]  #
###############################################################################

def num2cmap(x, max_magnitude, cmap, epsilon=1e-9):
    """
    DESCRIPTION:
    This function accepts a float value x and remaps it to a colorscale speci-
    fied via colormap parameter cmap. This function also accepts a limit on
    the expected value |x|--max_magnitude--which aids in normalization of the
    value x to the valid matplotlib colorscale specified by cmap (in string ID
    format). A stability/damping paramter epsilon is also provided to prevent
    against division by zero during the normalization of scalar/vector x which
    is required for cmap(), which itself outputs a 4D color vector per element.
    Thus, an array which has 1 color vector per element of x is output.

    INPUTS & OUTPUTS:
    :param x: input value to be remapped to some colormap defined by cmap
    :type x: float
    :param max_magnitude: max. |x| which could be expected by this function
    :type max_magnitude: float
    :param cmap: some valid matplotlib colormap string ID
    :type cmap: str
    :param epsilon: let epsilon > 0 be small (prevents division by zero)
    :type epsilon: float
    :returns: color vector on the color scale specified, proportional to input
    :rtype: numpy.ndarray
    """
    return cmap(0.5 + x / (epsilon + 2 * max_magnitude))

###############################################################################
#  Diagram an XOR connection between nodes [n, n + 1] in a Galois-style LFSR  #
###############################################################################

def lfsr_xor_connection(n, y_register=0, lw=1, xor_size=8):
    """
    DESCRIPTION:
    This function draws the wiring between nodes / bits [n, n + 1] of a Galois
    style LFSR for the specific case where there is an XOR gate receiving
    feedback tapped from the output of the register. The XOR gate is impro-
    vised from matplotlib markers. The wiring of the register is rerouted from
    its baseline value to form symmetric XOR gate inputs--1 from the n'th bit,
    1 from the feedback from the register output. The output of the register
    is then routed (to the right) to where the (n + 1)'st bit is positioned.
    This function is defined not only with respect to the y-coordinate of the
    LFSR state vector, but also that of the feedback loop (below it). The
    register size and wiring width are also custom input parameters.

    INPUTS & OUTPUTS:
    :param n: XOR gate is at interval of register bits indexed [n, n + 1]
    :type n: int
    :param y_register: y-coordinate of the register along plot vertical axis
    :type y_register: float
    :param lw: line width of XOR wiring in plot/diagram (FWDed to matplotlib)
    :type lw: int
    :param xor_size: size of the XOR gate icon improvised from plot markers
    :type xor_size: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    #############
    # XOR input #
    #############

    # Wire y-offset into XOR gate
    dy = 0.005

    # Verticle register wiring offset into XOR gate
    plt.plot([n + 0.2, n + 0.2], [-0.1, y_register - dy], lw=1, c='k')

    # Vertically displaced [horiz.] wire feeding XOR gate via tap
    plt.plot([n + 0.2, n + 0.55], [y_register - dy, \
            y_register - dy], lw=lw, c='k')

    # Vertically displaced [horiz.] wire feeding XOR gate via register
    plt.plot([n, n + 0.55], [y_register + dy, y_register + dy], lw=1, c='k')

    ############
    # XOR gate #
    ############

    # XOR gate contribution from '|' marker
    plt.plot([n + 0.55], [y_register],
             '|', c='k', lw=lw, markersize=xor_size)

    # XOR gate contribution from '>' marker
    plt.plot([n + 0.67], [y_register],
             '>', c='k', lw=lw, markersize=(xor_size - 1))

    ##############
    # XOR output #
    ##############

    # Output wire from XOR gate (level w/ register bits)
    plt.plot([n + 0.6, n + 1], [y_register, y_register], c='k', lw=lw)

###############################################################################
#  Diagram a shift (>>) connection between bits [n, n + 1] in a Galois LFSR   #
###############################################################################

def lfsr_shift_connection(n, y_register=0, lw=1, arrow_size=8):
    """
    DESCRIPTION:
    This function draws the wire connection between linear feedback shift
    register (LFSR) bits [n, n + 1] at the y-coordinate y-register. The edge
    us drawn with line width specifed by input lw. A right-shift operator is
    centered between bits [n, n + 1] on the wire, with size controlled by
    input argument arrow_size. This function basically draws a directed edge.

    INPUTS & OUTPUTS:
    :param n: index of LFSR bits [n, n + 1] b/w which shift connection is made
    :type n: int
    :param y_register: the y-coordinate of the register state vector nodes
    :type y_register: float
    :param lw: width of the line/wiring/edge between state vector bits
    :type lw: int
    :param arrow_size: size of right-shift (>>) operator b/w bits [n, n + 1]
    :type arrow_size: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # Actual wiring between bits [n, n + 1]
    plt.plot([n, n + 1], [y_register, y_register], c='k', lw=lw)

    # Shift operator on wire b/w bits [n, n + 1]
    plt.plot([n + 0.5], [y_register], '4', c='k',
             lw=lw, markersize=arrow_size)

###############################################################################
#  Initialize the input nodes (aka. the LFSR state vector) of the neural net  #
###############################################################################

def init_input_nodes(deg, seed_bits, markersize=8):
    """
    DESCRIPTION:
    This function accepts a set of seed_bits--a binary vector of some fixed
    length deg--and draws the feedforward neural net's input nodes correspon-
    ding to this state vector. The nodes are drawn using filled 'o' markers
    via matplotlib, and double as the state vector of the underlying linear
    feedback shift register (LFSR) which defines the system the neural net is
    designed/trained to emulate. The size of the total deg nodes is controlled
    by parameter markersize. Node colors {'black': bit = 1, 'white': bit = 0}.

    INPUTS & OUTPUTS:
    :param deg: degree of primitive LFSR tap polynomial & length of state
    :type deg: int
    :param seed_bits: initial binary state vector used for LFSR recursion
    :type seed_bits: array-like (e.g., numpy.ndarray or list)
    :param markersize: size of the 'o' marker defining graph input nodes
    :type markersize: int
    :returns: list of plot objects/artists (plotting nodes of layer)
    :rtype: list, dtype=matplotlib.lines.Line2D
    """
    # Initialize storage for plotted nodes
    nodes = []

    # For each n'th bit...
    for n in range(deg):

        # Assign black node for bit=1, white for bit=0
        nth_node_color = 'k' if seed_bits[deg - n - 1] else 'w'

        ########################################
        # Plot LFSR n'th input node of network #
        ########################################

        # Plot the n'th node in the register seed state
        nodes.extend(plt.plot([n], [0], 'o',
                              markersize=markersize,
                              markeredgecolor='k',
                              markerfacecolor=nth_node_color))

    # Return plotted layer nodes
    return nodes,

###############################################################################
#  Initialize the hidden nodes (2nd layer) of the feedforward neural network  #
###############################################################################

def init_hidden_nodes(activations, colormap, num_hidden,
                      deg, y_layer=0.5, markersize=8):
    """
    DESCRIPTION:
    This function populates/initializes the hidden layer of a feedforward
    binary neural network estimating the future state of a deg-bit LFSR, de-
    fined by a deg-degree primitive polynomial over GF(2). This function is
    responsible for not only drawing the [num_hidden total] hidden layer nodes
    using 'o' marker with specified markersize at y-coordinate y_layer, but
    also for shading the nodes according to the specified colormap, propor-
    tional to the input activations provided to the function. The assumption
    is that the activation scalars correspond to those output by the inter-
    mediate layer of an already-trained neural network, and that these values
    correspond to some initial state vector used to shade the input layer.

    INPUTS & OUTPUTS:
    :param activations: model hidden layer activations for initial/seed state
    :type activations: numpy.ndarray, dtype=float
    :param colormap: valid string ID for a matplotlib colormap (e.g., 'jet')
    :type colormap: str
    :param num_hidden: # of hidden nodes in feedforward network drawn
    :type num_hidden: int
    :param deg: # of bits in input state vector, degree/order of feedback
    :type deg: int
    :param y_layer: y-coordinate of the hidden layer on the current axes
    :type y_layer: float
    :param markersize: size of hidden layer node 'o'-marker representations
    :type markersize: int
    :returns: list of plot objects/artists (plotting nodes of layer)
    :rtype: list, dtype=matplotlib.lines.Line2D
    """
    # Initialize storage for plotted nodes
    nodes = []

    # Define colormap via matplotlib
    cmap = plt.get_cmap(colormap)

    # For each n'th node...
    for n in range(num_hidden):

        # Plot hidden node @ coordinate (n,y_layer)
        nodes.extend(plt.plot([n * (deg / num_hidden) - 0.25], [y_layer],
                              'o', markersize=markersize,
                              markeredgecolor='k',
                              markerfacecolor=cmap(1 - activations[n])))

    # Return plotted layer nodes
    return nodes,

###############################################################################
#  Initialize the network output/activation node & decision/prediction node   #
###############################################################################

def init_decision_nodes(activations, colormap, num_output,
                        y_lower=0.8, y_upper=1, markersize=8):
    """
    DESCRIPTION:
    This function draws the final activation nodes of a feedforward neural
    network learning to emulate an LFSR. This function is also responsible for
    drawing the post-activation decision layer neurons, which depict thresh-
    olded activations in the form of binary values, the bits of the estimated
    LFSR state vector. (The network is assumed to predict LFSR state n + 1
    from LFSR state n for arbitrary n.) This function accepts the activation
    values of the final network layer corrsponding to the intial seed vector/
    state given to the network before recursion. Corresponding to these acti-
    vations, the function will also plot the predicted binary labels of the
    future LFSR state. This function assumes that drawing of the edge linking
    these nodes is handled externally by a function which wires all of the
    network's edges. The activation nodes are shaded according to the speci-
    fied colormap and have the specified 'o' markersize. This function per-
    forms a fliplr()-like operation on the bits--a mirror imaging, plotting
    from right-to-left what reads left-to-right. This decision was made in
    the interest of cohesion with other functions in this module. In total,
    this function places a total of num_output 'o' markers on the current
    axes at y-coordinate y-lower, and another num_output at y-coordinate
    y_upper, arranged to be equidistant along the x-interval [0,num_output].

    INPUTS & OUTPUTS:
    :param activations: feedforward net activations via to LFSR seed vector
    :type activations: numpy.ndarray
    :param colormap: valid string ID for a matplotlib colormap (e.g., 'jet')
    :type colormap: str
    :param num_output: number of output nodes (should be = to order of LFSR)
    :type num_output: int
    :param y_lower: lower activation layer node's y-coordinate
    :type y_lower: float
    :param y_upper: upper thresholded [decision] layer node's y-coordinate
    :type y_upper: float
    :param markersize: size of activation & thresholded decision nodes
    :type markersize: int
    :returns: 2 lists of plot objects/artists (plotting nodes of layer)
    :rtype: list, dtype=matplotlib.lines.Line2D (x2 outputs)
    """
    # Initialize plotted node storage
    output_layer_nodes = []
    decision_layer_nodes = []

    # Define colormap via matplotlib
    cmap = plt.get_cmap(colormap)

    # For each n'th output node pair...
    for n in range(num_output):

        # Sigmoidal class prediction node
        output_layer_nodes.extend(
            plt.plot(
                [n], [y_lower], 'o',
                markersize=markersize,
                markeredgecolor='k',
                markerfacecolor=cmap(1 - activations[num_output - n - 1])))

        # Binary decision (n'th future LFSR state bit)
        b = 1 if activations[num_output - n - 1] >= 0.5 else 0

        # Thresholded bit prediction node
        decision_layer_nodes.extend(
            plt.plot([n], [y_upper], 'o',
                     markersize=markersize,
                     markeredgecolor='k',
                     markerfacecolor=('k' if b else 'w')))

    # Return the node plots/artists
    return output_layer_nodes, decision_layer_nodes

###############################################################################
# Given a colormap & set of model weights, map model weights to color values  #
###############################################################################

def network_weight_coloration(model, colormap='twilight_shifted'):
    """
    DESCRIPTION:
    This function accepts the Keras model of a 2-layer binary (sigmoid) feed-
    forward neural network used to estimate the future state of an LFSR. Its
    task is to map the model's weights (2 layers, a hidden layer 0 and output
    layer 1) to color values in the range of the specified matplotlib color-
    map. The max modulus of the weight vectors across the entire network is
    used to normalize the weight values to a relative color scale with the
    help of the function num2cmap() defined above. Returned are layer 0 and 1
    weight vectors remapped to color vectors--and hence, provisioned with an
    additional depth-4 dimension. Both outputs are numpy arrays of floats.

    INPUTS & OUTPUTS:
    :param model: Keras feedforward binary net model used for LFSR prediction
    :type model: tensorflow.keras.models.Model
    :param colormap: string ID of a valid matplotlib colormap used for edges
    :type colormap: str
    :returns: 4D color mappings per weight of the net (x 2 layers/outputs)
    :rtype: numpy.ndarray (x2 outputs)
    """
    # Max. weight modulus across network
    max_weight = np_max([
        np_max(model.weights[0].numpy()), # Layer 0 (input)
        np_max(model.weights[1].numpy()) # Layer 1 (hidden)
    ])

    # Define colormap via matplotlib
    cmap = plt.get_cmap(colormap)

    # Color values for weights of Layer 0
    layer0_colors = num2cmap(
        model.weights[0].numpy(),
        max_magnitude=max_weight,
        cmap=cmap)

    # Color values for weights of Layer 1
    layer1_colors = num2cmap(
        model.weights[1].numpy(),
        max_magnitude=max_weight,
        cmap=cmap)

    # Return color arrays (w/ depth = 4)
    return layer0_colors, layer1_colors

###############################################################################
#  Draw register tap wiring/gates in the LFSR diagram beneath the neural net  #
###############################################################################

def lfsr_polynomial_wiring(deg, taps, y_register=0, lw=1,
                           arrow_size=8, xor_size=8):
    """
    DESCRIPTION:
    This function handles the plotting of wiring of a linear feedback shift
    register (LFSR) which is diagrammed by a subset of the functions in this
    module. Specifically, it handles the drawing of links between bits in the
    LFSR's register / state vector, representing these as either right-shift
    (>>) operators or XOR-shift. The XOR gates are drawn using matplotlib
    markers and assigned at indices for which the feedback polynomial speci-
    fied by the binary array taps equals 1. The line width, right-shift arrow,
    and XOR gate size are all customizable. This function plots all of these
    components for a total of deg bits in the register state at arbitrary y.

    INPUTS & OUTPUTS:
    :param deg: degree of primitive polynomial defining binary recurrence
    :type deg: int
    :param taps: binary primitive polynomial coeff. defining LFSR feedback
    :type taps: numpy.ndarray
    :param y_register: y-coordinate of the register bits on the current axes
    :type y_register: float
    :param lw: line width of the LFSR wiring drawn by this function
    :type lw: int
    :param arrow_size: arrow size of shift operators on LFSR wiring
    :type arrow_size: int
    :param xor_size: size of XOR gate made from matplotlib markers
    :type xor_size: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # For each n'th polynomial coeff...
    for n in range(deg - 1):

        #######################################
        # Decorate wiring b/w bits [n, n + 1] #
        #######################################

        # If nonzero n'th coefficient...
        if n > 0 and taps[n]:

            # Place X0R gate w/ feedback taps
            lfsr_xor_connection(n, y_register=y_register,
                                lw=lw, xor_size=xor_size)

        # n'th coefficient is 0...
        else:

            # Place ordinary shift operator
            lfsr_shift_connection(n, y_register=y_register,
                                  lw=lw, arrow_size=arrow_size)

###############################################################################
#  Draw the feedback loop wiring in the LFSR diagram beneath the neural net   #
###############################################################################

def lfsr_feedback_loop_wiring(x_midpt, deg, y_register=0,
                              y_feedback=-0.1, lw=1, arrow_size=8):
    """
    DESCRIPTION:
    This function handles the drawing/plotting of the feedback loop of a
    Galois-style LFSR diagrammed by a subset of the functions in this module.
    The register is assumed to have order deg, which is short for the 'degree'
    of the primitive polynomial over GF(2) defining the LFSR's maximum-length
    recursion. The register bits are assumed to be plotted at y-coordinate
    y_register and the feedback loop's wiring is plotted, correspondingly, at
    y-coordinate y_feedback. The line width of the feedback circuit, along
    with the size of the shift operators / arrows, as matplotlib parameters.
    The midpoint of the axes the diagram is plotted on is required for this
    function to execute. No values are returned by this function.

    INPUTS & OUTPUTS:
    :param x_midpt: the midpoint of the plotting window along the horiz. axis
    :type x_midpt: float
    :param deg: degree of the feedback polynomial / order of LFSR
    :type deg: int
    :param y_register: y-coordinate of the LFSR bits in the current plot
    :type y_register: float
    :param y_feedback: y-coordinate of the feedback loop wiring in the plot
    :type y_feedback: float
    :param lw: line width of the LFSR feedback loop wiring plotted
    :type lw: int
    :param arrow_size: arrow size of shift operators depicted on wiring
    :type arrow_size: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    ##############################################
    # Register I/O wiring (level with LFSR bits) #
    ##############################################

    # Input wire (level with register)
    plt.plot([0, -0.5], [y_register, y_register], c='k', lw=lw)

    # Shift operator decorating input wire
    plt.plot([-0.25], [y_register], '4', c='k',
             markersize=arrow_size)

    # Output wire (level with register)
    plt.plot([deg - 1, deg - 0.5],
             [y_register, y_register],
             c='k', lw=lw)

    ###################################
    # Descending feedback loop wiring #
    ###################################

    # LFSR output wire, descending into feedback loop
    plt.plot([deg - 0.5, deg - 0.5],
             [y_register, y_feedback],
             c='k', lw=lw)

    # Shift operator decorating feedback loop wire
    plt.plot([deg - 0.5], [(y_register + y_feedback) / 2],
             '1', c='k', markersize=arrow_size)

    ##################################
    # Ascending feedback loop wiring #
    ##################################

    # LFSR input wire, ascending out of feedback loop
    plt.plot([-0.5, -0.5], [y_register, y_feedback], c='k', lw=lw)

    # Shift operator decorating feedback loop wire
    plt.plot([-0.5], [(y_register + y_feedback) / 2],
             '2', c='k', markersize=arrow_size)

    ##############################
    # Lower feedback loop wiring #
    ##############################

    # Lower LFSR feedback loop wire (spans register length)
    plt.plot([-0.5, deg - 0.5], [y_feedback, y_feedback], c='k', lw=lw)

    # Shift operator decorating feedback loop wire
    plt.plot([x_midpt], [y_feedback], '3', c='k', markersize=arrow_size)

###############################################################################
#  Diagram a Galois-style linear feedback shift register (LFSR), given taps   #
###############################################################################

def draw_lfsr(taps, deg, x_midpt, y_register=0,
              y_feedback=-0.1, arrow_size=8, lw=1):
    """
    DESCRIPTION:
    This function uses matplotlib lines and markers to depict/diagram a Galois
    style linear feedback shift register (LFSR) centered around a total of deg
    bits, doubling as inputs to a feedforward network estimating/emulating the
    LFSR. The LFSR bits, along with the neural net tapping them, are all cen-
    tered around the x-coordinate x_midpt, and at y-coordinate y_register. The
    draw_lfsr() function splits the drawing of the register connections into 2
    steps: drawing of the shift connections and XOR gates of the internal reg-
    ister state vector, and the feedback loop external to the register, resp.
    The arrows on the LFSR wiring are given a markersize specified by input
    arrow_size, and the wires themselves have width specified by input arg lw.
    This function assumes the register bits (a.k.a., the feedforward network's
    input layer) have already been drawn at y-coordinat y_register by another
    function in this module. This function outputs no values.

    INPUTS & OUTPUTS:
    :param taps: binary primitive tap polynomial defining LFSR feedback
    :type taps: numpy.ndarray
    :param deg: degree/order of polynomial defining LFSR recurrence
    :type deg: int
    :param x_midpt: x-coordinate of current axes LFSR is centered on
    :type x_midpt: float
    :param y_register: y-coordinate of current axes LFSR is centered on
    :type y_register: float
    :param y_feedback: y-coordinate of LFSR feedback loop on current axes
    :type y_feedback: float
    :param arrow_size: size of shift operator arrows on LFSR wiring
    :type arrow_size: int
    :param lw: line width of LFSR wiring plotted by this function
    :type lw: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # Within-register wiring
    lfsr_polynomial_wiring(deg, taps, y_register=y_register,
                           lw=lw, arrow_size=arrow_size)

    # Feedback loop wiring
    lfsr_feedback_loop_wiring(x_midpt, deg=deg, y_register=y_register,
                              y_feedback=y_feedback, lw=lw,
                              arrow_size=arrow_size)

###############################################################################
# Draw the linkage between the network output node & predicted bit (decision) #
###############################################################################

def draw_prediction_link(num_output, y_lower=0.8, y_upper=1, lw=1):
    """
    DESCRIPTION:
    This function draws one link per output node of the neural network to a
    set of register bits predicted/estimated by the network, plotting these as
    black line segments of width lw between y-coordinates y_lower and y_upper.
    These num_output total connections represent the thresholding / decision
    process that maps binary activations to predicted future register bits.

    INPUTS & OUTPUTS:
    :param num_output: the number of outputs of the neural net & bits in LFSR
    :type num_output: int
    :param y_lower: y-coordinate of the output later of the neural network
    :type y_lower: float
    :param y_upper: y-coordinate of the output/predicted LFSR bits in plot
    :type y_upper: float
    :param lw: line width used for wiring connecting activations to ouput bits
    :type lw: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # For each n'th output node...
    for n in range(num_output):

        # Edge connecting sigmoid node to thresholded bit prediction
        plt.plot([n, n], [y_lower, y_upper], c='k', lw=lw)

###############################################################################
#    Draw weighted network edges of layer 0 of the feedforward neural net     #
###############################################################################

def draw_layer0_linkages(colors, num_input, num_hidden, y_input=0,
                         y_hidden=0.5, alpha=0.75, lw=1):
    """
    DESCRIPTION:
    This function plots the feedforward binary neural network wiring labeled
    layer 0, consisting of the connections from the num_input-length input
    layer to the num_hidden-length hidden layer. These nodes are positioned at
    y-coordinates y_input and y_hidden, respectively, spaced uniformly along
    the horizontal axis. These nodes are assumed to exist, and this function
    draws color-coded dense layer connections. The color scheme of the network
    connections maps the scalar weights of the network to color values on the
    scale of an arbitrary valid matplotlib colormap. This function allows for
    custom alpha/transparency and line width for the network edges drawn.

    INPUTS & OUTPUTS:
    :param colors: colormapped network weights (4D color vector per weight)
    :type colors: numpy.ndarray
    :param num_input: # of input layer nodes from which linkages are drawn
    :type num_input: int
    :param num_hidden: # of hidden layer nodes to which linkages are drawn
    :type num_hidden: int
    :param y_input: y-coordinate of the input layer nodes (i.e., LFSR bits)
    :type y_input: float
    :param y_hidden: y-coordinate of hidden layer nodes
    :type y_hidden: float
    :param alpha: transparency/alpha value of layer 0 weighted edges
    :type alpha: float
    :param lw: line width of edges in layer 0 of network
    :type lw: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # For each n'th node in input layer...
    for n in range(num_input):

        # For each m'th node in hidden layer...
        for m in range(num_hidden):

            # Draw link between n'th input & m'th hidden nodes
            plt.plot([n, m * (num_input / num_hidden) - 0.25],
                     [y_input, y_hidden], c=colors[n, m, :],
                     alpha=alpha, lw=lw)

###############################################################################
#    Draw weighted network edges of layer 1 of the feedforward neural net     #
###############################################################################

def draw_layer1_linkages(colors, num_hidden, num_output, y_hidden=0.5,
                         y_output=0.8, alpha=0.75, lw=1):
    """
    DESCRIPTION:
    This function plots the feedforward binary neural network wiring labeled
    layer 1, consisting of the connections from the num_hidden-length hidden
    layer to the num_output-length output layer. These nodes are positioned at
    y-coordinates y_hidden and y_output, respectively, spaced uniformly along
    the horizontal axis. These nodes are assumed to exist, and this function
    draws color-coded dense layer connections. The color scheme of the network
    connections maps the scalar weights of the network to color values on the
    scale of an arbitrary valid matplotlib colormap. This function allows for
    custom alpha/transparency and line width for the network edges drawn.

    INPUTS & OUTPUTS:
    :param colors: neural net layer 1 weights remapped to 4D color vectors
    :type colors: fkoat
    :param num_hidden: # of hidden layer nodes from which linkages are drawn
    :type num_hidden: int
    :param num_output: # of output layer nodes to which linkages are drawn
    :type num_output: int
    :param y_hidden: y-coordinates of hidden layer neuron 'o' markers
    :type y_hidden: int
    :param y_output: y-coordinates of output layer neuron 'o' markers
    :type y_output: int
    :param alpha: transparency/alpha value of layer 1 weighted edges
    :type alpha: float
    :param lw: line width of edges in layer 1 of network
    :type lw: int
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # For each n'th node in hidden layer...
    for n in range(num_hidden):

        # For each m'th node in output layer...
        for m in range(num_output):

            # Draw link between n'th input & m'th hidden nodes
            plt.plot([n * (num_output / num_hidden) - 0.25, m],
                     [y_hidden, y_output], c=colors[n, m, :],
                     alpha=alpha, lw=lw)

###############################################################################
# Draw the wiring of the 2-layer feedforward neural network learning the LFSR #
###############################################################################

def draw_network_wiring(model, config):
    """
    DESCRIPTION:
    This function draws the neural network's wiring / edges according to the
    parameters specified in the config panel. The model has 2 layers: layer 0
    consists of dense connections from the input nodes / LFSR state n to the
    hidden layer nodes; layer 1 consists of dense layer connections from the
    hidden layer to the output nodes--which in turn feed the predicted LFSR
    state n + 1 bits. While the former two sets of connections are weighted,
    the latter are simply decision connections, thresholding activations at
    0.5 to predict the output bit of the network. The thresholding connections
    are plotted in black by default; however, the connections of layers 0 and
    1 of the feedforward binary network are colored according to their scalar
    weight values. The color choice depends on the colormap string ID speci-
    fied in the config panel provided at input. Nothing is returned.

    INPUTS & OUTPUTS:
    :param model: feedforward binary neural network model for LFSR prediction
    :type model: tensorflow.keras.models.Model
    :param config: configuration panel controlling the LFSR + network viz.
    :type config: dict
    :returns: nothing (this function only plots things)
    :rtype: None
    """
    # Map neural network weights to color scale
    c_w0, c_w1 = network_weight_coloration(model, colormap=config['cmap'])

    # Plot network weighted edges from input layer to
    # hidden layer, mapping weight values to colors
    draw_layer0_linkages(c_w0, num_input=config['num_input'],
                         num_hidden=config['num_hidden'],
                         y_input=config['y_input'],
                         y_hidden=config['y_hidden'],
                         alpha=config['net_link_alpha'],
                         lw=config['link_width'] / 2)

    # Plot network weighted edges from hidden layer to
    # output layer, mapping weight values to colors
    draw_layer1_linkages(c_w1, num_hidden=config['num_hidden'],
                         num_output=config['num_output'],
                         y_hidden=config['y_hidden'],
                         y_output=config['y_output'],
                         alpha=config['net_link_alpha'],
                         lw=config['link_width'] / 2)

    # Edge connecting sigmoid node to thresholded bit prediction
    draw_prediction_link(num_output=config['num_output'],
                         y_lower=config['y_output'],
                         y_upper=config['y_decision'],
                         lw=config['link_width'])

###############################################################################
#  Initialize feedforward neural network diagram, populating nodes and edges  #
###############################################################################

def init_network_diagram(model, config, seed_bits,
                         h_activations, y_activations):
    """
    DESCRIPTION:
    This function initializes the feedforward binary/sigmoid neural network
    designed to emulate an LFSR, predicting its (n + 1)'st state vector bits
    from its n'th state vector bits--i.e., the input layer. First, the edges
    of the neural network are drawn. The nodes are then layered on top of
    them. The nodes are initialized with the seed_bits (at the input layer),
    the hidden layer activations, the output layer, and then finally the
    decision layer bits computed by thresholding the output layer. The role
    of this function is limited to initialization of the neural networkk with
    initial state values and activations. Plot animation is out of scope.

    INPUTS & OUTPUTS:
    :param model: feedforward binary neural network model for LFSR prediction
    :type model: tensorflow.keras.models.Model
    :param config: configuration panel controlling the LFSR + network viz.
    :type config: dict
    :param seed_bits: binary LFSR seed state vector (neural net input layer)
    :type seed_bits: array-like (numpy.ndarray or list)
    :param h_activations: hidden layer activations of the feedforward net
    :type h_activations: numpy.ndarray
    :param y_activations: output activations of the feedforward network
    :type y_activations: numpy.ndarray
    :returns: 4 lists of plot objects/artists (plotting nodes of network)
    :rtype: list, dtype=matplotlib.lines.Line2D (x4 outputs)
    """
    #########
    # Edges #
    #########

    # Draw the network connections/edges
    draw_network_wiring(model, config)

    #########
    # Nodes #
    #########

    # Input LFSR nodes / bits (deg total)
    input_nodes, = init_input_nodes(
        deg=config['deg'],
        seed_bits=seed_bits,
        markersize=config['node_size'])

    # Hidden layer nodes (deg total)
    hidden_nodes, = init_hidden_nodes(
        activations=h_activations,
        colormap=config['cmap'],
        num_hidden=config['num_hidden'],
        deg=config['deg'],
        y_layer=config['y_hidden'],
        markersize=config['node_size'])

    # Output probability + decision nodes
    output_nodes, decision_nodes, = init_decision_nodes(
        activations=y_activations,
        colormap=config['cmap'],
        num_output=config['num_output'],
        y_lower=config['y_output'],
        y_upper=config['y_decision'],
        markersize=config['node_size'])

    # Return network node plots/artists
    return input_nodes, hidden_nodes, output_nodes, decision_nodes,
