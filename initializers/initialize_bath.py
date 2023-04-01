import numpy as np

def initialize_wigner_bath(number_of_modes, w, beta):
    """
    Input:
    1. number_of_modes    -- number of oscillators in the environment surrounding the system
    2. w                  -- frequency of each oscillator
    3. beta               -- inverse of thermal energy

    Output:
    1. momentum_coord -- an array containing the initialized momentum coordinates
    2. position_coord -- an array containing the initialized position coordinates
    """

    # creating the momentum and position coordinates, all initially set to zero
    p = np.zeros(number_of_modes, float)
    q = np.zeros(number_of_modes, float)

    for i in range(number_of_modes):
        u = beta * w[i] / 2
        tanhu = np.tanh(u)

        p[i] = np.random.normal(0.0, (w[i] / (2 * tanhu))**0.5)
        q[i] = np.random.normal(0.0, (1 / (2 * w[i] * tanhu))**0.5)

    return p, q

def initialize_classical_bath(number_of_modes, w, beta):
    """

    Input:
    1. number_of_modes    -- number of oscillators in the environment surrounding the system
    2. w                  -- frequency of each oscillator
    3. beta               -- inverse of thermal energy

    Output:
    1. momentum_coord -- an array containing the initialized momentum coordinates
    2. position_coord -- an array containing the initialized position coordinates
    """

    # creating the momentum and position coordinates, all initially set to zero
    p = np.zeros(number_of_modes, float)
    q = np.zeros(number_of_modes, float)

    for i in range(number_of_modes):
        p[i] = np.random.normal(0.0, (1 / beta)**0.5)
        q[i] = np.random.normal(0.0, (1 / (w[i]**2 * beta))**0.5)

    return p, q
