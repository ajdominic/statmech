import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import itertools as it


def create_lattices(dim=3):
    """
    This function creates a dim x dim spin lattice.
    Returns a tensor of dimension dim x dim x dim^2.
    """
    # total elements for each lattice and the possible spins
    n2 = dim * dim
    spins = ['1', '0']

    # total configuration number and the possible configurations
    configs = 2**n2
    possible_configs = ["".join(item) for item in it.product(spins, repeat=n2)]

    # populating the spin tensor
    spin_tensor = np.zeros([dim, dim, configs], float)
    for k in range(configs):
        config_temp = possible_configs[k]
        temp = [char for char in config_temp] # parses the characters
        array_temp = np.array(temp, float) # convert from list to array
        reshaped = np.reshape(array_temp, (3, 3)) # reshape the list
        spin_tensor[:, :, k] = np.where(reshaped == 0, -1, reshaped)

    return spin_tensor


def hammy(J, H, lattice):
    """
    This function computes the total energy
    of the given lattice.
    """
    dim = lattice.shape[0]
    nn_sum = 0.0
    for j in range(dim):
        for k in range(dim):
            nn_sum = nn(lattice, j, k)

    term1 = - J * nn_sum
    term2 = - H * np.sum(lattice)

    return term1 + term2


def nn(lat, r, c):
    # the number of rows/columns in the lattice
    dim = int(lat.shape[0])

    # the spin at a given index in the lattice
    spin = lat[r, c]

    # get the spins of the neighbors above and below
    up = lat[r-1, c]
    down = lat[(r+1)%dim, c]

    # get the spins of the right and left neighbors
    right = lat[r, (c+1)%dim]
    left = lat[r, c-1]

    # return the sum
    total = spin * (up + down + left + right)

    return total


def f1(theta, T, J, kb=1.0):
    """
    This function computes the integrand in subproject 2,
    which is then integrated later on using quadrature.
    """
    beta = 1.0 / (T * kb)
    k = 1.0 / (np.sinh(2.0 * beta * J))**2
    
    denom = (1 - 4 * k / (1 + k)**2 * (np.sin(theta))**2)
    integrand = np.sqrt(1.0 / denom)

    return integrand


def onsager(J, kb=1.0, lb=0.5, ub=50.0, dT=0.5):
    """
    This function computes the Onsager solution for
    an infinitely sized lattice.
    """
    # create the temperature array
    T_array = np.arange(lb, ub, dT)
    int_array = np.zeros_like(T_array, float) # computes the integral
    E_array = np.zeros_like(T_array, float) # Onsager solution
    for k in range(len(T_array)):
        T = T_array[k]
        beta = 1.0 / (kb * T)

        # integrate using quadrature
        val = integrate.quad(lambda x: f1(x, T, J), 0.0, np.pi / 2)
        int_array[k] = np.sum(val) # sum over the outputs

        # compute the rest of the terms in Onsager's solution
        temp2 = 2.0 * (np.tanh(2 * beta * J))**2 - 1
        temp3 = 1.0 + 2.0 * temp2 / np.pi * np.sum(val)
        E_array[k] = - J * 1 / np.tanh(2 * beta * J) * temp3

    return T_array, E_array


def plt_onsager(T, E):
    plt.figure(0, figsize=(7, 5))
    plt.plot(T, E, c="black", ls="-.", lw=3)
    plt.yticks(np.arange(-2.0, 0.1, 1.0), fontsize=14)
    plt.xticks(np.arange(0.0, 51.0, 25), fontsize=14)
    plt.xlabel("Temperature", fontsize=14)
    plt.ylabel("Internal energy per particle", fontsize=14)
    plt.savefig("onsager_exact.pdf")
    plt.show()


def is_in(item, array):
    """
    This function returns true if the given item is in the
    array and false if the given item is not in the array.
    """
    value = False
    ctr = 0
    while((value is False) and (ctr < array.shape[-1])):
        if (array[ctr] == item):
            value = True
        ctr += 1
    return value
