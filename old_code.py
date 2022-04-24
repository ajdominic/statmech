dim = 3
spins = ['1', '0'] 
possible_configs = ["".join(item) for item in it.product(spins, repeat=9)]
spin_tensor = np.zeros([dim, dim, configs])
for k in range(configs):
    config_temp = possible_configs[k] 
    temp = [char for char in config_temp]
    array_temp = np.array(temp, float)
    reshaped = np.reshape(array_temp, (3, 3))
    spin_tensor[:, :, k] = np.where(reshaped == 0, -1, reshaped)

def list_to_array(some_list, dim):
    length = len(some_list)
    temp = []
    for i in some_list:
        i.split()  
    array = np.zeros()


def nn(lat, r, c):
    """
    This function calculates the nearest neighbor sum
    in the Hamiltonian.
    """
    # the number of rows/columns in the lattice
    dim = int(lat.shape[0])

    # the spin at a given index in the lattice
    spin = lat[r, c]

    # get the spins of the neighbors above and below
    up = lat[(r-1)%dim, c]
    down = lat[(r+1)%dim, c]

    # get the spins of the right and left neighbors
    right = lat[r, (c+1)%dim]
    left = lat[r, (c-1)%dim]

    # return the sum
    total = spin * (up + down + left + right)

    return total

"""
J, H = 1.0, 0.0
lo_T, hi_T = 0.1, 1000.0
Q_lo_T = 0.0
Q_hi_T = 0.0
avg_lo = 0.0
avg_hi = 0.0
ham = np.zeros(configs, float)

# looping over each configuration
for k in range(configs):

    # get a particular lattice
    lattice = lattices[:, :, k]

    # calculate the energy
    ham[k] = c.hammy(J, H, lattice)

    # compute the Boltzman factors
    lo_exp = np.exp(- ham / lo_T)
    hi_exp = np.exp(- ham / hi_T)

    # compute the average energies
    avg_lo += ham * lo_exp
    avg_hi += ham * hi_exp

    # compute the partition functions
    Q_lo_T += lo_exp
    Q_hi_T += hi_exp

avg_lo /= Q_lo_T
avg_hi /= Q_hi_T
"""
