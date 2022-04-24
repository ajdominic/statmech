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
