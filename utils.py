
import os
import random
import numpy as np
import torch



def seeding(seed):
    """ Seeding the randomness. """
    random.seed(seed)
    
    # Python randomizes the order of dictionaries and sets by default for security.
    # We fix this so variables are always stored/retrieved in the same order.
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    
    # Sets the seed for weight initialization and operations on the CPU.
    torch.manual_seed(seed)
    
    # Sets the seed for operations happening on the GPU.
    torch.cuda.manual_seed(seed)
    
    # Forces the GPU to use 'deterministic' algorithms. 
    torch.backends.cudnn.deterministic = True



def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)



def epoch_time(start_time, end_time):
    """ Calculate the time taken """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
