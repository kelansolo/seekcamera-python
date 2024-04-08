import numpy as np

def get_slice(M,r):
    slice = []
    for i, temperature_image in enumerate(M):
        slice.append(temperature_image[r])
    return(np.array(slice))