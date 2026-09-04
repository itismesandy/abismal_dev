import tensorflow as tf

INPUT_KEYS = (  
    "asu_id",
    "hkl",
    "resolution",
    "wavelength",
    "metadata",
    "iobs",
    "sigiobs",
)

def split_dataset_train_test(data, test_frac, seed=1234):
    """ Deterministically split data into fractions """
    train = data.enumerate().filter(
        lambda i,x: tf.random.stateless_uniform([1], (i+seed, i+seed))[0] > test_frac
    ).map(
        lambda i,x: x
    )
    test  = data.enumerate().filter(
        lambda i,x: tf.random.stateless_uniform([1], (i+seed, i+seed))[0] <= test_frac
    ).map(
        lambda i,x: x
    )
    return train, test

"""RECHECK THIS BEFORE PUSHING TO REPO"""
"""Converts input tuple into a dictionary object
        Expected Input: (asu_id, hkl_in/hkl, resolution, wavelength, metadata, iobs, sigiobs)
                        {"asu_id": asu_id, "hkl_in": hkl_in, ..., "sigiobs": sigiobs}
        Returns Dictionary: dictionary object
    """
def unpack_inputs(inputs):
    if not isinstance(inputs,(tuple,list)): #checks that input is tuple
        raise TypeError(...)
    
    if len(inputs) != len(INPUT_KEYS): #checks that output is the same length as input
        raise ValueError(
            f"expected {len(INPUT_KEYS)} elements but got {len(inputs)}"
        )
    return dict(zip(INPUT_KEYS, inputs))
