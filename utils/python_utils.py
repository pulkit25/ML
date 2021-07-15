# Inverts key value pair mapping for a dictionary
def invert_dictionary(d):
    return {v:k for k,v in d.items()}