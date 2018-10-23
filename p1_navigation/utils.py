# Allows for dictionaries to be merged easily.
def merge(a,b):
    c = a.copy()
    c.update(b)
    return c