# Format for the binary of datasets (which is the same that it is followed when saving and loading Tensors)
The header of the binary file contains integers (4 bytes) in the following structure: [TYPE_OF_DATA, num_dimensions, dim1, dim2 , ..., dim_N].  The value for the TYPE_OF_DATA can be one of the following: CHAR (0), INTEGER (1), FLOAT (2). 

# Example: DIGITS
The binary for DIGITS (images) looks like the following: 0 4 1797 1 8 8 ... CHAR DATA ...

The binary for DIGITS (labels) looks like the following: 1 2 1797 1 ... INT DATA ...