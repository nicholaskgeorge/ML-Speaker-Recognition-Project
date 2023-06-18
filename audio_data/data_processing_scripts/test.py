import numpy as np

# Initialize an empty matrix
matrix = np.empty((0, 3))  # Empty matrix with 0 rows and 3 columns

# Define vectors to append
vectors = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
    np.array([7, 8, 9])
]

# Append vectors to the matrix and print the shape at each step
print(matrix.shape)
for vector in vectors:
    print(vector.shape)
    print(matrix)
    matrix = np.vstack((matrix, vector))
    print("Vector shape:", vector.shape)
    print("Matrix shape:", matrix.shape)
    print()
print(matrix)

# Output:
# Vector shape: (3,)
# Matrix shape: (1, 3)
#
# Vector shape: (3,)
# Matrix shape: (2, 3)
#
# Vector shape: (3,)
# Matrix shape: (3, 3)