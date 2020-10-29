# -*- coding: utf-8 -*-
"""
A Ng Machine Learning Week1 exercise.

Convert Matlab code to Python.
Created on Wed Sep 16 10:49:13 2020
@author: yooji
"""

import numpy as np

# % The ; denotes we are going back to a new row.
# A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]
a_data = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12]]
A = np.array(a_data)

# % Initialize a vector
# v = [1;2;3]
v_data = [1, 2, 3]
v = np.array(v_data)

# % Get the dimension of the matrix A where m = rows and n = columns
# [m,n] = size(A)
m, n = A.shape

# % You could also store it this way
# dim_A = size(A)

dim_A = A.shape

# % Get the dimension of the vector v
# dim_v = size(v)
dim_v = v.shape

# % Now let's index into the 2nd row 3rd column of matrix A
# A_23 = A(2,3)
A_12 = A[1, 2]

# %%
# % Initialize matrix A and B
# A = [1, 2, 4; 5, 3, 2]
# B = [1, 3, 4; 1, 1, 1]
A = np.array([[1, 2, 4],
              [5, 3, 2]])
B = np.array([[1, 3, 4],
              [1, 1, 1]])

# % Initialize constant s
s = 2

# % See how element-wise addition works
add_AB = A + B

# % See how element-wise subtraction works
sub_AB = A - B

# % See how scalar multiplication works
mult_As = A * s

# % Divide A by s
div_As = A / s

# % What happens if we have a Matrix + scalar?
add_As = A + s

# %%
# % Initialize matrix A
# A = [1, 2, 3; 4, 5, 6;7, 8, 9]
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# % Initialize vector v
# v = [1; 1; 1]
v = np.array([1, 1, 1])

# % Multiply A * v
Av = A.dot(v)

# %%
# % Initialize a 3 by 2 matrix
# A = [1, 2; 3, 4;5, 6]
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# % Initialize a 2 by 1 matrix
B = np.array([1, 2])

# % We expect a resulting matrix of (3 by 2)*(2 by 1) = (3 by 1)
mult_AB = A.dot(B)

# % Make sure you understand why we got that result

# %%
# % Initialize random matrices A and B
# A = [1,2;4,5]
# B = [1,1;0,2]
A = np.array([[1, 2],
              [4, 5]])
B = np.array([[1, 1],
              [0, 2]])

# % Initialize a 2 by 2 identity matrix
# I = eye(2)
I = np.identity(2)

# % The above notation is the same as I = [1,0;0,1]

# % What happens when we multiply I*A ?
# IA = I*A
IA = I.dot(A)

# % How about A*I ?
# AI = A*I
AI = A.dot(I)

# % Compute A*B
# AB = A*B
AB = A.dot(B)

# % Is it equal to B*A?
# BA = B*A
BA = B.dot(A)

# % Note that IA = AI but AB != BA

# %%
# % Initialize matrix A
# A = [1,2,0;0,5,6;7,0,9]
A = np.array([[1, 2, 0],
              [0, 5, 6],
              [7, 0, 9]])

# % Transpose A
A_trans = np.transpose(A)

# % Take the inverse of A
# A_inv = inv(A)
A_inv = np.linalg.inv(A)

# % What is A^(-1)*A?
A_invA = A_inv.dot(A)
