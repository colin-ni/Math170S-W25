'''
This performs linear regression in one dimension, i.e. the x-values are
all in R^1.
'''

import numpy as np

X = np.array([2.0, 3.3, 3.7, 2.0, 2.3, 2.7, 4.0, 3.7, 3.0, 2.3])
y = np.array([1.3, 3.3, 3.3, 2.0, 1.7, 3.0, 4.0, 3.0, 2.7, 3.0])

# append a column of ones to the LHS of X
X = np.concatenate([np.ones(shape=(len(X), 1)), np.expand_dims(X, 0).T], axis=1)

# compute the optimal parameters (X^T X)^{-1} X^T y
gamma_hat = np.linalg.inv(X.T @ X) @ X.T @ y.T

print(f'alpha_hat is {round(gamma_hat[0], 4)}')
print(f'beta_hat is {round(gamma_hat[1], 4)}')