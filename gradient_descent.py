import numpy as np


def matrix_inner_product(A, x):
    """
    Computer the A inner product where A is some matrix and x is some vector. The equation is x_t * A * x

    Args:
        A: matrix
        x: vector
    Returns:
        x_t * A * x
    """
    return np.matmul(np.matmul(np.transpose(x), A), x)


def find_vector_norm(v):
    """
    Returns the infinity norm of a vector

    Args:
        v: n x 1 vector
    Returns:
        maximum of the absolute values of each entry in the vector
    """
    return np.max(np.absolute(v))


def gradient_descent(A, b):
    """
    Runs gradient descent until the residual is within a specified range

    Args:
        A: Matrix
        b: vector
    Returns:
        approximated value of u in the equation A * u = b
    """
    n = len(A)
    u = np.zeros(n)
    residual = b - np.matmul(A, u)
    residual_norm = find_vector_norm(residual)

    while residual_norm > .0001:
        alpha_k = (np.inner(residual, residual)) / (matrix_inner_product(A, residual))
        u += alpha_k * residual
        residual = b - np.matmul(A, u)
        residual_norm = find_vector_norm(residual)

    return u
