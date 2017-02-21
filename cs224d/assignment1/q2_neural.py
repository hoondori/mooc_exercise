import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    # data : N * Dx
    # W1   : Dx * H
    # b1   : 1 * H
    # W2   : H * Dy
    # b2   : 1 * Dy
    N = data.shape[0]

    z1 = data.dot(W1) + b1
    a1 = sigmoid(z1)  # N * H
    z2 = a1.dot(W2) + b2
    a2 = softmax(z2)  # N * Dy

    cost = np.sum(-np.log(a2[labels == 1])) / N

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    delta_score = a2 - labels  # 1 * Dy
    delta_score /= N

    gradW2 = np.dot(a1.T, delta_score)  # H * 1 * 1 * Dy = H * Dy
    gradb2 = np.sum(delta_score, axis=0)

    grad_h = np.dot(delta_score, W2.T)  # 1 * Dy * Dy * H = 1 * H
    grad_h = sigmoid_grad(a1) * grad_h

    gradW1 = np.dot(data.T, grad_h)
    gradb1 = np.sum(grad_h, axis=0)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    forward_backward_prop(data, labels, params, dimensions)

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

if __name__ == "__main__":
    sanity_check()