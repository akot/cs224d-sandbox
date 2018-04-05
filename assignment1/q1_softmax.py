import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.
    """
    if 1 == len(x.shape):
        x = x.reshape(1, x.shape[0])

    x = np.exp(x - np.max(x, axis=1).reshape(x.shape[0], 1)) 
    return x / np.sum(x, axis=1).reshape(x.shape[0], 1)

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    test1 = softmax(np.array([1000,1]))
    assert np.amax(np.fabs(test1 - np.array(
        [1.0, 0.]))) <= 1e-6
    
    test2 = softmax(np.array([[3, 3, 7, 7], [1, 3, 5, 7]]))
    assert np.amax(np.fabs(test2 - np.array(
        [[0.0089931,  0.0089931,  0.4910069,  0.4910069], [0.00214401,  0.0158422 ,  0.11705891,  0.86495488]]))) <= 1e-6
 

if __name__ == "__main__":
    ### END YOUR CODE
    test_softmax_basic()
    test_softmax()
