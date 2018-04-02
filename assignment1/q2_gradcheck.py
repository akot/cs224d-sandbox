import numpy as np
import random

# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    assert len(grad) == len(x)

    h = 1e-4
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later

        # save coordinate value
        x_i = x[ix]
        
        # "wiggle" coordinate value and compute function value.
        x[ix] = x_i + h
        random.setstate(rndstate)
        f_h1, _ = f(x)

        # "wiggle" coordinate value and compute function value.
        x[ix] = x_i - h
        random.setstate(rndstate)
        f_h2, _ = f(x)

        # restore coordinate value
        x[ix] = x_i

        # compute derivative of f 
        numgrad = (f_h1 - f_h2) / (2.*h)

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
 
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return
 
        it.iternext() # Step to next dimension

    print "Gradient check passed!"

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test
    print ""

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    # Identity function
    gradcheck_naive(lambda x: (np.sum(x), np.ones(x.shape)), np.array(123.456))      # scalar test
    gradcheck_naive(lambda x: (np.sum(x), np.ones(x.shape)), np.random.randn(3, 5))   

    # exp
    gradcheck_naive(lambda x: (np.sum(np.exp(x)), np.exp(x)), np.array(123.456))      # scalar test
    gradcheck_naive(lambda x: (np.sum(np.exp(x)), np.exp(x)), np.random.randn(3, 5)) 
    
    # sqrt
    gradcheck_naive(lambda x: (np.sum(np.sqrt(x)), 0.5/np.sqrt(x)), np.array(123.456))      # scalar test
    x = np.random.randn(3, 5)
    x += abs(np.min(x)) + 0.1
    gradcheck_naive(lambda x: (np.sum(np.sqrt(x)), 0.5/np.sqrt(x)), x) 
 
if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
