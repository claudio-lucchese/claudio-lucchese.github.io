#!/usr/bin/python

# -----------------------------------------------------------------------------
# (C) 2010 Claudio Lucchese
# Logistic PCA
# This is a porting to python of Andrew I. Schein's matlab code
# from the paper "A Generalized Linear Model for Principal Component Analysis of Binary Data"
# -----------------------------------------------------------------------------

### The function name 'wbias' is to indicate that this code includes
### a bias term.  The code differs from the manuscript 
### presentation in that the bias vector is encoded as an extra 
### (L+1) row in the V matrix. 

import random, numpy

def lpca_fitwbias(X,L,numiterations=0,threshold=0.5):
    ## INPUTS:
    ## X: the binary data matrix, rows are observations
    ## L: the size of the desired latent space, e.g. 5
    ## numiterations: the maximum number of iterations
    ## when set to 0, we use a threshold change in LL instead
    ## to determine when to stop

    ## OUTPUTS:
    ## U: the coordinates in the latent space
    ## V: the basis of the latent space
    ## D: the bias vector

    random.seed()

    # Initialization
    N, D = X.shape     # Get the dimensions of the data
    U = numpy.array([[random.random() for j in range(L+1)] for i in range(N)])  # ...so we can create U and V matrices
    V = numpy.array([[random.random() for j in range(D)]   for i in range(L+1)]) 
  
    U[:,L] = 1.0          # This is for the bias vector.
   
    oldU = U;
    oldV = V;

    iter = 0;           # Used to count number of iterations in loop.
    changel = 1000.0;   # keep track of the change in LL between iterations;
    oldLL = 0.0;        # keep track of the last LL (init to -Infinity)
    stop = 0;           # Conditions for stopping the model-fitting

  
    i0 = [(i,j) for i in range(N) for j in range(D) if X[i,j]==0] 
    i1 = [(i,j) for i in range(N) for j in range(D) if X[i,j]==1]
 
 
    # Get a starting log likelilood.
    z = numpy.dot(U,V)
    oldLL = -sum( [numpy.log(1+numpy.exp(-z[i])) for i in i0] ) \
            -sum( [numpy.log(1+numpy.exp( z[i])) for i in i1] )
    print 'Iterations \t log_likelihood'
    print iter,'\t',oldLL

    # The loop
    while stop==0:
        iter += 1;
 
        # Begin U Update
        z = numpy.dot(U,V)
        T = numpy.tanh(z/2) /z;
        b = numpy.dot( (2*X-1), V[0:L,:].T ) - \
            numpy.dot(T * numpy.tile(V[L,:],(N,1)), V[0:L,:].T ) 
      
        for n in range(N):
            A = numpy.dot( numpy.tile(T[n,:],(L,1)) *  V[0:L,:], V[0:L,:].T )
            U[n,0:L] = numpy.linalg.solve(A, b[n,:])

        # Begin V Update
        z = numpy.dot(U,V)
        T = numpy.tanh(z/2) /z;
        b = numpy.dot( 2*X.T-1, U )
        for d in range(D):
            A = numpy.dot( (numpy.tile(T[:,d],(L+1,1)).T * U).T, U)
            V[:,d] = numpy.linalg.solve(A, b[d,:])

        # Report the Log Likelihood
        z = numpy.dot(U,V)
        newLL = -sum( [numpy.log(1+numpy.exp(-z[i])) for i in i0] ) \
                -sum( [numpy.log(1+numpy.exp( z[i])) for i in i1] )
        change = newLL-oldLL
        oldLL = newLL;
        print 'Iterations \t log_likelihood'
        print iter,'\t',oldLL

        # Run some checks to see if we go through the loop again.
        if numiterations>0:
            if iter>=numiterations:
                stop = 1
        elif change<threshold:
            stop = 1

        # Catch numerical errors
        if numpy.isnan(newLL):
            stop = 1;
            U = oldU;
            V = oldV;
        elif stop == 0:
            oldU = U;
            oldV = V;

    return U[:,:-1],V[:-1,:], V[-1,:]


def lpca_fit(X,L,numiterations=0,threshold=0.5):
    ## INPUTS:
    ## X: the binary data matrix, rows are observations
    ## L: the size of the desired latent space, e.g. 5
    ## numiterations: the maximum number of iterations
    ## when set to 0, we use a threshold change in LL instead
    ## to determine when to stop

    ## OUTPUTS:
    ## U: the coordinates in the latent space
    ## V: the basis of the latent space
    ## D: the bias vector

    random.seed()

    # Initialization
    N, D = X.shape     # Get the dimensions of the data
    U = numpy.array([[random.random() for j in range(L)] for i in range(N)])  # ...so we can create U and V matrices
    V = numpy.array([[random.random() for j in range(D)]   for i in range(L)]) 
  
    oldU = U;
    oldV = V;

    iter = 0;           # Used to count number of iterations in loop.
    changel = 1000.0;   # keep track of the change in LL between iterations;
    oldLL = 0.0;        # keep track of the last LL (init to -Infinity)
    stop = 0;           # Conditions for stopping the model-fitting

  
    i0 = [(i,j) for i in range(N) for j in range(D) if X[i,j]==0] 
    i1 = [(i,j) for i in range(N) for j in range(D) if X[i,j]==1]
 
 
    # Get a starting log likelilood.
    z = numpy.dot(U,V)
    oldLL = -sum( [numpy.log(1+numpy.exp(-z[i])) for i in i0] ) \
            -sum( [numpy.log(1+numpy.exp( z[i])) for i in i1] )
    print 'Iterations \t log_likelihood'
    print iter,'\t',oldLL

    # The loop
    while stop==0:
        iter += 1;
 
        # Begin U Update
        z = numpy.dot(U,V)
        T = numpy.tanh(z/2) /z;
        b = numpy.dot( (2*X-1), V.T )
        for n in range(N):
            A = numpy.dot( numpy.tile(T[n,:],(L,1)) *  V, V.T )
            U[n,:] = numpy.linalg.solve(A, b[n,:])

        # Begin V Update
        z = numpy.dot(U,V)
        T = numpy.tanh(z/2) /z;
        b = numpy.dot( 2*X.T-1, U )
        for d in range(D):
            A = numpy.dot( (numpy.tile(T[:,d],(L,1)).T * U).T, U)
            V[:,d] = numpy.linalg.solve(A, b[d,:])

        # Report the Log Likelihood
        z = numpy.dot(U,V)
        newLL = -sum( [numpy.log(1+numpy.exp(-z[i])) for i in i0] ) \
                -sum( [numpy.log(1+numpy.exp( z[i])) for i in i1] )
        change = newLL-oldLL
        oldLL = newLL;
        print 'Iterations \t log_likelihood'
        print iter,'\t',oldLL

        # Run some checks to see if we go through the loop again.
        if numiterations>0:
            if iter>=numiterations:
                stop = 1
        elif change<threshold:
            stop = 1

        # Catch numerical errors
        if numpy.isnan(newLL):
            stop = 1;
            U = oldU;
            V = oldV;
        elif stop == 0:
            oldU = U;
            oldV = V;

    return U,V

if __name__=="__main__":
    print "test"
    X = numpy.array( [  [1.,1.,1.,0.,0.,0.,0.], \
                        [1.,1.,1.,0.,0.,0.,0.], \
                        [1.,1.,1.,0.,0.,1.,1.], \
                        [1.,1.,1.,0.,0.,1.,1.], \
                        [0.,0.,0.,1.,1.,0.,0.], \
                        [0.,0.,0.,1.,1.,0.,0.]  ])
    
    #X = numpy.array( [  [1.,1.,1.,0.,0.], \
    #                    [1.,1.,1.,0.,0.], \
    #                    [1.,1.,1.,0.,0.], \
    #                    [1.,1.,1.,0.,0.], \
    #                    [0.,0.,0.,1.,1.], \
    #                    [0.,0.,0.,1.,1.]  ])


    #U,V,D = lpca_fit(X, 3, 10)
    #print U
    #print V
    #print D
    #
    #print numpy.dot(U,V)+numpy.tile(D, (6,1)) # 6 is the number of row in X
    #print numpy.sign(numpy.dot(U,V)+numpy.tile(D, (6,1)) ) # 6 is the number of row in X
    
    U,V = lpca_fit(X, 3, 10)
    print U
    print V
    
    print numpy.dot(U,V) # 6 is the number of row in X
    print numpy.sign(numpy.dot(U,V)) # 6 is the number of row in X
    