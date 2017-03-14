import numpy as np
import sys

def nnls( y, A, At = None, tol_fun = 1e-4, max_iter = 100, verbose = 1, x0 = None ) :
    """Solve non negative least squares problem of the following form:

       min 0.5*||y-A x||_2^2 s.t. x >= 0

    The problem is solved using the forward-backward algorithm with FISTA-like acceleration.

    Parameters
    ----------
    y : 1-d array of doubles.
        Contains the measurements.

    A : matrix or class exposing the .dot() method.
        It is the forward measurement operator.

    At : matrix or class exposing the .dot() method.
        It is the corresponding adjoint operator (default: computed as A.T).

    tol_fun : double, optional (default: 1e-4).
        Minimum relative change of the objective value. The algorithm stops if:
               | f(x(t)) - f(x(t-1)) | / f(x(t)) < tol_fun,
        where x(t) is the estimate of the solution at iteration t.

    max_iter : integer, optional (default: 100).
        Maximum number of iterations.

    verbose : integer, optional (default: 1).
        0 no log, 1 print each iteration results.

    x0 : 1-d array of double, optional (default: automatically computed).
        Initial solution.

    Returns
    -------
    x : 1-d array of doubles.
        Best solution in the least-squares sense.

    Notes
    -----
    Author: Rafael Carrillo
    E-mail: rafael.carrillo@epfl.ch
    """
    # Initialization
    if At is None :
        At = A.T

    if x0 is not None :
        xhat = x0
        res = A.dot(xhat) - y
    else :
        xhat = np.zeros( A.shape[1], dtype=np.float64 )
        res = -y
    grad = At.dot(res)
    prev_obj = 0.5 * np.linalg.norm(res)**2
    iter = 1
    told = 1
    prev_x = xhat
    beta = 0.9
    qfval = prev_obj

    # Step size computation
    L = np.linalg.norm( A.dot(grad) )**2 / np.linalg.norm(grad)**2
    mu = 1.9 / L

    # Main loop
    while True :
        if verbose >= 1 :
            print "%4d  |" % iter,
            sys.stdout.flush()

        # Gradient descend step
        x = xhat - mu*grad

        # Projection onto the positive orthant
        x = np.real( x )
        x[ x<0 ] = 0

        # Stepsize check
        tmp = x-xhat
        q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2
        res = A.dot(x) - y
        curr_obj = 0.5 * np.linalg.norm(res)**2

        # Backtracking
        while curr_obj > q :
            # Gradient descend step
            mu = beta*mu
            x = xhat - mu*grad

            # Projection onto the positive orthant
            x = np.real( x )
            x[ x<0 ] = 0

            # New stepsize check
            tmp = x-xhat
            q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2
            res = A.dot(x) - y
            curr_obj = 0.5 * np.linalg.norm(res)**2

        # Global stopping criterion
        rel_obj  = np.abs(curr_obj - prev_obj) / curr_obj
        if verbose >= 1 :
            print "  %13.7e  %13.7e" % ( np.sqrt(2.0*curr_obj), rel_obj )

        if rel_obj < tol_fun :
            criterion = "REL_OBJ"
            break
        elif iter >= max_iter :
            criterion = "MAX_IT"
            break

        # FISTA update
        t = 0.5 * ( 1 + np.sqrt(1+4*told**2) )
        xhat = x + (told-1)/t * (x - prev_x)

        # Gradient computation
        res = A.dot(xhat) - y
        grad = At.dot(res)

        # Update variables
        iter += 1
        prev_obj = curr_obj
        prev_x = x
        told = t
        qfval = 0.5 * np.linalg.norm(res)**2

    if verbose >= 1 :
        print "\t< Stopping criterion: %s >" % criterion

    return x


def nnglasso( y, A, lambda_v, indexes, w, At = None, tol_fun = 1e-4, max_iter = 100, verbose = 1, x0 = None ) :
    """solve_nnglasso - Solve the non negative group lasso problem

       sol = solve_nnlasso(y, A, At, PARAM) solves:

          min  0.5*||y-A x||_2^2 + lambda*||x||_{2,1} s.t. x >= 0

    Y contains the measurements. A is the forward measurement operator and At the associated
    adjoint operator.
    Parameters
    ----------
    y : 1-d array of doubles.
        Contains the measurements.
    A : matrix or class exposing the .dot() method.
        It is the forward measurement operator.
    At : matrix or class exposing the .dot() method.
        It is the corresponding adjoint operator (default: computed as A.T).
    tol_fun : double, optional (default: 1e-4).
        Minimum relative change of the objective value. The algorithm stops if:
               | f(x(t)) - f(x(t-1)) | / f(x(t)) < tol_fun,
        where x(t) is the estimate of the solution at iteration t.
    max_iter : integer, optional (default: 100).
        Maximum number of iterations.
    verbose : integer, optional (default: 1).
        0 no log, 1 print each iteration results.
    x0 : 1-d array of double, optional (default: automatically computed).
        Initial solution.
    Returns
    -------
    x : 1-d array of doubles.
        Best solution in the least-squares sense.
    Notes
    -----
    Author: Rafael Carrillo
    E-mail: rafael.carrillo@epfl.ch
    """
    ng = len(indexes)-1

    # Initialization
    if At is None :
        At = A.T

    if x0 is not None :
        xhat = x0
        res = A.dot(xhat) - y
        grad = At.dot(res)
        fval = l2l1_prox( x, 1, w, ng, indexes )[1]
        prev_obj = 0.5 * np.linalg.norm(res)**2 + lambda_v*fval
    else :
        res = -y
        grad = At.dot(res)
        xhat = np.zeros( grad.shape[0], dtype=np.float64 )
        prev_obj = 0.5 * np.linalg.norm( res )**2

    iteration = 1
    prev_x = xhat
    told = 1
    beta = 0.9
    qfval = prev_obj

    # Step size computation
    L = np.linalg.norm( A.dot(grad) )**2 / np.linalg.norm(grad)**2
    mu = 1.9 / L

    # Main loop
    while True :
        if verbose >= 1 :
            print "%4d  |" % iteration,
            sys.stdout.flush()

        # Gradient descend step
        x = xhat - mu*grad

        # Prox non negative L1 norm
        x = np.maximum( x,0 )

        sol = l2l1_prox( x, lambda_v*mu, w, ng, indexes )

        # Stepsize check
        tmp = sol[0]-xhat
        q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2 + lambda_v*sol[1]
        res = A.dot(sol[0]) - y
        curr_obj = 0.5 * np.linalg.norm(res)**2 + lambda_v*sol[1]
        # Backtracking
        while curr_obj > q :
            # Gradient descend step
            mu = beta*mu
            x = xhat - mu*grad

            # Prox non negative L1 norm
            x = np.maximum( x,0 )

            sol = l2l1_prox( x, lambda_v*mu, w, ng, indexes )

            # New stepsize check
            tmp = sol[0]-xhat
            q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2 + lambda_v*sol[1]
            res = A.dot(sol[0]) - y
            curr_obj = 0.5 * np.linalg.norm(res)**2 + lambda_v*sol[1]

        # Global stopping criterion
        rel_obj  = np.abs(curr_obj - prev_obj) / curr_obj
        if verbose >= 1 :
            print "  %13.7e  %13.7e" % ( np.sqrt(2.0*curr_obj), rel_obj )

        if rel_obj < tol_fun :
            criterion = "REL_OBJ"
            break
        elif iteration >= max_iter :
            criterion = "MAX_IT"
            break

        # FISTA update
        t = 0.5 * ( 1 + np.sqrt(1+4*told**2) )
        xhat = sol[0] + (told-1)/t * (sol[0] - prev_x)

        # Gradient computation
        res = A.dot(xhat) - y
        grad = At.dot(res)

        # Update variables
        iteration += 1
        prev_obj = curr_obj
        prev_x = sol[0]
        told = t
        qfval = 0.5 * np.linalg.norm(res)**2

    if verbose >= 1 :
        print "\t< Stopping criterion: %s >" % criterion

    return sol[0]


def l2l1_prox( x, lambda_v, w, NG, indexes ) :
    """L2-L1 proximal operator (group sparsity)

    Inputs:

    x : input vector

    lambda : regularization parameter (multiplicative constant)

    w : weights per group (size NG)

    NG : number of groups

    indexes : array containing the start and end indexes of every group

    """
    sol = np.zeros( x.shape[0], dtype=np.float64 )
    T = np.zeros( w.shape[0], dtype=np.float64 )

    for k in range (0, NG) :

        # Pointers
        pointer1 = indexes[k]
        pointer2 = indexes[k+1]

        # Compute L2 norm of the group
        normvec = np.linalg.norm(x[pointer1:pointer2])

        # Soft thresholding of the norm
        T[k] = np.maximum(normvec - lambda_v * w[k],0)

        # Avoid division by zero
        if normvec == 0 :
            normvec = np.spacing(1)

        # Scaling
        sol[pointer1:pointer2] = np.multiply(x[pointer1:pointer2], T[k]) / normvec

    fval = np.dot( w, T )

    return sol, fval
