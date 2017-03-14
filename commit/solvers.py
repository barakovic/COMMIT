import numpy as np
import sys

eps = np.finfo(float).eps


def nnls( y, A, At = None, tol_fun = 1e-4, tol_x = 1e-9, max_iter = 100, verbose = 1, x0 = None ) :
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

    tol_x : double, optional (default: 1e-9).
        Minimum relative change of the solution x. The algorithm stops if:
               || x(t) - x(t-1) || / || x(t) || < tol_x,
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
    if verbose >= 1 :
        print "      |     ||Ax-y||     |  Cost function    Abs error      Rel error    |     Abs x          Rel x"
        print "------|------------------|-----------------------------------------------|------------------------------"

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
        abs_obj = np.abs(curr_obj - prev_obj)
        rel_obj = abs_obj / curr_obj
        abs_x   = np.linalg.norm(x - prev_x)
        rel_x   = abs_x / ( np.linalg.norm(x) + eps )
        if verbose >= 1 :
            print "  %13.7e  |  %13.7e  %13.7e  %13.7e  |  %13.7e  %13.7e" % ( np.sqrt(2.0*curr_obj), curr_obj, abs_obj, rel_obj, abs_x, rel_x )

        if abs_obj < eps :
            criterion = "ABS_OBJ"
            break
        elif rel_obj < tol_fun :
            criterion = "REL_OBJ"
            break
        elif abs_x < eps :
            criterion = "ABS_X"
            break
        elif rel_x < tol_x :
            criterion = "REL_X"
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
        print "< Stopping criterion: %s >" % criterion

    return x


def nntv_nnlasso( y, A, lambda_v1, lambda_v2, lenIC = None, Psit = None, Psi = None, At = None, tol_fun = 1e-5, tol_x = 1e-9, max_iter = 500, verbose = 1, x0 = None ) :
    """solve

          min  0.5*||y-A x||_2^2 + lambda1*||xIC||_{TV} + lambda2*||xEC||_1 s.t. x >= 0

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
    """

    # Initialization
    if At is None :
        At = A.T

    if x0 is not None :
        xhat = x0
        res = A.dot(xhat) - y
        grad = At.dot(res)
        fval = prox_tv_l1(xhat, 1, 1, lenIC, Psit, Psi)[1:3]
        prev_obj = 0.5 * np.linalg.norm(res)**2 + lambda_v1*fval[0] + lambda_v2*fval[1]
    else :
        res = -y
        grad = At.dot(res)
        xhat = np.zeros( grad.shape[0], dtype=np.float64 )
        prev_obj = 0.5 * np.linalg.norm( res )**2

    iter = 1
    prev_x = xhat
    told = 1
    beta = 0.9
    qfval = prev_obj

    # Step size computation
    L = np.linalg.norm( A.dot(grad) )**2 / np.linalg.norm(grad)**2
    mu = 1.9 / L

    if verbose >= 1 :
        print "      |     ||Ax-y||     |  Cost function    Abs error      Rel error    |     Abs x          Rel x"
        print "------|------------------|-----------------------------------------------|------------------------------"


    # Main loop
    while True :
        if verbose >= 1 :
            print "%4d  |" % iter,
            sys.stdout.flush()

        # Gradient descend step
        x = xhat - mu*grad

        # Prox non negative L1 norm
        x = np.maximum( x,0 )

        sol = prox_tv_l1( x, lambda_v1*mu, lambda_v2*mu, lenIC, Psit, Psi )

        # Stepsize check
        tmp = sol[0]-xhat
        q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2 + lambda_v1*sol[1] + lambda_v2*sol[2]
        res = A.dot(sol[0]) - y
        curr_obj = 0.5 * np.linalg.norm(res)**2 + lambda_v1*sol[1] + lambda_v2*sol[2]
        # Backtracking
        while curr_obj > q :
            # Gradient descend step
            mu = beta*mu
            x = xhat - mu*grad

            # Prox non negative L1 norm
            x = np.maximum( x,0 )

            sol = prox_tv_l1( x, lambda_v1*mu, lambda_v2*mu, lenIC, Psit, Psi )

            # New stepsize check
            tmp = sol[0]-xhat
            q = qfval + np.real( np.dot(tmp,grad) ) + 0.5/mu * np.linalg.norm(tmp)**2 + lambda_v1*sol[1] + lambda_v2*sol[2]
            res = A.dot(sol[0]) - y
            curr_obj = 0.5 * np.linalg.norm(res)**2 + lambda_v1*sol[1] + lambda_v2*sol[2]

        # Global stopping criterion
        abs_obj = np.abs(curr_obj - prev_obj)
        rel_obj = abs_obj / curr_obj
        abs_x   = np.linalg.norm(x - prev_x)
        rel_x   = abs_x / ( np.linalg.norm(x) + eps )
        if verbose >= 1 :
            print "  %13.7e  |  %13.7e  %13.7e  %13.7e  |  %13.7e  %13.7e" % ( np.sqrt(2.0*curr_obj), curr_obj, abs_obj, rel_obj, abs_x, rel_x )

        if abs_obj < eps :
            criterion = "ABS_OBJ"
            break
        elif rel_obj < tol_fun :
            criterion = "REL_OBJ"
            break
        elif abs_x < eps :
            criterion = "ABS_X"
            break
        elif rel_x < tol_x :
            criterion = "REL_X"
            break
        elif iter >= max_iter :
            criterion = "MAX_IT"
            break

        # FISTA update
        t = 0.5 * ( 1 + np.sqrt(1+4*told**2) )
        xhat = sol[0] + (told-1)/t * (sol[0] - prev_x)

        # Gradient computation
        res = A.dot(xhat) - y
        grad = At.dot(res)

        # Update variables
        iter += 1
        prev_obj = curr_obj
        prev_x = sol[0]
        told = t
        qfval = 0.5 * np.linalg.norm(res)**2
    if verbose >= 1 :
        print "< Stopping criterion: %s >" % criterion

    return sol[0]


def prox_tv_l1(x, lambda_v1, lambda_v2, lenIC = None, Psit = None, Psi = None, nu = 1, weights = 1, tight = 0, pos = 0, real = 0, tol_fun = 1e-4, max_iter = 200, verbose = 0) :
    """
    min_{z} 0.5*||x - z||_2^2 + lambda * ||Psit x||_1

    - Psit: Sparsifying transform (default: Id).

    - Psi: Adjoint of Psit (default: Id).

    - tight: 1 if Psit is a tight frame or 0 if not (default = 1)

    - nu: bound on the norm of the operator A, i.e.
    ||Psi x||^2 <= nu * ||x||^2 (default: 1)

    - max_iter: max. nb. of iterations (default: 200).

    - rel_obj: minimum relative change of the objective value (default:
   1e-4)
       The algorithm stops if
           | ||x(t)||_1 - ||x(t-1)||_1 | / ||x(t)||_1 < rel_obj,
       where x(t) is the estimate of the solution at iteration t.

    - verbose: 0 no log, 1 a summary at convergence, 2 print main
   steps (default: 1)

    - weights: weights for a weighted L1-norm (default = 1)


    References:
    [1] M.J. Fadili and J-L. Starck, "Monotone operator splitting for optimization problems in sparse recovery" , IEEE ICIP, Cairo, Egypt, 2009.
    [2] Amir Beck and Marc Teboulle, "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",  SIAM Journal on Imaging Sciences (2009), no. 1, 183--202.

    """

    sol = np.zeros( len(x), dtype=np.float64 )
    iter_l1 = 0

    if lenIC is None :
        lenIC = len(x)

    if Psi is None :
        Psit = np.identity(len(x[0:lenIC])).T
        Psi = np.identity(len(x[0:lenIC]))

    if tight and not(pos) :
        temp = Psit*x[0:lenIC]
        sol[0:lenIC] = x[0:lenIC] + 1/nu * Psi*(np.sign(temp)*np.maximum(np.abs(temp)-(lambda_v1*nu*weights),0)-temp)
        dummy = Psit*sol[0:lenIC]
        norm_l1 = np.sum(weights*np.abs(dummy))

    else :

        # Initializations
        u_l1 = np.zeros((np.dot(Psit,x[0:lenIC])).shape)
        sol[0:lenIC] = x[0:lenIC] - np.dot(Psi,u_l1)
        prev_l1 = 0

        # soft-thresholding
        # init
        if verbose >= 1 :
            print "  Proximal l1 operator:\n"

        while True :

            # L1 norm of the estimate
            dummy = np.dot(Psit,sol[0:lenIC])

            norm_l1 = .5*np.linalg.norm(x[0:lenIC]-sol[0:lenIC], 2)**2 + lambda_v1 * np.sum(weights*np.abs(dummy))
            rel_l1 = np.abs(norm_l1-prev_l1)/norm_l1

            # Log
            if verbose >= 1 :
                print "  Iter %d, ||Psit x||_1 = %8.4e, rel_l1 = %8.4e\n" % (iter_l1, norm_l1, rel_l1)

            if rel_l1 < tol_fun :
                criterion = "REL_OBJ"
                break
            elif iter_l1 >= max_iter :
                criterion = "MAX_IT"
                break

            # Soft-thresholding
            res = u_l1*nu + dummy
            dummy = np.sign(res)*np.maximum(res-(lambda_v1*nu*weights),0)
            u_l1 = 1/nu * (res - dummy)
            sol[0:lenIC] = x[0:lenIC] - np.dot(Psi,u_l1)

            if pos :
                sol[0:lenIC] = np.real(sol[0:lenIC])
                sol[0:lenIC] = np.maximum(sol[0:lenIC],0)

            if real :
                sol[0:lenIC] = np.real(sol[0:lenIC])

            # Update
            prev_l1 = norm_l1
            iter_l1 = iter_l1+1

        # Log after the projection onto the L2-ball
        if verbose >= 1 :
            print "  prox_L1: ||Psi x||_1 = %8.4e, %d\n" % (norm_l1, iter_l1)
    fval1 = norm_l1

    sol[lenIC:len(x)] = np.maximum(x[lenIC:len(x)] - lambda_v2, 0)
    fval2 = np.linalg.norm( x[lenIC:len(x)],1 )

    return sol, fval1, fval2
