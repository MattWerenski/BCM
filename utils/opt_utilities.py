'''
Function for solving the optimization problem: 

min    x^TAx
s.t.   x_i >= 0 for all i = 1,...,p  
       x_1 + ... + x_p = 1

which is essential to our method, as well as for finding the optimal
linear projection of a digit onto the convex hull of the references
'''

import numpy as np
import cvxopt

def solve(inner_products, return_val=False):
    '''
    opt_utilities.solve
        Actually solves the minimization procedure we've defined, given the 
        evaluation of the inner products in the tangent space.
        
    parameters
        inner_products - p by p matrix of inner products in the tangent space.

        return_val - whether or not to return the value of the objective

    returns
        the optimal mixture weight, and value if return_val = True
    '''

    p = inner_products.shape[0]
    
    P = cvxopt.matrix(inner_products)
    q = cvxopt.matrix(np.zeros(p))
    G = cvxopt.matrix(-np.eye(p))
    h = cvxopt.matrix(np.zeros(p))
    A = cvxopt.matrix(np.ones((1,p)))
    b = cvxopt.matrix(np.ones((1,1)))

    cvxopt.solvers.options['show_progress'] = False
    
    soln = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    lam = np.squeeze(np.array(soln['x']))
    
    if return_val:
        return [lam, soln['primal objective']]
    return lam

    

def linear_projection(source, points):
    '''
    opt_utilities.linear_projection
        Projects the source vector onto the convex hull of the points
        
    parameters
        source - np.array of length k

        points - np.array of shape (nrefs, k)

    returns
        the projection of the source onto conv_hull(points)
    '''
    
    p = points.shape[0]
    P = cvxopt.matrix(points @ points.T)
    G = cvxopt.matrix(-np.eye(p))
    h = cvxopt.matrix(np.zeros(p))
    q = cvxopt.matrix(-source @ points.T)
    A = cvxopt.matrix(np.ones((1,p)))
    b = cvxopt.matrix(np.ones((1,1)))
    soln = cvxopt.solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    lam = np.squeeze(np.array(soln['x']))
    proj = lam @ points
    
    return proj