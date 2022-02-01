import numpy as np
import scipy as sp
import scipy.linalg
import cvxopt
import torch
from torch.autograd import Variable

sqrtm = sp.linalg.sqrtm
inv = sp.linalg.inv

def wass_dist(A,B,A_h=None):
    '''
    gaussian_utilities.wass_dist
        Computes the Wasserstein-2 distance between gaussians

    parameters:
        A - source matrix

        B - target matrix

        A_h - cached computation A^{1/2}
        
    returns:
        Wasserstein-2 distance between A and B.
    '''

    if A_h is None:
        A_h = sqrtm(A)
    return np.sqrt(np.trace(A + B - 2 * sqrtm(A_h @ B @ A_h)))

def optimal_map(A,B,A_h=None, A_nh=None):
    '''
    gaussian_utilities.optimal_map
        Computes the transport matrix from A to B 
        (T(x) = Cx where C is the matrix computed by this function)

    parameters:
        A - source matrix

        B - target matrix

        A_h - cached computation A^{1/2}

        A_nh - cached computation of A^{-1/2}
        
    returns:
        The transport matrix from A to B
    '''

    if A_h is None:
        A_h = sqrtm(A)
    if A_nh is None:
        A_nh = inv(A_h)
    
    opt_map = A_nh @ sqrtm(A_h @ B @ A_h) @ A_nh
    return opt_map

def mccann_interp(A,B,eta=0.5, A_h=None, A_nh=None):
    '''
    gaussian_utilities.mccann_interp
        Computes mccann interpolants between A and B

    parameters:
        A - source matrix

        B - target matrix

        eta - fractional amount of way to move

        A_h - cached computation A^{1/2}

        A_nh - cached computation of A^{-1/2}
        
    returns:
        The transport matrix from A to B
    '''

    if A_h is None:
        A_h = sqrtm(A) # half power
    if A_nh is None:
        A_nh = inv(A_h) #neg half
    
    # optimal map from A to B
    opt_map = optimal_map(A,B,A_h=A_h, A_nh=A_nh)
    
    # map for the McCann interpolation going eta amount
    interp_map = (1 - eta) * np.eye(opt_map.shape[0]) + eta * opt_map
    
    # Formula for cov. under linear transform
    return interp_map @ A @ interp_map

def grad_norm(S_arr, lam, S, S_h=None, S_nh=None):
    '''
    gaussian_utilities.grad_norm
        Computes value of the grad norm (squared) of the references 
        at S with coordinate lam

    parameters:
        S_arr - reference matrices

        lam - barycentric coordinate 

        S - matrix at which the loss is estimated

        S_h - cached computation S^{1/2}

        S_nh - cached computation of S^{-1/2}
        
    returns:
        The value of the gradient norm squared. 
    '''

    p = len(S_arr)
    
    if S_h is None:
        S_h = sqrtm(S)
    if S_nh is None:
        S_nh = inv(S_h)
    
    # M_i, L_i, and R_i matrics 
    M_arr = [sqrtm(S_h @ Si @ S_h) for Si in S_arr]
    L_arr = [S_nh @ Mi for Mi in M_arr]
    R_arr = [Mi @ S_nh for Mi in M_arr]
    
    D = np.array([np.einsum('ij,ji->',Li,Rj) for Li in L_arr for Rj in R_arr]).reshape((p,p))
    e = -2 * np.array([np.trace(Mi) for Mi in M_arr])
    z = np.trace(S)
    
    return lam.T @ D @ lam + lam @ e + z

def true_bc(S_arr, lam, iters=20, initial=None):
    '''
    gaussian_utilities.true_bc
        Finds the optimal setting of lambda based on the gradient norm criteria

    parameters:
        S_arr - list of reference matrices

        lam - coordinate to find the barycenter for

        initial - starting guess for fixed point iteration
            if not provided, defaults to S_arr[0]
        
    returns:
        S - the true barycenter of the references with coordinate lam
    '''

    p = len(S_arr)
    
    if initial is None:
        S = S_arr[0]
    else:
        S = initial
    
    dim = S.shape[0]
    
    for i in range(iters):
        S_h = sqrtm(S)   # matrix square root
        S_nh = inv(S_h) # inverse matrix square root
        
        # first line of Chewi alg 1, integration
        T = np.zeros((dim,dim))
        for j in range(p):
            T = T + lam[j] * sqrtm(S_h @ S_arr[j] @ S_h)
            
        # update covariance using second formula
        S = S_nh @ T @ T @ S_nh
    
    return S


def opt_lam_grad_norm(Si_terms, S, initial=None, S_nh=None):
    '''
    gaussian_utilities.opt_lam_grad_norm
        Finds the optimal setting of lambda based on the gradient norm criteria

    parameters:
        Si_terms - precomputed list of (S^{1/2} @ Si @ S^{1/2})^{1/2} 
            where Si is a reference matrix

        S - matrix to approximate

        initial - starting guess for convex optimization, if None defaults
            to all 1/p vector

        S_nh - Cached version of S^{-1/2}
        
    returns:
        opt_lam - the optimal lambda found by the algorithm
    '''

    p = len(Si_terms)
    
    # constructs the PSD matrix and Linear vector
    
    # pre-computed for many formulas
    if S_nh is None:
        S_nh = inv(sqrtm(S))
        
    L_arr = [S_nh @ Si for Si in Si_terms]
    R_arr = [Si @ S_nh for Si in Si_terms]
    
    # this is equivalent to what we have in the paper 
    D = np.array([np.einsum('ij,ji->',Li,Rj) for Li in L_arr for Rj in R_arr]).reshape((p,p))
    e = -2 * np.array([np.trace(Si) for Si in Si_terms])
    z = np.trace(S)
    
    cvxopt.solvers.options['show_progress'] = False
    
    # see docs for more info https://cvxopt.org/userguide/coneprog.html#quadratic-programming
    # P,q - specify the objective
    # G,h - specify non-negative constraints
    # A,b - sum-to-one constraint
    
    A = cvxopt.matrix(np.ones((1,p))) # for equality constraint
    b = cvxopt.matrix(1.0, (1,1))
    G = cvxopt.matrix(-np.eye(p)) # for inequality constraints
    h = cvxopt.matrix(0.0, (p,1))
    
    if initial is None:
        init = cvxopt.matrix(1/p, (p,1))
    else:
        init = cvxopt.matrix(initial)
    
    # solves the optimization
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P=cvxopt.matrix(2*D), q=cvxopt.matrix(e), 
                                 G=G, h=h, A=A,b=b, initvals={'x':init})
    
    solution['primal objective'] += z
    solution['dual objective'] += z
    return np.squeeze(np.array(solution['x']))


def opt_lam_mle(refs, emp, gamma=0.0003, iters=500, sqrt_iter=10, fixed_iter=10):
    '''
    gaussian_utilities.opt_lam_mle
        Finds the Maximum Likelihood Estimator lambda using gradient descent

    parameters:
        refs - Reference measures as a list of p np.arrays of size [dim, dim]

        emp - measure to approximate as an np array of size [dim, dim]

        gamma - step size parameter

        iters - number of iterations to run

        sqrt_iter - number of iterations for computing the matrix square roots

        fixed_iter - number of fixed point iterations to run
        
    returns:
        opt_lam - the optimal lambda found by the algorithm
    '''

    refs = torch.tensor(refs, dtype=torch.float32)
    emp = torch.tensor(emp, dtype=torch.float32)
    
    p = len(refs)
    lam = torch.tensor(np.ones(p)/p, dtype=torch.float32, requires_grad=True)
    
    # gradient descent iteration
    for i in range(iters):
        # construct the barycenter
        bc = barycenter_torch(refs, lam, emp, sqrt_iter=sqrt_iter, fixed_iter=fixed_iter)

        # compute the loss between the bc and empirical
        loss = kl_loss(emp, bc)

        # back prop
        loss.backward(retain_graph=True)

        with torch.no_grad():
            prev_lam = lam.clone()
            lam = lam - gamma * lam.grad # gradient descent
            lam = proj_vec(lam) # project back onto simplex
        lam.requires_grad = True
        
        if (prev_lam - lam).abs().sum() < 0.0000001:
            break

    return lam.detach().numpy()

def proj_vec(lam):
    '''
    gaussian_utilities.proj_vec
        Projects the vector onto the probability simplex

    parameters:
        lam - tensor of size [dim] to be projected
        
    returns :
        projected - the L2 projection of lam onto the simplex
    '''

    p = lam.shape[0] 
    dtype = lam.dtype
    
    mat0 = torch.zeros(p,p)
    mat0[torch.arange(p)>=torch.transpose(torch.arange(p).unsqueeze(0),1,0)] = 1
    mat0 = mat0.to(dtype=dtype)
    
    mat1 = torch.diag(1/(torch.arange(p)+1))
    mat1 = mat1.to(dtype=dtype)
    
    [U,_] = torch.sort(lam,descending=True)
    U_ = U + (1 - U @ mat0) @ mat1
    rho = torch.max((U_ > 0).nonzero())

    U[torch.arange(p) > rho] = 0
    rho = (1 - U.sum()) / (rho + 1)

    lam = lam + rho
    lam = torch.max(lam, torch.tensor([0.0],dtype=dtype))
    return lam



def kl_loss(S_0, S_1):
    '''
    gaussian_utilities.kl_loss
        Computes the important parts of the KL divergence between S_0 and S_1 for
        optimizing in S_1

    parameters:
        S_0 - tensor of size [dim, dim] representing the first measure

        S_1 - tensor of size [dim, dim] representing the second measure
        
    returns:
        loss - D_KL( N(0,S_0) || N(0,S_1) ), but only including the terms 
        involving S_1
    '''

    inv = torch.inverse(S_1)
    # slogdet returns a tuple containing (sign, logabsdet)
    return torch.trace(inv @ S_0) + torch.slogdet(S_1)[1]

def barycenter_torch(refs, lam, init, sqrt_iter=10, fixed_iter=10):

    '''
    gaussian_utilities.barycenter_torch
        Runs the fixed-point iteration algorithm to find the barycenter of the 
        given reference measures with cooridinate lambda, starting from the given
        initial point

    parameters:
        refs - tensor of size [p, dim, dim] with refs[i] being the i'th ref measure

        lam - barycentric coordinate

        sqrt_iter - number of iterations to run the square root calculation for

        fixed_iter - numbder of fixed-point iterations
        
    returns:
        bc - tensor of size [dim, dim] containing the barycenter
    '''

    [p, dim, _] = refs.shape
    
    # use the initial guess provided spread p times (size [p, dim, dim])
    bc = init.clone()
    for t in range(fixed_iter): 
        # compute the terms inside the integral / sum (size [p, dim, dim])
        
        #             expand because this function expects 3D tensors
        bc_sqrt = sqrt_newton_schulz_autograd(bc.view(1,dim,dim), sqrt_iter)[0]
        sum_terms = bc_sqrt @ refs @ bc_sqrt
        sqrt_terms = sqrt_newton_schulz_autograd(sum_terms, sqrt_iter)
        
        # summing along the first axis, adds the matrices together (size [dim,dim])
        S_t = (lam.view(p,1,1) * sqrt_terms).sum(0) 
        
        # compute the -1/2 power of the current bc (size [dim,dim])
        # we take the 0'th because bc_sqrt was repeating the p times on first axis.
        bc_sqrt_inv = torch.inverse(bc_sqrt)
        
        bc = bc_sqrt_inv @ S_t @ S_t @ bc_sqrt_inv
    return bc

def sqrt_newton_schulz_autograd(mats, numIters):
    '''
    gaussian_utilities.sqrt_newron_schulz_autograd
        Runs the Newton-Shulz algorithm for finding the square root of a matrix
        and does so in a differentiable manner

    parameters:
        mats - tensor of size [p, dim, dim] where we take the sqrt the matrices
            stored as mats[i] for i = 0,...,p-1
            
        numIters - number of iterations to run for

    returns:
        sqrt_mats - tensor os size [p,dim,dim] where sqrt_mats[i] is the 
            sqrt of mats[i]
    '''

    dtype = mats.dtype
    p = mats.data.shape[0]
    dim = mats.data.shape[1]
    
    mat_norms = (mats @ mats).sum(dim=(1,2)).sqrt()
    
    # divide each matrix by its norm (size [p, dim, dim])
    Y = mats.div(mat_norms.view(p, 1, 1).expand_as(mats))
    
    # repeated identy matrix (size [p, dim, dim])
    I = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(p, 1, 1).type(dtype), requires_grad=False)
    
    # repeated identy matrix (size [p, dim, dim])
    Z = Variable(torch.eye(dim, dim).view(1, dim, dim).
                 repeat(p, 1, 1).type(dtype), requires_grad=False)

    # Newton-Schulz iterations
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y)) # bmm multiplies Z[i] @ Y[i]
        Y = Y.bmm(T)
        Z = T.bmm(Z)
        
    sqrt_mats = Y * torch.sqrt(mat_norms).view(p, 1, 1).expand_as(mats)
    
    return sqrt_mats
