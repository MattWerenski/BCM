'''
Functions for working with MNIST digits
'''

import numpy as np
import ot
import ot.bregman

DIM = 28

def unwrap_image(arr, bounds):
    '''
    mnist_utilities.unwrap_image
        Flattens an image of height (bounds[1] - bounds[0]) and 
        width (bounds[3] - bounds[2]) into a vector of corresponding length

    parameters
        arr - 2D np.array representing the image

        bounds - (bounds[1] - bounds[0]) is the height and 
            (bounds[3] - bounds[2]) is the width
        
    returns
        an np.array containing a flattened version of the image
    '''

    # this whole method could probably be arr.reshape(-1)

    height = bounds[1] - bounds[0]
    width = bounds[3] - bounds[2]
    
    unwrapped = []
    for j in range(height):
        for k in range(width):
            unwrapped += [arr[j,k]]
    
    return np.array(unwrapped)

def wrap_image(arr, bounds):
    '''
    mnist_utilities.wrap_image
        Takes a flatten image and returns it to an image
        using the sizes in the bounds array

    parameters
        arr - 1D np.array representing the flat image

        bounds - (bounds[1] - bounds[0]) is the height and 
            (bounds[3] - bounds[2]) is the width
        
    returns
        an 2D np.array containing the un-flattened image
    '''

    # this whole method could probably be arr.reshape((height, width))

    height = bounds[1] - bounds[0]
    width = bounds[3] - bounds[2]
    
    wrapped = np.zeros((height,width))
    for j in range(height):
        for k in range(width):
            wrapped[j,k] = arr[j*width+k]
            
    return wrapped

def crop(images):
    '''
    mnist_utilities.crop
        Takes a set of images and returns versions of them with all the mutual 
        padding removed. For example, if none of the images have an mass in the 
        two columns leftmost columns, then those are removed. However if
        any image puts mass in the third column then that column is kept for all.

    paramaters
        images - list of 28x28 np.arrays to crop

    returns
        cropped - cropped versions of the images
        
        bounds - indices of the rows and columns the section corresponds to
            bounds[0] = first row, bounds[1] = last row, 
            bounds[2] = first col bounds[3] = last col
    '''

    # rows and column sums for each image
    rows = [img.sum(1) for img in images]
    cols = [img.sum(0) for img in images]
    
    # sum across the sums of each image
    rows = np.asarray(rows).sum(0)
    cols = np.asarray(cols).sum(0)

    # above lines can probably be np.array(images).sum(1 or 2)
    

    # find the tightest row and column indices of the shared support
    top = 0
    while rows[top] == 0:
        top += 1
    bottom = DIM - 1
    while rows[bottom] == 0:
        bottom -= 1
    bottom += 1
    left = 0
    while cols[left] == 0:
        left += 1
    right = DIM - 1
    while cols[right] == 0:
        right -= 1
    right += 1
    
    # Crop the images
    cropped = [img[top:bottom, left:right] for img in images]
    # and record the indices
    bounds = (top, bottom, left, right)
    
    return cropped, bounds

def uncrop(image, bounds):
    '''
    mnist_utilities.uncrop
        Takes in a smaller patch of an image and pastes it onto the
        spot on a 28x28 grid  described by bouds

    paramaters
        image - 2D np.array of sizes at most 28x28
        bounds - indices of the rows and columns to paste the image

    returns
        cropped - cropped versions of the images
        bounds - indices of the rows and columns the section corresponds to
            bounds[0] = first row, bounds[1] = last row, 
            bounds[2] = first col bounds[3] = last col
    '''

    uncropped = np.zeros((DIM, DIM))
    uncropped[bounds[0]:bounds[1], bounds[2]:bounds[3]] = image
    return uncropped

def plan_to_map(plan, support):
    '''
    mnist_utilities.plan_to_map
        Computes the barycentric projection of an optimal plan to obtain a map

    parameters
        plan - 2D np.array of assignments. The marginal sums corresponds to the 
            mass allocations of the source and target distributions

        support - Placements of the support of the target distributions

    returns
        2D np.array of size nsource x 2 where nsource is the number of points
            in the source distribution. 
    '''

    marginal = plan.sum(1)
    has_mass = marginal > 0
    no_mass = 1 - has_mass
    
    # if the rows of plan summed to 1, this would be the average location
    # that the plan sends each coordinate to 
    unweighted = plan @ support 
    
    
    # plan.shape[0] is number of points in the source
    # support.shape[1] is the dimension we work in
    weighted = np.zeros((plan.shape[0],support.shape[1]))
    
    # here we account for the fact that the rows don't sum to 1
    weighted[has_mass,:] = unweighted[has_mass,:] / marginal[has_mass,np.newaxis]
    
    # if theres no mass there, then this value doesn't matter
    # so we set it to map to itself
    weighted[no_mass,:] = support[no_mass,:]
    
    return weighted

def barycenter(refs, weights, threshold=0.00001, entropy=0.1):
    '''
    mnist_utilities.barycenter 
        Computes the Wasserstein barycenter of the reference measures for the
        given weights. This function relies on the POT library 
        
    parameters
        refs - list of reference measures, each represented
            as a 2D array of intensities, normalized to sum to 1.

        weight - mixture weight for which the barycenter is computed

        threshold - any coordinate with less mass than this is considered 0

        entropy - amount of entropy to use
        
    returns
        the barycenter of the given references using the mixture weight
        as a DIM by DIM np.array.
    '''

    # filter out any refs given mass below the threshold.
    # this helps with speed and stability
    weights_used = weights[weights > threshold]
    weights_used = weights_used / weights_used.sum()
    refs_used = refs[weights > threshold,:,:]
    
    # this helps shrink the problem by removing all zero rows and columns
    cropped, bounds = crop(refs_used)
    
    # creates a list of supports for the cropped images
    height = bounds[1] - bounds[0]
    width = bounds[3] - bounds[2]
    support = []
    for j in range(height):
        for k in range(width):
            support += [[j,k]]
    # normalizes so the support is a patch of [0,1] x [0,1]
    # this is optional, all that matters is ratio of distances to entropy
    support = np.array(support) / 28
    
    # turns the 2D array into a 1D array
    unwrap_refs = np.asarray([unwrap_image(ref, bounds) for ref in cropped]).T
    
    # distance or cost matrix
    M = ot.dist(support, metric='sqeuclidean')
    M = np.asarray(M, dtype=np.float64)
    
    # computes the barycenter using sinkhorn 
    unwrapped_bc = ot.barycenter(unwrap_refs, M,  entropy,
                                 weights=weights_used)
    
    # wraps the vector back into an image
    uncropped_bc = uncrop(wrap_image(unwrapped_bc, bounds), bounds)
    
    return uncropped_bc

def inner_products(base, refs, supp=None):
    
    '''
    mnist_utilities.inner_products
        Computes the inner products of the maps from base to ref WITHOUT ENTROPY
        in the tangent space at the base and returns a matrix of these.
        
    parameters
        base - the measure we are trying to approximate with a barycenter

        refs - list of reference measures, each represented
            as a 2D array of intensities, normalized to sum to 1.

        supp - 2D array of shape npoints x 2, each entry being a support point
            if this is not passed, then the default 28x28 grid is used
        
    returns
        p x p np.array A with A_ij the inner product for references i and j
    '''

    [dim1, dim2] = base.shape
    
    if supp is None: # faster to cache than re-compute
        supp = []
        for i in range(dim1):
            for j in range(dim2):
                supp += [[i,j]]
        supp = np.array(supp)
        
    # flattens images to arrays
    base_uw = base.reshape(dim1*dim2)
    ref_uws = [ref.reshape(dim1*dim2) for ref in refs]
    
    # take out zero mass parts of the base
    base_used = base_uw > 0
    base_dist = base_uw[base_used]
    base_dist = base_dist / base_dist.sum()
    base_supp = supp[base_used, :]
    
    opt_maps = []
    
    for ref_uw in ref_uws:
        
        # take out zero mass parts of the reference
        ref_used = ref_uw > 0
        ref_dist = ref_uw[ref_used]
        ref_dist = ref_dist / ref_dist.sum()
        ref_supp = supp[ref_used,:]
        
        # distance matrix
        M = ot.dist(base_supp, x2=ref_supp, metric='sqeuclidean')
        
        # compute the optimal plant
        gamma = ot.lp.emd(base_dist, ref_dist, M)
        
        # and obtain a map as the barycentric projection
        opt_maps += [plan_to_map(gamma, ref_supp)]
        
    # pre-perform the subtraction of the identity
    adjusted_maps = [opt_map - base_supp for opt_map in opt_maps]
    
    p = refs.shape[0]
    A = np.zeros((p,p))
    
    # actually fill in the A matrix
    for i, map_i in enumerate(adjusted_maps):
        for j, map_j in enumerate(adjusted_maps):
            ip = np.dot((map_i * map_j).sum(1), base_dist)
            A[i,j] = ip
            
    return A

def entropic_inner_products(base, refs, entropy=5, supp=None):

    '''
    mnist_utilities.inner_products
        Computes the inner products of the maps from base to ref WITH ENTROPY
        in the tangent space at the base and returns a matrix of these.
        
    parameters
        base - the measure we are trying to approximate with a barycenter

        refs - list of reference measures, each represented
            as a 2D array of intensities, normalized to sum to 1.

        entropy - the amount of entropy to use in the optimization procedure
        
        supp - 2D array of shape npoints x 2, each entry being a support point
            if this is not passed, then the default 28x28 grid is used
        
    returns
        p x p np.array A with A_ij the inner product for references i and j
    '''
    
    [dim1, dim2] = base.shape
    
    if supp is None: # faster to cache than re-compute
        supp = []
        for i in range(dim1):
            for j in range(dim2):
                supp += [[i,j]]
        supp = np.array(supp)
        
    # flattens images to arrays
    base_uw = base.reshape(dim1*dim2)
    ref_uws = [ref.reshape(dim1*dim2) for ref in refs]
    
    # take out zero mass parts of the base
    base_used = base_uw > 0
    base_dist = base_uw[base_used]
    base_dist = base_dist / base_dist.sum()

    base_supp = supp[base_used, :]
    
    opt_maps = []
    
    for ref_uw in ref_uws:
        
        # take out zero mass parts of the reference
        ref_used = ref_uw > 0
        ref_dist = ref_uw[ref_used]
        ref_dist = ref_dist / ref_dist.sum()
        ref_supp = supp[ref_used,:]
        
        M = ot.dist(x1=base_supp, x2=ref_supp, metric='sqeuclidean')
        
        gamma = ot.bregman.sinkhorn(base_dist, ref_dist, M, entropy, warn=False)
        
        opt_maps += [plan_to_map(gamma, ref_supp)]
        
    # pre-perform the subtraction of the identity
    adjusted_maps = [opt_map - base_supp for opt_map in opt_maps]
    
    p = refs.shape[0]
    A = np.zeros((p,p))
    
    # actually fill in the A matrix
    for i, map_i in enumerate(adjusted_maps):
        for j, map_j in enumerate(adjusted_maps):
            ip = np.dot((map_i * map_j).sum(1), base_dist)
            A[i,j] = ip
            
    return A

    