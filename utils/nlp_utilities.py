import numpy as np
import ot
import utils.opt_utilities as ou

def compute_ent_map(data, bow_x, idx1, idx2, entropy=0.1):
    '''
    nlp_utilities.compute_ent_map
        Computes the estiamte of the map between two point clouds
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        idx1 - index of the source document

        idx2 - index of the target document

        entropy - amount of entropy to use in finding the map

    returns
        a matrix representation of the map estimate
    '''

    # pull out the point clouds
    x1 = data[idx1].T
    x2 = data[idx2].T
    a = bow_x[idx1][0]
    a = a / a.sum()
    b = bow_x[idx2][0]
    b = b / b.sum()

    # compute the optimal coupling matrix
    gamma = ot.bregman.empirical_sinkhorn(x1,x2,entropy,a=a,b=b)
    
    # turn it into a map
    ent_map = (gamma @ x2) / a[:, np.newaxis]
    
    return ent_map

def inner_products(data, bow_x, base_idx, ref_idxs, entropy=0.1):
    '''
    nlp_utilities.inner_products
        Computes the inner product matrix A
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        entropy - amount of entropy to use in finding the map

    returns
        the estimated inner product matrix A
    '''

    # compute the entropic maps
    ent_maps = [compute_ent_map(data, bow_x, base_idx, ref_idx, entropy=entropy) for ref_idx in ref_idxs]
    
    # subtract the support out for comp below
    supp = data[base_idx].T
    adj_maps = [em - supp for em in ent_maps]
    
    # get the base distribution
    dist = bow_x[base_idx][0]
    dist = dist / dist.sum()
    
    p = len(ref_idxs)
    A = np.zeros((p,p))
    # fill in the matrix A
    for i, map_i in enumerate(adj_maps):
        for j, map_j in enumerate(adj_maps):
            A[i,j] = np.dot((map_i*map_j).sum(1), dist)
            
    return A

def w2_dist(data, bow_x, idx1, idx2):
    '''
    nlp_utilities.w2_dist
        Computes W_2^2 between two point clouds (using NO entropy)
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        idx1 - index of the source document

        idx2 - index of the target document

    returns
        the estimated inner product matrix \hat{A}
    '''

    # pull out the point clouds
    x1 = data[idx1].T
    x2 = data[idx2].T
    a = bow_x[idx1][0]
    a = a / a.sum()
    b = bow_x[idx2][0]
    b = b / b.sum()

    # compute the distance
    M = ot.dist(x1, x2)
    return ot.lp.emd2(a,b,M)


def knn_multiclass_loss(data, bow_x, base_idx, ref_idxs, classes):
    '''
    nlp_utilities.knn_multiclass_loss
        Finds the class of the nearest ref document to the base document
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents

    returns
        the label of the nearest reference documnet
    '''

    # compute the distances and extracts the minimizing index
    nearest_index = np.argmin([w2_dist(data, bow_x, base_idx, ref_idx) for ref_idx in ref_idxs])
    # return the corresponding class
    return classes[nearest_index]

def avgdist_multiclass_loss(data, bow_x, base_idx, ref_idxs, classes):
    '''
    nlp_utilities.avgdist_multiclass_loss
        Finds the class which is on average closest (lowest avg. W22) to the base 
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents

    returns
        the label of the nearest reference documnet
    '''

    # compute the distances
    dists = [w2_dist(data, bow_x, base_idx, ref_idx) for ref_idx in ref_idxs]
    
    class_labels = np.unique(classes)
    total_dists = np.zeros(len(class_labels))
    
    # sum the distances, stratified by class label
    for i,c in enumerate(classes):
        total_dists[class_labels == c] += dists[i]

    # return the class with the lowest total distance 
    # (we always use the same number of refs in each class so this matches the avg.)    
    return class_labels[np.argmin(total_dists)]

def bc_multiclass_loss(data, bow_x, base_idx, ref_idxs, classes):
    '''
    nlp_utilities.bc_multiclass_loss
        Finds the class achieves the best objective in the quadratic program
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents

    returns
        the label of the class which minimizes the gradient norm objective
    '''

    class_labels = np.unique(classes)
    losses = np.zeros(len(class_labels))
    
    # iterate over the classes
    for i,class_label in enumerate(class_labels):
        # get the A matrix
        A = inner_products(data, bow_x, base_idx, ref_idxs[classes == class_label])

        # recover lambda and the loss
        [_, loss] = ou.solve(A, return_val=True)
        
        losses[i] = loss
    
    # return the label that gets the best loss
    return class_labels[np.argmin(losses)]

def mc_multiclass_loss(data, bow_x, base_idx, ref_idxs, classes):
    '''
    nlp_utilities.mc_multiclass_loss
        Finds the class that is used the most in the barycentric approximation
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents

    returns
        the label of the class that is used the most in the barycentric approximation
    '''

    # get the A matrix over everything
    A = inner_products(data, bow_x, base_idx, ref_idxs)
    
    # recover lambda 
    lam = ou.solve(A)
    
    class_labels = np.unique(classes)
    uses = np.zeros(len(class_labels))
    
    # sum over the coordintes of lambda
    for i,c in enumerate(classes):
        uses[class_labels == c] += lam[i]
    
    # return the most used classes
    return class_labels[np.argmax(uses)]


def w2_predictors(data, bow_x, base_idx, ref_idxs, classes):
    '''
    nlp_utilities.w2_predictors
        Performs both knn and min. avg. distance together, re-using the 
        distances to avoid redundant compuatations
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents

    returns
        [knn_label, avd_label] - both predictions
    '''
    
    
    # compute the distances
    dists = [w2_dist(data, bow_x, base_idx, ref_idx) for ref_idx in ref_idxs]

    # knn
    nearest_index = np.argmin(dists)
    knn_label = classes[nearest_index]
    
    # min. avg. distance
    class_labels = np.unique(classes)
    total_dists = np.zeros(len(class_labels))
    for i,c in enumerate(classes):
        total_dists[class_labels == c] += dists[i]
    avd_label = class_labels[np.argmin(total_dists)]
    
    return [knn_label, avd_label]


def bc_predictors(data, bow_x, base_idx, ref_idxs, classes):
    '''
    nlp_utilities.bc_predictors
        Performs both min. bc loss and max coord together, re-using the 
        inner-products to avoid redundant compuatations
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents

    returns
        [bc_label, mc_label] - both predictions
    '''
    

    # get the A matrix, we need the whole thing for max coordinate
    # but only sub matrices for bc loss
    A = inner_products(data, bow_x, base_idx, ref_idxs)
    
    # recover lambda in the max coordinate setting
    lam_mc = ou.solve(A)
    
    # tabulate which class has the most used coordinates
    class_labels = np.unique(classes)
    uses = np.zeros(len(class_labels))
    for i,c in enumerate(classes):
        uses[class_labels == c] += lam_mc[i]
    mc_label = class_labels[np.argmax(uses)]

    # iterate over the classes
    losses = np.zeros(len(class_labels))
    
    # iterate over the classes
    for i,class_label in enumerate(class_labels):
        # get the part of the matrix above for this class
        A_i = A[classes == class_label][:, classes == class_label]

        # recover lambda and the loss
        [_, loss] = ou.solve(A_i, return_val=True)
        
        losses[i] = loss
    bc_label = class_labels[np.argmin(losses)]
    
    return [bc_label, mc_label]

# we do this to save the time of computing the optimal maps twice which 
# takes a the most time in the experiments
def nested_w2_predictors(data, bow_x, base_idx, ref_idxs, classes, refs_range):
    '''
    nlp_utilities.nested_w2_predictors
        Performs knn and min average distance across a range of number of references.
        This massively reduces the amount of time needed for the computations
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents 
            assumed to be structured like [1,1,1,2,2,2,3,3,3] with each digit repeated
            max(refs_range) times.

        refs_range - list of number of references to used (assumed to)

    returns
        [[knn_label, avd_label]]  - array of array of both predictions, each row corresponding
        to a number of references in refs_range
    '''

    class_labels = np.unique(classes)
    nclasses = len(class_labels)
    
    results = np.zeros((len(refs_range), 2))
    
    # compute all the distances
    dists = np.array([w2_dist(data, bow_x, base_idx, ref_idx) for ref_idx in ref_idxs])
    
    for i,nrefs in enumerate(refs_range):
        # this gives the indices of the refs being used when only using nrefs
        ref_inds = np.arange(nclasses).repeat(nrefs) * refs_range[-1] + np.tile(np.arange(nrefs), nclasses)
        ref_inds = ref_inds.astype(int)
        
        # extracts the corresponding distances and classes
        sub_dists = dists[ref_inds]
        sub_classes = classes[ref_inds]
        
        # knn prediction on subset
        nearest_index = np.argmin(sub_dists)
        knn_label = sub_classes[nearest_index]

        # min. avg. dist. prediction on subset
        total_dists = np.zeros(len(class_labels))
        for j,c in enumerate(sub_classes):
            total_dists[class_labels == c] += sub_dists[j]
        avd_label = class_labels[np.argmin(total_dists)]
        
        results[i,0] = knn_label 
        results[i,1] = avd_label
    
    return results

def nested_bc_predictors(data, bow_x, base_idx, ref_idxs, classes, refs_range):
    '''
    nlp_utilities.nested_bc_predictors
        Performs min bc loss and max coord across a range of number of references.
        This massively reduces the amount of time needed for the computations
        
    parameters
        data - list of point cloud supports. data[idx1] is a matrixthe support of the base cloud

        bow_x - bag-of-words vector. bow_x[idx1] is the word counts in the base document

        base_idx - index of the source document

        ref_idxs - indices of the reference documents

        classes - labels for the reference documents 
            assumed to be structured like [1,1,1,2,2,2,3,3,3] with each digit repeated
            max(refs_range) times.

        refs_range - list of number of references to used (assumed to)

    returns
        [[knn_label, avd_label]]  - array of array of both predictions, each row corresponding
        to a number of references in refs_range
    '''

    class_labels = np.unique(classes)
    nclasses = len(class_labels)
    
    results = np.zeros((len(refs_range), 2))
    # get the giant A matrix, we will only use certain parts
    # dependent on the number of refs and technique used
    A = inner_products(data, bow_x, base_idx, ref_idxs)
    
    for i,nrefs in enumerate(refs_range):
        # this gives the indices of the refs being used when only using nrefs
        ref_inds = np.arange(nclasses).repeat(nrefs) * refs_range[-1] + np.tile(np.arange(nrefs), nclasses)
        ref_inds = ref_inds.astype(int)
        
        # extracts the parts of A corresponding to this number of references
        A_submat = A[ref_inds][:, ref_inds]
        # extracts the classes too
        sub_classes = classes[ref_inds]
    
        # recover lambda in the max coordinate setting
        lam_mc = ou.solve(A_submat)

        # tabulate which class has the most used coordinates
        uses = np.zeros(len(class_labels))
        for j,c in enumerate(sub_classes):
            uses[class_labels == c] += lam_mc[j]
        mc_label = class_labels[np.argmax(uses)]

        # iterate over the classes
        losses = np.zeros(len(class_labels))
        # iterate over the classes
        for j,class_label in enumerate(class_labels):
            # get the part of the matrix above for this class
            A_j = A_submat[sub_classes == class_label][:, sub_classes == class_label]

            # recover lambda and the loss
            [_, loss] = ou.solve(A_j, return_val=True)

            losses[j] = loss
        bc_label = class_labels[np.argmin(losses)]
        
        results[i,0] = bc_label
        results[i,1] = mc_label
    
    return results