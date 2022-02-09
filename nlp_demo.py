'''
NLP

This script compares several methods for predicting the topic of a document.
This code follows the setup described in section 4.4 and it can be used 
to generate Figure 5
'''

import numpy as np
from tqdm import tqdm
import scipy.io as sio
from joblib import Parallel, delayed
from utils import nlp_utilities as nu
import matplotlib.pyplot as plt

# USE THIS ONE FOR BBCSports
mat_fname = '../NLP/bbcsport-emd_tr_te_split.mat' 
mat_contents = sio.loadmat(mat_fname)
data = mat_contents['X'][0]# each document contains a set of support points (word2vec)
labels = mat_contents['Y'][0]# each document's class
BOW_X = mat_contents['BOW_X'][0]# Same shape as data, the value shows how often a certain word is shown in the document, 

'''
# USE THIS ONE FOR News20
mat_fname = '../NLP/20ng2_500-emd-tr-te.mat' 
mat_contents = sio.loadmat(mat_fname)
data = mat_contents['xtr'][0]# each document contains a set of support points (word2vec)
labels = mat_contents['ytr'][0]# each document's class
BOW_X = mat_contents['BOW_xtr'][0]# Same shape as data, the value shows how often a certain word is shown in the document, 
'''

nlabels = np.unique(labels).shape[0] # number of unique labels

nreps = 50
ntest = 100
nrefs_range = np.arange(11)+2 # 2-12
nclasses = len(np.unique(labels))

def nested_trial(data, bow_x, base_idx, ref_idxs, classes, refs_range): 
    w2_results = nu.nested_w2_predictors(data, bow_x, base_idx, ref_idxs, classes, refs_range)
    bc_results = nu.nested_bc_predictors(data, bow_x, base_idx, ref_idxs, classes, refs_range)
    return np.hstack([w2_results, bc_results])


pbar = tqdm(total=nreps*ntest*nrefs_range.sum(), position=0, leave=True)
results = np.zeros((len(nrefs_range), nreps, 4))
 
for rep in range(nreps):

    # generate the set of reference and test measures

    max_refs = nrefs_range[-1]
    
    # extract a random set of references, nrefs from each class
    ref_classes = np.arange(nclasses).repeat(max_refs) + 1
    ref_idxs = np.zeros(max_refs * nclasses, dtype=int)
    for j in range(nclasses):
        # find all the dists with label j+1
        inclass = np.where(labels == j+1)[0]

        # choose nrefs of them at random
        perm = np.random.permutation(len(inclass))
        refs = inclass[perm[:max_refs]]

        ref_idxs[j*max_refs:(j+1)*max_refs] = refs

    # all the non-reference measures
    non_refs = np.delete(np.arange(len(labels)), ref_idxs)

    # take ntest random to try on
    perm = np.random.permutation(len(non_refs))
    test_idxs = non_refs[perm[:ntest]]
    test_labels = labels[test_idxs]

    # njobs is the number of processes to run at the same time. 
    # if you have more cores, you can take advantage of them
    # by increasing this number
    predictions = Parallel(n_jobs=1)(
        delayed(nested_trial)(data, BOW_X, test_idx, ref_idxs, ref_classes, nrefs_range) 
        for test_idx in test_idxs
    )
    predictions = np.array(predictions)
    
    correct = predictions == test_labels[:,np.newaxis,np.newaxis]
    acc = correct.sum(0) / ntest
    results[:, rep, :] = acc
    
    pbar.update(ntest*nrefs_range.sum())
                
pbar.close()

plt.plot(nrefs_range, results.mean(1), linewidth=2.2)
plt.xlabel('Number of Reference Documents per Class', labelpad=0.5)
plt.ylabel('Accuracy', labelpad=0.5)
plt.legend(['1NN','Min. Avg. Dist.','Min. BC Loss','Max. Coordinate'], loc='lower right')
plt.title('BBC Sports Topic Prediction')
plt.tight_layout()
plt.show()
print("DONE")