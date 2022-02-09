'''
Small Samples

This script compares several methods for estimating the coordinate lambda.
It is especially focused on the small sample regime where there may be 
a large discrepancy between the empirical covariance and the true covariance.

It can be used to generate Figure 3
'''

import numpy as np
import scipy as sp
import scipy.stats
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from utils import gaussian_utilities as gu

sqrtm = sp.linalg.sqrtm
    
#========= Trial Settings ==========

print('Trial Settings')

# change these to do more repititions or change the number
# of samples being seen.
ntrials = 250
testpoints = (np.arange(30)+1)*20 

print('  p', 6) # needs to be hardcoded below
print('  dim', 10) # needs to be hardcoded below
print('  testpoints', testpoints)
print('  ntrials',ntrials)

#=========== Real Code Starts Here ===========

print('=============== EXECUTION =================\n\n')

params = list(testpoints) * ntrials

def trial(nsamples):
    
    p = 6
    dim = 10
    
    while True:
        try:

            # generates the reference measures
            S_arr = [sp.stats.wishart.rvs(dim, np.eye(dim)) + 0.5*np.eye(dim) for k in range(p)]

            # generates the true lambda
            lam = np.squeeze(sp.stats.dirichlet.rvs(alpha=np.ones(p)))

            # generates the true bc
            S = gu.true_bc(S_arr, lam, iters=20)
            S_h = sqrtm(S) # cache the square root
            
            # samples from the true bc
            samples = sp.stats.multivariate_normal.rvs(cov=S, size=nsamples)
            
            # empirical covariance
            S_emp = (samples.T @ samples) / nsamples
            S_emp_h = sqrtm(S_emp)
            
            # pre compute this
            Si_terms = [sqrtm(S_emp_h @ Si @ S_emp_h) for Si in S_arr]
            
            lam_mle = gu.opt_lam_mle(S_arr, S_emp)
            lam_gnm = gu.opt_lam_grad_norm(Si_terms, S_emp)
            
            S_mle = gu.true_bc(S_arr, lam_mle, iters=7, initial=S)
            S_gnm = gu.true_bc(S_arr, lam_gnm, iters=7, initial=S)
            
            emp_dist = gu.wass_dist(S,S_emp, A_h=S_h)
            mle_dist = gu.wass_dist(S,S_mle, A_h=S_h)
            gnm_dist = gu.wass_dist(S,S_gnm, A_h=S_h)
            
            mle_l1 = np.abs(lam_mle - lam).sum()
            gnm_l1 = np.abs(lam_gnm - lam).sum()
            
            return [nsamples, emp_dist, gnm_dist, mle_dist, gnm_l1, mle_l1] 
        
        except:
            print("Exception Occurred, Likely related to auto-diff")
    
# njobs is the number of processes to run at the same time. 
# if you have more cores, you can take advantage of them
# by increasing this number

results = Parallel(n_jobs=1)(delayed(trial)(nsamples) for nsamples in tqdm(params, position=0, leave=True))
results = np.array(results)

# parse the reults 
# Wasserstein distances between estimate and true distribution
emp_dists = results[:,1].reshape((250,30))
gnm_dists = results[:,2].reshape((250,30))
mle_dists = results[:,3].reshape((250,30))

# L1 error in estimating the coordinate.
gnm_l1s = results[:,4].reshape((250,30))
mle_l1s = results[:,5].reshape((250,30))

linspace = (np.arange(30) + 1) * 20

# compute average distances
emp_means = emp_dists.mean(0)
mle_means = mle_dists.mean(0)
gnm_means = gnm_dists.mean(0)
# and standard deviations
emp_stds = emp_dists.std(0)
mle_stds = mle_dists.std(0)
gnm_stds = gnm_dists.std(0)

# then plot everything
plt.fill_between(linspace, emp_means - emp_stds, emp_means + emp_stds, 
                 alpha=0.2, label='_nolegend_')
plt.plot(linspace, emp_means)

plt.fill_between(linspace, mle_means - mle_stds, mle_means + mle_stds, 
                 alpha=0.2, label='_nolegend_')
plt.plot(linspace, mle_means)

plt.fill_between(linspace, gnm_means - gnm_stds, gnm_means + gnm_stds, 
                 alpha=0.2, label='_nolegend_')
plt.plot(linspace, gnm_means)

plt.ylim([0,1.5])
plt.legend(['Emprical','MLE','Gradient Norm'])
plt.xlabel('Number of Samples')
plt.ylabel('Wasserstein-2 Distance')
plt.title('Distance to Recovered Matrix')
plt.tight_layout()
plt.show()

print("DONE")
