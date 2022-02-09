import numpy as np
import scipy as sp
import scipy.stats
import mnist
import ot
import tqdm
import matplotlib.pyplot as plt

from utils import mnist_utilities, opt_utilities

# Load the data

# change this to where you want MNIST saved to / where it is saved
mnist.temporary_dir = lambda: '../mnist'

train_images = mnist.train_images()
train_labels = mnist.train_labels()


# ============= GENERATE REF AND TARGS ============= 

print("Genererating Dataset")

# set up the dataset (noiseless)
digit = 4
nref_digit = 10
ntarg_digit = 500

indices = np.where(train_labels == digit)[0]
perm = np.random.permutation(len(indices))
ref_inds = indices[perm[0:nref_digit]]
targ_inds = indices[nref_digit:nref_digit+ntarg_digit]

ref_digits = [train_images[ri] for ri in ref_inds]
targ_digits = [train_images[ti] for ti in targ_inds]        
for i in range(nref_digit):
    ref_digits[i] = ref_digits[i] / ref_digits[i].sum()
for i in range(ntarg_digit):
    targ_digits[i] = targ_digits[i] / targ_digits[i].sum()
ref_digits = np.array(ref_digits)


# ============= PERTURB DIGITS ============= 

print("Perturbing Dataset")

'''
# run this for additive noise

# set up the dataset (noisy)
noise_level = 0.5

ref_perts = [] # add expected noise
targ_perts = [] # add noise
        
for i in range(nref_digit):
    ref_digits[i] = ref_digits[i] / ref_digits[i].sum()
    ref_perts += [(1-noise_level)*ref_digits[i] + noise_level/(28*28)]
    
for i in range(ntarg_digit):
    noise = sp.stats.uniform.rvs(0,1,size=(28,28)) 
    noise = noise / noise.sum()
    targ_digits[i] = targ_digits[i] / targ_digits[i].sum()
    targ_perts += [(1-noise_level)*targ_digits[i] + noise_level*noise]
'''

# run this for occlussion

mask = np.ones((28,28))
mask[10:18,10:18] = 0

ref_perts = [] # apply the mask
targ_perts = [] # apply the mask
        
for i in range(nref_digit):
    ref_digits[i] = ref_digits[i] / ref_digits[i].sum()
    ref_pert = ref_digits[i] * mask
    ref_pert = ref_pert / ref_pert.sum()
    ref_perts += [ref_pert]
    
for i in range(ntarg_digit):
    targ_digits[i] = targ_digits[i] / targ_digits[i].sum()
    targ_pert = targ_digits[i] * mask
    targ_pert = targ_pert / targ_pert.sum()
    targ_perts += [targ_pert]

ref_perts = np.array(ref_perts)

# ============= RECOVER LAMBDA ============= 

print("Recovering Lambda")
print("  No Entropy")

# no entropy
noe_lams = np.zeros((ntarg_digit,nref_digit))
for i in tqdm.tqdm(range(ntarg_digit)):
    targ_pert = targ_perts[i]
    
    # compute the matrix A (filled with inner products)
    A = mnist_utilities.inner_products(targ_pert, ref_perts)
    
    # recovers the estimate of lambda by solving
    lam = opt_utilities.solve(A)
    
    noe_lams[i,:] = lam

print("  Some Entropy")

# some entropy
supp = []
for i in range(28):
    for j in range(28):
        supp += [[i,j]]
supp = np.array(supp)

ent_lams = np.zeros((ntarg_digit,nref_digit))
for i in tqdm.tqdm(range(ntarg_digit)):
    targ_pert = targ_perts[i]
    
    # compute the matrix A (filled with inner products)
    A = mnist_utilities.entropic_inner_products(targ_pert, ref_perts, entropy=10, supp=supp)
    
    # recovers the estimate of lambda by solving
    lam = opt_utilities.solve(A)
    
    ent_lams[i,:] = lam


# ============= RECOVER TARGETS ============= 

print("Recovering Digit")

linspace = np.arange(28)
M = ot.dist(supp, metric='sqeuclidean')
M = np.asarray(M, dtype=np.float64)

noe_dists = np.zeros(ntarg_digit)
ent_dists = np.zeros(ntarg_digit)
for i in tqdm.tqdm(range(ntarg_digit)):
    noe_lam = noe_lams[i]
    ent_lam = ent_lams[i]
    targ_digit = targ_digits[i]
    
    noe_bc = mnist_utilities.barycenter(ref_digits, noe_lam, entropy=0.001)
    ent_bc = mnist_utilities.barycenter(ref_digits, ent_lam, entropy=0.001)
    
    
    noe_unwrap = mnist_utilities.unwrap_image(noe_bc, [0,28,0,28])
    ent_unwrap = mnist_utilities.unwrap_image(ent_bc, [0,28,0,28])
    targ_unwrap = mnist_utilities.unwrap_image(targ_digit, [0,28,0,28])
    
    noe_unwrap = noe_unwrap / noe_unwrap.sum()
    ent_unwrap = ent_unwrap / ent_unwrap.sum()
    targ_unwrap = targ_unwrap / targ_unwrap.sum()
    
    noe_dist = ot.lp.emd2(noe_unwrap, targ_unwrap, M)
    ent_dist = ot.lp.emd2(ent_unwrap, targ_unwrap, M)
    
    noe_dists[i] = noe_dist
    ent_dists[i] = ent_dist


print(f"  No Entropy {noe_dists.mean()}")
print(f"  Entropy {ent_dists.mean()}")

# pick a random perturbed target and generate a row of Figure 4
i = np.random.randint(0,ntarg_digit)

ax = plt.subplot(1,4,1)
ax.imshow(targ_perts[i])

ax = plt.subplot(1,4,2)
proj = opt_utilities.linear_projection(targ_perts[i].reshape(-1), ref_digits.reshape((nref_digit,-1)))
proj = proj.reshape((28,28))
ax.imshow(proj)

ax = plt.subplot(1,4,3)
bc = mnist_utilities.barycenter(ref_digits,ent_lams[i], entropy=0.001, threshold=0.001)
ax.imshow(bc)

ax = plt.subplot(1,4,4)
ax.imshow(targ_digits[i])

plt.show()


print("DONE")