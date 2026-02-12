import numpy as np
from numba import njit, prange

@njit
def blahut_update(p_x, p_y_given_x, eps=1e-20):
    """
    Blahut algorithm for computing the mutual information between variable X and variable Y.
    The function takes as input:
    - xvec: the values of the random variable X (N-dim array)
    - yvec: the values of the random variable Y (M-dim array)
    - p_x: the probability mass function of X (N-dim array)
    - p_y_given_x: the conditional probability mass function of Y given X (MxN array)
    The function returns as output:
    - I_L: the lower bound to the mutual information (scalar)
    - I_U: the upper bound to the mutual information (scalar)
    - f_kl_new: updated Bayes risk (N-dim array)
    - p_y_new: updated probability mass function of Y (M-dim array)
    - p_x_new: updated probability mass function of X (N-dim array)
    """
    N = len(p_x)
    M = len(p_y_given_x)

    p_y = np.zeros(M) # output dist
    f_kl = np.zeros(N) # Bayes risk
    p_x_new = np.zeros(N) # updated input dist
    for a in range(N):
        for b in range(M):
            p_y[b] += p_y_given_x[b,a]*p_x[a]
            
    for a in range(N):
        for b in range(M):  
            f_kl[a] += p_y_given_x[b,a]*np.log((p_y_given_x[b,a]+eps)/(p_y[b]+eps))
        p_x_new[a] = p_x[a]*np.exp(f_kl[a]) 

    p_x_new /= p_x_new.sum() # normalize

    p_y_new = np.zeros(M) # updated output dist
    f_kl_new = np.zeros(N) # updated Bayes risk
    I_L = 0 # lower bound to capacity
    for a in range(N):
        for b in range(M):
            p_y_new[b] += p_y_given_x[b,a]*p_x_new[a]
    
    I_U = 0 # upper bound to capacity
    for a in range(N):
        for b in range(M):
            f_kl_new[a] += p_y_given_x[b,a]*np.log((p_y_given_x[b,a]+eps)/(p_y_new[b]+eps))
        I_L += p_x_new[a]*f_kl_new[a]
        if f_kl_new[a] > I_U:
            I_U = f_kl_new[a]

    return I_L, I_U, f_kl_new, p_x_new

def weight_step(convergence_goal, max_iter, weightvec, p_y_given_x):
    """
    When starting out with a discrete prior with K atoms, this function will run a Blahut algorithm until a chosen level of convergence to set the weights of the atoms. 
    The function takes as input:
    - convergence goal: goal for I_U - I_L (scalar)
    - max_iter: maximum number of iterations (scalar)
    - yvec: the values of the random variable Y (M-dim array)
    - weightvec: the weight of each atom of X (K-dim array)
    - p_y_given_x: the conditional probability mass function of Y given X (MxK array)
    The function returns as output:
    - err: the difference I_U - I_L (scalar)
    - weightvec: optimised weights for given position (K-dim array)
    """
    I_L, I_U = 0, 1
    tau = 0
    while I_U - I_L > convergence_goal and tau < max_iter:
        I_L, I_U, f_kl_new, weightvec = blahut_update(weightvec, p_y_given_x)
        tau += 1
    err = I_U - I_L
    return err, weightvec

@njit
def pos_step(yvec, weightvec, p_y_given_x, d_pyx_dx):
    """
    When starting out with a discrete prior with K atoms, this function will return the gradients of the positions of the atoms.
    - yvec: the values of the random variable Y (M-dim array)
    - weightvec: the weight of each atom of X (K-dim array)
    - p_y_given_x: the conditional probability mass function of Y given X (MxK array)
    - d_pyx_dx: the gradient of the conditional probability mass function of Y given X with respect to the positions of the atoms (MxK array)
    The function returns as output:
    - pos_grad: the gradient of the positions of the atoms (K-dim array)
    """

    # discretisation steps
    dy = yvec[1] - yvec[0]

    # output distribution
    p_y = np.zeros(len(yvec))
    for i in range(len(yvec)):
        for j in range(len(weightvec)):
            p_y[i] += p_y_given_x[i,j]*weightvec[j]

    # position gradient 
    pos_grad = np.zeros(len(weightvec))
    for i in range(len(yvec)):
        for j in range(len(weightvec)):
            pos_grad[j] += d_pyx_dx[i,j]*np.log((p_y_given_x[i,j]+1e-10)/(p_y[i]+1e-10))*dy

    pos_grad = pos_grad*weightvec

    return pos_grad