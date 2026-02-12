import numpy as np

def blahut_update(xvec, yvec, p_x, p_y_given_x, discrete_x=False):
    """
    Blahut algorithm for computing the mutual information between discrete or continuous variable X and continuous variable Y.
    The function takes as input:
    - xvec: the values of the random variable X (Nx1 array)
    - yvec: the values of the random variable Y (Mx1 array)
    - p_x: the probability mass function of X (Nx1 array)
    - p_y_given_x: the conditional probability mass function of Y given X (MxN array)
    - discrete_x: boolean indicating whether X is discrete or continuous (bool)
    The function returns as output:
    - I_L: the lower bound to the mutual information (scalar)
    - I_U: the upper bound to the mutual information (scalar)
    - f_kl_new: updated Bayes risk (Nx1 array)
    - p_x_new: updated probability mass function of X (Nx1 array)
    """
    # discretisation steps
    if not discrete_x:
        dx = xvec[1] - xvec[0] # scalar
    dy = yvec[1] - yvec[0] # scalar

    # output distribution
    p_y = np.matmul(p_y_given_x, p_x) # Mx1 array
    p_y /= np.sum(p_y*dy)
    p_y.reshape(-1,1)

    # Bayes risk
    f_kl = np.sum(p_y_given_x*np.log((p_y_given_x+1e-10)/(p_y+1e-10)), axis=0)*dy 
    f_kl = f_kl.reshape(-1,1) # Nx1 array

    # update prior
    p_x_new = p_x*np.exp(f_kl).copy() # Nx1 array
    if not discrete_x:
        p_x_new /= np.sum(p_x_new*dx)
    if discrete_x:
        p_x_new /= np.sum(p_x_new)

    # new output distribution
    p_y_new = np.matmul(p_y_given_x, p_x_new) # Mx1 array
    p_y_new /= np.sum(p_y_new*dy)
    p_y_new.reshape(-1,1)

    f_kl_new = np.sum(p_y_given_x*np.log((p_y_given_x+1e-10)/(p_y_new+1e-10)), axis=0)*dy
    f_kl_new = f_kl_new.reshape(-1,1) # Nx1 array

    # lower and upper bounds to channel capacity
    if not discrete_x:
        I_L = (np.sum(p_x_new*f_kl_new)*dx)[0]
    if discrete_x:
        I_L = (np.sum(p_x_new*f_kl_new))

    I_U = np.max(f_kl_new)
    return I_L, I_U, f_kl_new, p_x_new

def weight_step(convergence_goal, max_iter, yvec, pos, weightvec, p_y_given_x):
    """
    When starting out with a discrete prior with K atoms, this function will run a Blahut algorithm until a chosen level of convergence to set the weights of the atoms. 
    The function takes as input:
    - convergence goal: goal for I_U - I_L (scalar)
    - max_iter: maximum number of iterations (scalar)
    - yvec: the values of the random variable Y (Mx1 array)
    - weightvec: the weight of each atom of X (Kx1 array)
    - p_y_given_x: the conditional probability mass function of Y given X (MxK array)
    The function returns as output:
    - err: the difference I_U - I_L (scalar)
    - weightvec: optimised weights for given position (Kx1 array)
    """
    I_L, I_U = 0, 1
    tau = 0
    while I_U - I_L > convergence_goal and tau < max_iter:
        #I_L, I_U, weightvec = discrete_blahut_update(yvec, weightvec, p_y_given_x)
        I_L, I_U, f_kl_new, weightvec = blahut_update(pos, yvec, weightvec, p_y_given_x, discrete_x=True)
        tau += 1
    err = I_U - I_L
    return err, weightvec

def pos_step(yvec, weightvec, p_y_given_x, d_pyx_dx):
    # discretisation steps
    dy = yvec[1] - yvec[0]

    # output distribution
    p_y = np.matmul(p_y_given_x, weightvec) # Mx1 array

    # position gradient 
    pos_grad = np.sum(d_pyx_dx*np.log((p_y_given_x+1e-10)/(p_y+1e-10)), axis=0)*dy # Kx1 array
    pos_grad = pos_grad.reshape(-1,1) # Kx1 array
    pos_grad = pos_grad*weightvec

    return pos_grad

def blahut_update_disc(p_x, p_y_given_x):
    """
    Blahut algorithm for computing the mutual information between discrete variable X and discrete variable Y.
    The function takes as input:
    - p_x: the probability mass function of X (Nx1 array)
    - p_y_given_x: the conditional probability mass function of Y given X (MxN array)
    The function returns as output:
    - I_L: the lower bound to the mutual information (scalar)
    - I_U: the upper bound to the mutual information (scalar)
    - f_kl_new: updated Bayes risk (Nx1 array)
    - p_x_new: updated probability mass function of X (Nx1 array)
    """

    # output distribution
    p_y = np.matmul(p_y_given_x, p_x) # Mx1 array
    p_y /= np.sum(p_y)
    p_y.reshape(-1,1)

    # Bayes risk
    f_kl = np.sum(p_y_given_x*np.log((p_y_given_x+1e-10)/(p_y+1e-10)), axis=0) 
    f_kl = f_kl.reshape(-1,1) # Nx1 array

    # update prior
    p_x_new = p_x*np.exp(f_kl).copy() # Nx1 array
    p_x_new /= np.sum(p_x_new)

    # new output distribution
    p_y_new = np.matmul(p_y_given_x, p_x_new) # Mx1 array
    p_y_new /= np.sum(p_y_new)
    p_y_new.reshape(-1,1)

    f_kl_new = np.sum(p_y_given_x*np.log((p_y_given_x+1e-10)/(p_y_new+1e-10)), axis=0)
    f_kl_new = f_kl_new.reshape(-1,1) # Nx1 array

    # lower and upper bounds to channel capacity
    I_L = (np.sum(p_x_new*f_kl_new))

    I_U = np.max(f_kl_new)
    return I_L, I_U, f_kl_new, p_x_new
