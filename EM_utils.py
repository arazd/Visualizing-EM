import numpy as np
import os
import matplotlib.pyplot as plt
import math
import scipy.stats as stats
import scipy

# this file contains python functions
# for visualization of EM steps for Gaussian Mixture Model

def plot_pdf(u_arr, sigma_arr):
    x = np.linspace(-1, 3, 1000)
    plt.rcParams['figure.figsize']=(9,3)

    for mu, s in zip(u_arr, sigma_arr):
        plt.plot(x, stats.norm.pdf(x, mu, s))


def plot_points(points, R=np.nan):
    if (np.isnan(R).any()):
        plt.scatter(points, np.zeros((points.shape[0])), color='black', s=12, marker='o')
    else:
        for i, point in enumerate(points):
            if R[i,0] < R[i,1]:
                col = 'orange'
            else:
                col = 'blue'
            plt.scatter(point, np.zeros((1)), color=col, s=12, marker='o')



def E_step(x_arr, pi_arr, u_arr, sigma_arr):

    N1 = scipy.stats.norm(u_arr[0], sigma_arr[0])
    N2 = scipy.stats.norm(u_arr[1], sigma_arr[1])

    R = np.empty((x_arr.shape[0],2))

    for i, point in enumerate(x_arr):
        R[i,0] = pi_arr[0]*N1.pdf(x_arr[i]) / ( pi_arr[0]*N1.pdf(x_arr[i])+pi_arr[1]*N2.pdf(x_arr[i]) )
        R[i,1] = pi_arr[1]*N2.pdf(x_arr[i]) / ( pi_arr[0]*N1.pdf(x_arr[i])+pi_arr[1]*N2.pdf(x_arr[i]) )

    return R



def M_step(x_arr, R):
    pi = np.average(R, axis=0)

    u = np.empty(2)
    for i in range(2):
        u[i] = np.dot(R[:,i],x_arr) / np.sum(R[:,i])

    sigma = np.empty(2)
    for i in range(2):
        sigma[i] = np.sqrt( np.dot(R[:,i],(x_arr-u[i])**2) / np.sum(R[:,i]) )

    return pi, u, sigma



def get_log_likelihood(x_arr, u_arr, sigma_arr, pi_arr):
    N1 = scipy.stats.norm(u_arr[0], sigma_arr[0])
    N2 = scipy.stats.norm(u_arr[1], sigma_arr[1])

    log_likelihood = 0
    for i, point in enumerate(x_arr):
        log_likelihood += np.log(pi_arr[0]*N1.pdf(x_arr[i])+pi_arr[1]*N2.pdf(x_arr[i]))

    return log_likelihood


def get_lower_bound(x_arr, u_arr, sigma_arr, pi_arr, R):
    N1 = scipy.stats.norm(u_arr[0], sigma_arr[0])
    N2 = scipy.stats.norm(u_arr[1], sigma_arr[1])

    log_likelihood = np.empty(100)
    for i, point in enumerate(x_arr):
        # we want to avoid numerical overflow in case of small values
        if (R[i,1]<0.0001):
            log_likelihood += R[i,0]*np.log(pi_arr[0]*N1.pdf(x_arr[i])/R[i,0])
        elif (R[i,0]<0.0001):
            log_likelihood += R[i,1]*np.log(pi_arr[1]*N2.pdf(x_arr[i])/R[i,1] )
        else:
            log_likelihood += R[i,0]*np.log(pi_arr[0]*N1.pdf(x_arr[i])/R[i,0]) + R[i,1]*np.log(pi_arr[1]*N2.pdf(x_arr[i])/R[i,1] )

    return log_likelihood
