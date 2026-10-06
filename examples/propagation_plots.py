import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, norm
from scipy.stats import chi2

def distribution_plot(mean,covariance,samples):

    # Generate X and Y values for plotting
    x1_Upper_bound = mean[0] + 4*covariance[0][0]
    x1_Lower_bound = mean[0] - 4*covariance[0][0]
    x2_Upper_bound = mean[1] + 4*covariance[1][1]
    x2_Lower_bound = mean[1] - 4*covariance[1][1]
    x = np.linspace(x1_Lower_bound, x1_Upper_bound, 100)
    y = np.linspace(x2_Lower_bound, x2_Upper_bound, 100)

    # Extract the marginal distributions
    x1 = norm.pdf(x, mean[0], np.sqrt(covariance[0][0]))  # Marginal for X
    x2 = norm.pdf(y, mean[1], np.sqrt(covariance[1][1]))  # Marginal for Y

    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 📌 Subplot 1: Distribution of x1
    ax1 = axes[0]
    ax1.plot(x, x1, label="x1 Marginal", color="b")
    ax1.hist(samples[:,0], bins=30, density=True, alpha=0.3, color="blue", label="x1 Samples")
    ax1.set_title("Distribution of x1")
    ax1.legend()

    # 📌 Subplot 2: Distribution of x2
    ax2 = axes[1]
    ax2.plot(y, x2, label="x2 Marginal", color="r")
    ax2.hist(samples[:,1], bins=30, density=True, alpha=0.3, color="red", label="x2 Samples")
    ax2.set_title("Distribution of x2")
    ax2.legend()

    # Adjust layout and show the plots
    plt.tight_layout()
    #plt.show()

def result_plot_grid(points, bools, mean, covariance, percent, ax, row, col):

    # Split points into two groups based on the boolean condition
    green_points = points[bools]
    red_points = points[~bools]  # Inverts the boolean mask

    # Plot green points (True)
    ax[row,col].scatter(green_points[:, 0], green_points[:, 1], color='green', label="Valid (True)", s=1)

    # Plot red points (False)
    ax[row,col].scatter(red_points[:, 0], red_points[:, 1], color='red', label="Invalid (False)", s=1)

    # Find and plot Ellipsoid (2D)
    ellipse_scaled = find_ellipse(covariance, mean, percent)
    ax[row,col].plot(ellipse_scaled[0, :], ellipse_scaled[1, :], label=f'{percent*100}% Ellipsoid')

    #ax.legend()

    return ax

def result_plot(ax, points, bools, mean, covariance, percent):
    bools = np.asarray(bools, dtype=bool)

    green_points = points[bools]
    red_points = points[~bools]

    ax.scatter(green_points[:, 0], green_points[:, 1], color='green', label="Valid (True)", s=1)
    ax.scatter(red_points[:, 0], red_points[:, 1], color='red', label="Invalid (False)", s=1)

    ellipse_scaled = find_ellipse(covariance, mean, percent)
    ax.plot(ellipse_scaled[0, :], ellipse_scaled[1, :], label=f'{percent*100}% Ellipsoid')

    #ax.legend()
    return ax



def result_plot_DA(points, bools, mean, percent):

    # Split points into two groups based on the boolean condition
    green_points = points[bools]
    red_points = points[~bools]  # Inverts the boolean mask

    # Create plot
    plt.figure(figsize=(6, 6))

    # Plot green points (True)
    plt.scatter(green_points[:, 0], green_points[:, 1], color='green', label="Valid (True)", s=1)

    # Plot red points (False)
    plt.scatter(red_points[:, 0], red_points[:, 1], color='red', label="Invalid (False)", s=1)

    #Find Ellipsoid (2D)
    #ellipse_scaled = find_ellipse(covariance, mean, percent)

    #plt.plot(ellipse_scaled[0, :], ellipse_scaled[1, :], label=f'{percent*100}% Ellipsoid')




def find_ellipse(covariance, mean, percent):
    covariance_final_2d = covariance[:2, :2]
    eigvals, eigvecs = np.linalg.eigh(covariance_final_2d)
    eigvecs_n1 = np.linalg.norm(eigvecs[:,0])
    eigvecs_n2 = np.linalg.norm(eigvecs[:,1])
    scalling_factor = np.sqrt(chi2.ppf(percent, 2)) 
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.array([np.cos(theta), np.sin(theta)])
    ellipse_scaled = (eigvecs @ np.diag(np.sqrt(eigvals) * scalling_factor) @ ellipse)
    ellipse_scaled[0, :] += mean[0]
    ellipse_scaled[1, :] += mean[1]

    return ellipse_scaled

import numpy as np
from scipy.stats import chi2

def find_single_point_on_ellipse_6d(covariance, mean, percent):
    eigvals, eigvecs = np.linalg.eigh(covariance)  # Eigen decomposition for 6D
    scalling_factor = np.sqrt(chi2.ppf(percent, 6))
    ellipse_point = np.random.multivariate_normal(np.zeros(6), np.identity(6))
    ellipse_point = ellipse_point/ np.linalg.norm(ellipse_point)
    #print("Ellipse Point", np.linalg.norm(ellipse_point))
    ellipse_scaled = eigvecs @ np.diag(np.sqrt(eigvals) * scalling_factor) @ ellipse_point
    ellipse_scaled = ellipse_scaled.reshape(6)
    ellipse_scaled += mean  # Translate to the correct mean
    
    return ellipse_scaled

def find_single_point_on_ellipse_2d(covariance, mean, percent):
    eigvals, eigvecs = np.linalg.eigh(covariance)  # Eigen decomposition for 6D
    scalling_factor = np.sqrt(chi2.ppf(percent, 2))
    ellipse_point = np.random.multivariate_normal(np.zeros(2), np.identity(2))
    ellipse_point = ellipse_point/ np.linalg.norm(ellipse_point)
    ellipse_scaled = eigvecs @ np.diag(np.sqrt(eigvals) * scalling_factor) @ ellipse_point
    ellipse_scaled = ellipse_scaled.reshape(2)
    ellipse_scaled += mean  # Translate to the correct mean
    
    return ellipse_scaled