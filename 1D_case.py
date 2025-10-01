import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator

def mixture_gaussian(x, mu1, sigma1, mu2, sigma2, weight1=0.5, weight2=0.5):
    return weight1 * norm.pdf(x, mu1, sigma1) + weight2 * norm.pdf(x, mu2, sigma2)

# Parameters for two Gaussians
mu1, sigma1 = 0, 1   # mean=0, std=1
sigma2 = 1 # std=1

mean2_values = np.linspace(-6, 6, 71)  # Vary mean of second Gaussian from -6 to 6
ci_values = np.linspace(0,1, 71)

percentage_within_bounds_list = []
difference_list = []
max_confidence_list1 = [0 for _ in range(len(ci_values))]
max_confidence_list2 = [0 for _ in range(len(ci_values))]
max_mean_list1 = [0 for _ in range(len(ci_values))]
max_mean_list2 = [0 for _ in range(len(ci_values))]

for i in range(len(mean2_values)):
    mu2 = mean2_values[i]
      
    percentage_within_bounds_ci = []
    difference = []
    for j in range(len(ci_values)):
        ci = ci_values[j]
        lower_bound1, upper_bound1 = norm.interval(ci, loc=mu1, scale=sigma1)
        lower_bound2, upper_bound2 = norm.interval(ci, loc=mu2, scale=sigma2)

        a = min(lower_bound1, lower_bound2)
        b = max(upper_bound1, upper_bound2)
        percentage_within_bounds_1 = quad(mixture_gaussian, lower_bound1, upper_bound1, args=(mu1, sigma1, mu2, sigma2))[0] * 100
        percentage_within_bounds_2 = quad(mixture_gaussian, lower_bound2, upper_bound2, args=(mu1, sigma1, mu2, sigma2))[0] * 100
        
        overlap_lower = max(lower_bound1, lower_bound2)
        overlap_upper = min(upper_bound1, upper_bound2)
        
        if overlap_lower < overlap_upper:  # There is an overlap
            overlap_percentage = quad(mixture_gaussian, overlap_lower, overlap_upper, args=(mu1, sigma1, mu2, sigma2))[0] * 100
        else:
            overlap_percentage = 0
            
        percentage_within_bounds = percentage_within_bounds_1 + percentage_within_bounds_2 - overlap_percentage
        
        diff = percentage_within_bounds - ci*100
        if mu2 < 0 and diff > max_confidence_list1[j]:
            max_confidence_list1[j] = diff
            max_mean_list1[j] = mu2
        if mu2 > 0 and diff > max_confidence_list2[j]:
            max_confidence_list2[j] = diff
            max_mean_list2[j] = mu2
        percentage_within_bounds_ci.append(percentage_within_bounds)
        difference.append(percentage_within_bounds - ci*100)
    percentage_within_bounds_list.append(percentage_within_bounds_ci)
    difference_list.append(difference)

interp = RegularGridInterpolator((mean2_values, ci_values), difference_list)
mean2_fine = np.linspace(-6, 6, 1001)
ci_fine = np.linspace(0, 1, 1001)
X_fine, Y_fine = np.meshgrid(mean2_fine, ci_fine)
Z_fine = interp((X_fine, Y_fine))    
    

# # plot contour plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis')
ax.set_title('Effect of Mean and Confidence Interval on Sample Coverage')
ax.set_ylabel('Confidence Interval')
ax.set_ybound(0,1)
ax.set_xbound(-6, 6)
ax.set_xlabel('Mean of Second Gaussian')
ax.set_zlabel('Difference between Actual and Expected Percentage')

# Contour plot
plt.figure(figsize=(10, 6))
plt.contourf(X_fine, Y_fine, Z_fine, levels=100, cmap='viridis')
plt.colorbar(label='Difference between Actual and Expected Percentage')
plt.contour(X_fine, Y_fine, Z_fine, levels=50, colors='k', linestyles='solid', linewidths=0.8)
plt.plot(max_mean_list1, ci_values, color='r', marker='o', markersize=6, linestyle='None', label='Max Difference')
plt.plot(max_mean_list2, ci_values, color='r', marker='o', markersize=6, linestyle='None')
plt.legend()
plt.title('Effect of Mean and Confidence Interval on Sample Coverage')
plt.ylabel('Confidence Interval')
plt.xlabel('Mean of Second Gaussian')

plt.show()