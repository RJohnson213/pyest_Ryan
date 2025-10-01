import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from STMint.STMint import STMint
from diskcache import Cache

import pyest.gm as pygm
import pyest.gm.split as split
from pyest.filters.sigma_points import SigmaPointOptions
from pyest.gm import GaussianMixture
from pyest.linalg import triangularize

from pyest.filters.sigma_points import unscented_transform
import os
from datetime import datetime
from scipy.stats import chi2
import json

# plotting functions
mpl.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

def temp_funct(py, samples, alpha):
    # Function is used to determine number of points that fall within specified bounds
    count = 0
    threshold = chi2.ppf(alpha, df=py.m.shape[1])
    mu = py.get_m()
    for i in range(len(samples)):
        x = samples[i]
        for mixand in range(py.m.shape[0]):
            L = py.Schol[mixand]   
            y = np.linalg.solve(L, (x - mu[mixand]))
            D_2 = y.T @ y
            if D_2 <= threshold:
                count += 1
                break
    percentage = count / len(samples)
    # print(f"Number of points within {alpha*100:.2f}% confidence interval: {count} out of {len(samples)} ({percentage*100:.2f}%)")
    return percentage

def temp2(py, samples, split_method, expected_percentage):
    actual_percentage = []
    difference = []
    print('split method: ', split_method)
    for alpha in expected_percentage:
        actual = temp_funct(py, samples, alpha)
        actual_percentage.append(actual)
        difference.append(actual - alpha)
    return actual_percentage, difference
    # plt.figure()
    # plt.plot(expected_percentage, actual_percentage, '-o')
    # plt.xlabel('Expected Percentage within Confidence Interval')
    # plt.ylabel('Actual Percentage within Confidence Interval')
    # plt.title('Actual vs expected Percentage, ' + split_method)
    # plt.xlim([min(expected_percentage), max(expected_percentage)])
    # plt.figure()
    # plt.plot(expected_percentage, difference, '-x')
    # plt.xlabel('Expected Percentage within Confidence Interval')
    # plt.ylabel('Difference between Actual and Expected')
    # plt.title('Difference between Actual and Expected Percentage, ' + split_method)
    # plt.xlim([min(expected_percentage), max(expected_percentage)])
    # plt.figure()
    # plt.plot(actual_percentage, '-o')
    # plt.plot(expected_percentage, '-x')
    # plt.xlabel('Index of Confidence Interval')
    # plt.ylabel('Percentage within Confidence Interval')
    # plt.title('Actual vs expected Percentage, ' + split_method)
    # plt.legend(['Actual', 'Expected'])
    # plt.ylim([0, 1])
    # plt.xlim([0, len(expected_percentage) - 1])
    # for i in range(len(expected_percentage)):
    #     if i == 0:
    #         off = 'left'
    #     else:
    #         off = 'right'
    #     plt.annotate(
    #         "", 
    #         xy=(i, expected_percentage[i]),   # arrow end
    #         xytext=(i, actual_percentage[i]), # arrow start
    #         arrowprops=dict(arrowstyle="<->", color="gray", lw=1)
    #     )
    #     # Place the difference label midway between the two curves
    #     mid_y = (expected_percentage[i] + actual_percentage[i]) / 2
    #     diff = expected_percentage[i] - actual_percentage[i]
    #     plt.text(i, mid_y, f"{diff:.2f}", ha=off, va="center", fontsize=13, color="black")
    # Added by RJ end


def sample_final(num_samples, py):
    # Sample from the Gaussian mixture model
    samples = np.zeros((num_samples, py.m.shape[1]))
    weights = py.get_w()
    cum_weights = np.cumsum(weights)
    for i in range(num_samples):
        r = np.random.rand()
        for j in range(len(cum_weights)):
            if r <= cum_weights[j]:
                samples[i, :] = np.random.multivariate_normal(py.m[j], py.P[j])
                break
    return samples

def bounds_from_meshgrids(XX1, YY1, XX2, YY2):
    x_max = np.max(np.concatenate([XX1.ravel(), XX2.ravel()]))
    x_min = np.min(np.concatenate([XX1.ravel(), XX2.ravel()]))
    y_max = np.max(np.concatenate([YY1.ravel(), YY2.ravel()]))
    y_min = np.min(np.concatenate([YY1.ravel(), YY2.ravel()]))
    return x_min, x_max, y_min, y_max


def save_figure(example, split_method, ax, fig, w=3, h=3):
    # save title text before clearing the title
    title_text = ax.get_title()
    ax.set_title('')
    fig.set_size_inches(w=w, h=h)
    filename = example + split_method.replace(' ', '_')
    filename = folder_name + "/" + filename
    fig.savefig(filename + '.svg', bbox_inches='tight', pad_inches=0)
    ax.set_title(title_text)


def plot_split_and_transformed(p_split, py, split_method_str, example, dims=(0, 1),
                               scatter_means=True, xf_lim=None, yf_lim=None, ax_equal=False):
    num_contours = 100
    scatter_plt_args = {'marker': 'x', 'zorder': 2, 'color': 'k'}
    scatter_plt_overlay_args = {'s': 5**2, 'marker': 'x',
                                'zorder': 2.1, 'color': 'w', 'alpha': 0.9, 'linewidth': 1}
    # Plot the split density
    pp, XX, YY = p_split.pdf_2d(res=300, dimensions=dims)
    plt.figure()
    plt.contour(XX, YY, pp, num_contours)
    plt.title('Original Density, split,  ' + split_method_str, wrap=True)
    plt.colorbar()
    if scatter_means:
        plt.scatter(p_split.m[:, dims[0]],
                    p_split.m[:, dims[1]], **scatter_plt_args)
        plt.scatter(p_split.m[:, dims[0]], p_split.m[:,
                    dims[1]], **scatter_plt_overlay_args)
    plt.grid()
    labels = ['$x$', '$y$', '$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
    plt.xlabel(labels[dims[0]])
    plt.ylabel(labels[dims[1]])

    save_figure(example, split_method_str + "before_map" +
                str(dims), plt.gca(), plt.gcf())

    # Plot the transformed split density
    pp, XX, YY = py.pdf_2d(res=300, dimensions=dims, xbnd=xf_lim, ybnd=yf_lim)
    fig, ax = plt.subplots()
    c = ax.contour(XX, YY, pp, num_contours, linewidths=0.5)
    fig.colorbar(c)
    ax.set_title('Transformed Density, ' + split_method_str, wrap=True)
    if scatter_means:
        ax.scatter(py.m[:, dims[0]], py.m[:, dims[1]], **scatter_plt_args)
        ax.scatter(py.m[:, dims[0]], py.m[:, dims[1]],
                   **scatter_plt_overlay_args)
    ax.grid()
    if xf_lim is not None:
        ax.set_xlim(xf_lim)
    if yf_lim is not None:
        ax.set_ylim(yf_lim)
    if ax_equal:
        ax.set_aspect('equal', adjustable='box')

    labels = ['$x$', '$y$', '$z$', r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']
    ax.set_xlabel(labels[dims[0]])
    ax.set_ylabel(labels[dims[1]])

    save_figure(example, split_method_str + '_' +
                str(dims[0]) + '_' + str(dims[1]), plt.gca(), plt.gcf())

    pp, XX, YY = py.pdf_2d(res=300, dimensions=dims)
    return pp, XX, YY
# end plotting utilities


# square root EKF propagation for individual mixands
def transform_density_ekf(p_split, ny, g, G):
    my = np.zeros((len(p_split), ny))
    Sy = np.zeros((len(p_split), ny, ny))
    for i in range(len(p_split)):
        my[i] = g(p_split.m[i])
        Gval = G(*p_split.m[i])
        Sy[i] = triangularize(Gval @ p_split.Schol[i])

    wy = p_split.w.copy()
    return GaussianMixture(wy, my, Sy, cov_type='cholesky')

def transform_density_ukf(p_split, ny, g, sigma_pt_opts, residual_fun=None, mean_fun=None):
    # Compute the unscented transform of the Gaussian mixture
    my = np.zeros((len(p_split),ny))
    Sy = np.zeros((len(p_split),ny,ny))
    for i in range(len(p_split)):
        # padding for non-positive definiteness issue with unscented transform of very small covariances
        my[i], Sy[i], _, _, _ = unscented_transform(p_split.m[i], p_split.Schol[i], g,
                                                    sigma_pt_opts=sigma_pt_opts, cov_type='cholesky',
                                                    residual_fun=residual_fun, mean_fun=mean_fun)
    wy = p_split.w.copy()
    return GaussianMixture(wy, my, Sy, cov_type='cholesky')

# density propagation example in a Cislunar NRHO
def cislunar_example(split_count):
    example = 'cislunar'
    # nrho ics
    mu = 1.0 / (81.30059 + 1.0)
    x0 = 1.02202151273581740824714855590570360
    z0 = -0.182096761524240501132977765539282777
    yd0 = -0.103256341062793815791764364248006121
    period = 1.5111111111111111111111111111111111111111
    transfer_time = period * 0.5

    x_0 = np.array([x0, 0, z0, 0, yd0, 0])
    ny = 6  # dimension of y
    nx = 6  # dimension of x

    weights = np.array([1])  # single component
    cov_0 = 0.00001**2 * np.identity(6) + \
        0.0001**2 * (np.diag([1, 0, 1, 0, 0, 0]))
    p0 = GaussianMixture(weights, np.array([x_0]), np.array([cov_0]))

    # nrho propagator
    integrator = STMint(preset="threeBody", preset_mult=mu,
                        variational_order=2)
    max_integrator_step = period/500.0
    int_tol = 1e-13

    # outputs x_f, STM, STT
    def flow_info(x, y, z, vx, vy, vz): return integrator.dynVar_int2(
        [0, transfer_time], [x, y, z, vx, vy, vz], rtol=int_tol, atol=int_tol, output="final"
    )
    
    # outputs just the hessian
    def hessian_func(x, y, z, vx, vy, vz): return integrator.dynVar_int2(
        [0, transfer_time], [x, y, z, vx, vy, vz], rtol=int_tol, atol=int_tol, output="final"
    )[2]
    
    # outputs just the jacobian
    def jacobian_func(x, y, z, vx, vy, vz): return integrator.dynVar_int(
        [0, transfer_time], [x, y, z, vx, vy, vz], rtol=int_tol, atol=int_tol, output="final"
    )[1]
    # outputs flow of state only

    def propagation(x_0): return integrator.dyn_int([0, transfer_time], x_0,
                                                    max_step=max_integrator_step,
                                                    t_eval=[transfer_time]).y[:, -1]

    # apply splitting methods
    split_opts = pygm.GaussSplitOptions(
        L=3, lam=1e-3, recurse_depth=split_count, min_weight=-np.inf) # L^recurse_depth

    # Define the unscented transform parameters
    sigma_pt_opts = SigmaPointOptions(alpha=1e-3, beta=2, kappa=0)

    print("running monte carlo")
    # create/load split cache
    cislunar_mc_cache = Cache(__file__[:-3] + 'cislunar_mc_cache')
    # reference Monte Carlo (store points and pdf value at point)
    num_points = int(1e4)
    rng = np.random.default_rng(100)
    if 'samples' in cislunar_mc_cache:
        print("cache found, loading samples from cache")
        samples = cislunar_mc_cache['samples']
        final_samples = cislunar_mc_cache['final_samples']
        assert (len(samples) == num_points)
    else:
        print("cache not found, propagating samples")
        samples = rng.multivariate_normal(x_0, cov_0, num_points)
        final_samples = list(map(propagation, samples))
        cislunar_mc_cache['samples'] = samples
        cislunar_mc_cache['final_samples'] = final_samples
        print("MC propagation complete, cache saved with {} samples".format(num_points))

    # idx_pairs = [(0, 1), (0, 2), (1, 2), (3, 4),
    #             (4, 5), (3, 5), (0, 4), (1, 3)]
    # axis_labels = ['$x$', '$y$', '$z$',
    #                r'$\dot{x}$', r'$\dot{y}$', r'$\dot{z}$']

    # # scatter plotting for Monte Carlo
    # xlim = dict()
    # ylim = dict()
    # for idx_pair in idx_pairs:
    #     plt.figure()
    #     plt.scatter(np.array(final_samples)[:, idx_pair[0]], np.array(
    #         final_samples)[:, idx_pair[1]], marker='+', alpha=0.025)
    #     plt.xlabel(axis_labels[idx_pair[0]])
    #     plt.ylabel(axis_labels[idx_pair[1]])
    #     save_figure(example, "truth_scatter" + '_' +
    #                 str(idx_pair[0]) + '_' + str(idx_pair[1]), plt.gca(), plt.gcf())
    #     plt.figure()
    #     plt.hist2d(np.array(final_samples)[:, idx_pair[0]], np.array(
    #         final_samples)[:, idx_pair[1]], 40)
    #     plt.xlabel(axis_labels[idx_pair[0]])
    #     plt.ylabel(axis_labels[idx_pair[1]])
    #     # save axis limits for later
    #     xlim[idx_pair] = plt.gca().get_xlim()
    #     ylim[idx_pair] = plt.gca().get_ylim()
    #     save_figure(example, "truth_hist" + '_' +
    #                 str(idx_pair[0]) + '_' + str(idx_pair[1]), plt.gca(), plt.gcf())

    recursive_split_args = {}
    # use the same number of recursive splits for each mixand
    split_tol = -np.inf
    # settings for the SADL and ALoDT based metrics
    diff_stat_det_sigma_pt_opts = SigmaPointOptions(
        alpha=0.5)  # spread sigma points farther

    # define parameters associated with each splitting method
    # recursive_split_args['variance'] = (split.id_variance, split_tol)
    # recursive_split_args['USFOS'] = (split.id_usfos, jacobian_func, split_tol)
    # recursive_split_args['WUSSADL'] = (
    #     split.id_wussadl, jacobian_func, propagation, diff_stat_det_sigma_pt_opts, split_tol)
    recursive_split_args['WUSSOLC'] = (
        split.id_wussolc, hessian_func, jacobian_func, split_tol)

    # additional splitting methods
    # uncomment these if desired
    # recursive_split_args['ALoDT'] = (split.id_max_alodt, propagation, diff_stat_det_sigma_pt_opts, split_tol)
    # recursive_split_args['FOS'] = (split.id_fos, jacobian_func, split_tol)
    # recursive_split_args['SAFOS'] = (split.id_safos, jacobian_func, split_tol)
    # recursive_split_args['USFOS'] = (split.id_usfos, jacobian_func, split_tol)
    # recursive_split_args['SOS'] = (split.id_sos, hessian_func, jacobian_func, split_tol)
    # recursive_split_args['SASOS'] = (split.id_sasos, hessian_func, split_tol)
    # recursive_split_args['WSASOS'] = (split.id_wsasos, hessian_func, jacobian_func, split_tol)
    # recursive_split_args['WUSSOS'] = (split.id_wussos, hessian_func, jacobian_func, split_tol)
    # recursive_split_args['SOLC'] = (split.id_solc, hessian_func, split_tol)
    # recursive_split_args['USSOLC'] = (split.id_ussolc, hessian_func, split_tol)
    # recursive_split_args['SADL'] = (split.id_sadl, jacobian_func, propagation, diff_stat_det_sigma_pt_opts, split_tol)
    
    PY_UKF = []
    PY_EKF = []
    methods = []

    # plot the resulting GMM densities propagated
    for split_method, args in recursive_split_args.items():
        p_split = split.recursive_split(p0, split_opts, *args)
        # py = transform_density_ekf(p_split, ny, propagation, jacobian_func)
        
        # # Added by RJ start
        
        # # Define the unscented transform parameters
        # sigma_pt_opts = SigmaPointOptions(alpha=1e-3, beta=2, kappa=0)
        
        py_ukf = transform_density_ukf(p_split, ny, propagation, sigma_pt_opts)
        py_ekf = transform_density_ekf(p_split, ny, propagation, jacobian_func)
        
        py = py_ukf

        # for idx_pair in idx_pairs:
        #     _, XX, YY = plot_split_and_transformed(
        #         p_split, py, split_method, example, idx_pair, xf_lim=xlim[idx_pair], yf_lim=ylim[idx_pair])
            
        PY_UKF.append(py_ukf)
        PY_EKF.append(py_ekf)
        methods.append(split_method)

    # plt.show()
    return PY_UKF, PY_EKF, methods, samples, final_samples, p0


if __name__ == '__main__':
    # run the example
    
    # This section of code is added by RJ to save figures in a timestamped folder
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.exists("Figures") is False:
        os.makedirs("Figures")
        print("Figures will be saved to new Figures folder")
    else:
        print("Figures will be saved to existing Figures folder")
    folder_name = f"Figures/Cislunar_example_{current_time}"
    os.makedirs(folder_name, exist_ok=True)
    
    folder = "savedresults"
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    run = input('would you like to generate new results? (y/n): ')
    if run == 'y' or len(files) == 0:
        print("running cislunar example")
        
        # These lists will store results for different numbers of splits. each sublist corresponds to a different number of splits.
        # Each element of the sublist corresponds to a different splitting method. P0, Samples and Final samples are the same for each splitting method so they are not nested lists
        UKF = []
        EKF = []
        P0 = []
        methods_list = []
        Samples = []
        Final_samples = []
        
        split_count = [0, 1, 2, 3, 4, 5]  # number of splits is 3**split_count;

        
        for i in range(len(split_count)):
            split_c = split_count[i]
            PY_UKF, PY_EKF, methods, samples, final_samples, p0 = cislunar_example(split_c)
            UKF.append(PY_UKF)
            EKF.append(PY_EKF)
            P0.append(p0)
            methods_list.append(methods)
            Samples.append(samples)
            Final_samples.append(final_samples)

        print("collection complete")
        
        expected_percentage = np.linspace(0.8, 0.99, 20)
        
        # following lists store the actual and difference values for each number of splits and each splitting method
        # each sublist corresponds to a different number of splits. each element of the sublist corresponds to a different splitting method
        Init_Actual = []
        Init_Difference = []
        Final_Actual_UKF = []
        Final_Difference_UKF = []
        Final_Actual_EKF = []
        Final_Difference_EKF = []
        for i in range(len(split_count)):
            # print(f"analyzing {3**split_count[i]} splits")
            PY_UKF = UKF[i]
            PY_EKF = EKF[i]
            p0 = P0[i]
            methods = methods_list[i]
            samples = Samples[i]
            final_samples = Final_samples[i]
            Init_Actual_temp = []
            Init_Difference_temp = []
            Final_Actual_UKF_temp = []
            Final_Difference_UKF_temp = []
            Final_Actual_EKF_temp = []
            Final_Difference_EKF_temp = []
            for j in range(len(methods)):
                # if len(methods) == 1:
                #     py_ukf = PY_UKF[j]
                #     py_ekf = PY_EKF[j]
                #     p = p0
                #     split_method = methods[j]
                #     sample = samples
                #     final_sample = final_samples
                # else:
                #     py_ukf = PY_UKF[j]
                #     py_ekf = PY_EKF[j]
                #     p = p0
                #     split_method = methods[j]
                #     sample = samples
                #     final_sample = final_samples
                
                py_ukf = PY_UKF[j]
                py_ekf = PY_EKF[j]
                p = p0
                split_method = methods[j]
                sample = samples
                final_sample = final_samples
                
                # print(f"analyzing method {split_method}")
                # print('initial')
                # init_actual, init_difference = temp2(p, sample, split_method, expected_percentage)
                # print('ukf')
                final_actual_ukf, final_difference_ukf = temp2(py_ukf, final_sample, split_method, expected_percentage)
                # print('ekf')
                final_actual_ekf, final_difference_ekf = temp2(py_ekf, final_sample, split_method, expected_percentage)
                # Init_Actual_temp.append(init_actual)
                # Init_Difference_temp.append(init_difference)
                Final_Actual_UKF_temp.append(final_actual_ukf)
                Final_Difference_UKF_temp.append(final_difference_ukf)
                Final_Actual_EKF_temp.append(final_actual_ekf)
                Final_Difference_EKF_temp.append(final_difference_ekf)
            Init_Actual.append(Init_Actual_temp)
            Init_Difference.append(Init_Difference_temp)
            Final_Actual_UKF.append(Final_Actual_UKF_temp)
            Final_Difference_UKF.append(Final_Difference_UKF_temp)
            Final_Actual_EKF.append(Final_Actual_EKF_temp)
            Final_Difference_EKF.append(Final_Difference_EKF_temp)
            
        
        # Saving results for each method to seperate file
        
        methods_used = methods_list[0]  # all methods are the same for each number of splits so just use the first one
        print("saving results to file")
        for i in range(len(methods_used)):
            splitstring = "_".join(str(j) for j in split_count if isinstance(j, (int, float)))
            fullstring = methods_used[i] + "_" + splitstring
            Final_act_UKF = []
            Final_diff_UKF = []
            Final_act_EKF =[]
            Final_diff_EKF = []
            for s in range(len(split_count)):
                Final_act_UKF.append(Final_Actual_UKF[s][i])
                Final_diff_UKF.append(Final_Difference_UKF[s][i])
                Final_act_EKF.append(Final_Actual_EKF[s][i])
                Final_diff_EKF.append(Final_Difference_EKF[s][i])
            with open(f'savedresults\{fullstring}.json', 'w') as f:
                json.dump({
                    'Final_Actual_UKF': Final_act_UKF,
                    'Final_Difference_UKF': Final_diff_UKF,
                    'Final_Actual_EKF': Final_act_EKF,
                    'Final_Difference_EKF': Final_diff_EKF,
                    'split_count': [3**s for s in split_count],
                    'methods': methods_used[i],
                    'expected_percentage': expected_percentage.tolist()
                }, f, indent=4)
        
        print("Results saved to savedresults folder")
        
        
    else:
        print("the following files already exist in the savedresults folder:")
        for i in range(len(files)):
            print("(" + str(i) + ") " + files[i])
        run = input('Which file would you like to load? (enter number): ')
        with open(f'savedresults\{files[int(run)]}', 'r') as f:
            Results = json.load(f)
        Final_Actual_UKF = Results['Final_Actual_UKF']
        Final_Difference_UKF = Results['Final_Difference_UKF']
        Final_Actual_EKF = Results['Final_Actual_EKF']
        Final_Difference_EKF = Results['Final_Difference_EKF']
        split_count = Results['split_count']
        methods = Results['methods']
        expected_percentage = Results['expected_percentage']
    
        print("Plotting results as number of splits increases")
        individual = input('Would you like to see individual plots for each confidence interval? (y/n): ')
        
        split_method = methods
        # # UKF plots
        # Fixed percentage, varied number of splits
        plot1 = plt.figure()
        ax1 = plot1.add_subplot(111)   # create an Axes inside the summary figure
        plot1.canvas.manager.set_window_title(f'UKF, {split_method}, Confidence Interval at multiple expected percentages')
        for j in range(len(expected_percentage)):
            Actual = [sublist[j] for sublist in Final_Actual_UKF]
            Difference = [x - expected_percentage[j] for x in Actual]

            if individual == 'y':
                # Individual plot
                fig, ax = plt.subplots()
                fig.canvas.manager.set_window_title(f'UKF, {split_method}, Confidence Interval at {expected_percentage[j]*100:.1f}% expected')
                ax.plot(split_count, Difference, '-o')
                ax.axhline(y=0, color='r', linestyle='--', label='difference = 0')
                ax.set_xlabel('Number of splits')
                ax.set_ylabel('Difference between Actual and Expected')
                ax.set_ylim([min(Difference) - 0.05, max(Difference) + 0.05])
                ax.set_title(
                    split_method
                    + ', UKF, Difference between actual and Confidence Interval at '
                    + str(expected_percentage[j]*100)
                    + '% expected'
                )

            # Plot on the summary figure if condition met
            if j == 0 or j == len(expected_percentage) - 1 or j == len(expected_percentage)//2 or j == len(expected_percentage)//4 or j == 3*len(expected_percentage)//4:
                ax1.plot(split_count, Difference, '-o', label=f'Expected = {expected_percentage[j]*100:.1f}%')
    
        # finalize the summary plot
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Number of Splits')
        ax1.set_ylabel('Difference between Actual and Expected')
        ax1.legend()
        ax1.set_title(f'{split_method}, UKF, Summary of Differences, varying number of splits')
        
        # Fixed number of splits, varied percentage   
        plot2 = plt.figure()
        ax2 = plot2.add_subplot(111)   # create an Axes inside the summary figure
        plot2.canvas.manager.set_window_title(f'UKF, {split_method}, Difference between Actual and Expected at varying split counts')
        for j in range(len(split_count)):
            Difference = Final_Difference_UKF[j]
            ax2.plot(expected_percentage, Difference, '-o', label=f'Split Count = {split_count[j]}')

        # finalize the summary plot
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Expected Percentage')
        ax2.set_ylabel('Difference between Actual and Expected')
        ax2.legend()
        ax2.set_title(f'{split_method}, UKF, Summary of Differences, varying expected percentage')
        
        # # EKF plots
        # Fixed percentage, varied number of splits      
        plot3 = plt.figure()
        ax3 = plot3.add_subplot(111)   # create an Axes inside the summary figure
        plot3.canvas.manager.set_window_title(f'EKF, {split_method}, Confidence Interval at multiple expected percentages')
        for j in range(len(expected_percentage)):
            Actual = [sublist[j] for sublist in Final_Actual_EKF]
            Difference = [x - expected_percentage[j] for x in Actual]
            
            if individual == 'y':
                # Individual plot
                fig, ax = plt.subplots()
                fig.canvas.manager.set_window_title(f'EKF, {split_method}, Confidence Interval at {expected_percentage[j]*100:.1f}% expected')
                ax.plot(split_count, Difference, '-o')
                ax.axhline(y=0, color='r', linestyle='--', label='difference = 0')
                ax.set_xlabel('Number of splits')
                ax.set_ylabel('Difference between Actual and Expected')
                ax.set_ylim([min(Difference) - 0.05, max(Difference) + 0.05])
                ax.set_title(
                    split_method
                    + ', EKF, Difference between actual and Confidence Interval at '
                    + str(expected_percentage[j]*100)
                    + '% expected'
                )

            # Plot on the summary figure if condition met
            if j == 0 or j == len(expected_percentage) - 1 or j == len(expected_percentage)//2 or j == len(expected_percentage)//4 or j == 3*len(expected_percentage)//4:
                ax3.plot(split_count, Difference, '-o', label=f'Expected = {expected_percentage[j]*100:.1f}%')

        # finalize the summary plot
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Number of splits')
        ax3.set_ylabel('Difference between Actual and Expected')
        ax3.legend()
        ax3.set_title(f'{split_method}, EKF, Summary of Differences, varying number of splits')  
        
        # Fixed number of splits, varied percentage
        plot4 = plt.figure()
        ax4 = plot4.add_subplot(111)   # create an Axes inside the summary figure
        plot4.canvas.manager.set_window_title(f'EKF, {split_method}, Difference between Actual and Expected at varying split counts')
        for j in range(len(split_count)):
            Difference = Final_Difference_EKF[j]
            # Plot on the summary figure if condition met
            
            ax4.plot(expected_percentage, Difference, '-o', label=f'Split Count = {split_count[j]}')

        # finalize the summary plot
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('Expected Percentage')
        ax4.set_ylabel('Difference between Actual and Expected')
        ax4.legend()
        ax4.set_title(f'{split_method}, EKF, Summary of Differences, varying expected percentage')

        plt.show()
