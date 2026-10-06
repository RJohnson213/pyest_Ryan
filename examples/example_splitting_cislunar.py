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
from propagation_plots import find_single_point_on_ellipse_6d
from pyest.gm import GaussianMixture, eval_gmpdfchol, eval_gmpdf

from pyest.filters.sigma_points import unscented_transform
import os
from datetime import datetime
from scipy.stats import chi2
import json

# plotting functions
mpl.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
    }
)


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


def temp_funct_HDR(py, samples, alpha, cov_0, x_0):
    point_on_ellipse = find_single_point_on_ellipse_6d(cov_0, x_0, alpha)
    pdf_original = eval_gmpdf(
        np.array(point_on_ellipse).reshape(1, -1),
        np.array([1]),
        x_0.reshape(1, -1),
        cov_0.reshape(1, 6, 6),
    )
    # Function is used to determine number of points that fall within specified bounds
    result_GMM = np.full(len(samples), True)
    measurment_PDF = eval_gmpdfchol(samples, py.w, py.m, py.Schol)
    for i in range(len(samples)):
        if measurment_PDF[i] >= pdf_original:
            result_GMM[i] = True
        else:
            result_GMM[i] = False
    percentage = np.mean(result_GMM)
    return percentage


def temp2(py, samples, split_method, expected_percentage, cov_0, x_0):
    actual_percentage = []
    actual_percentage_HDR = []
    difference = []
    difference_HDR = []
    print("split method: ", split_method)
    for alpha in expected_percentage:
        actual = temp_funct(py, samples, alpha)
        actual_HDR = temp_funct_HDR(py, samples, alpha, cov_0, x_0)
        actual_percentage.append(actual)
        actual_percentage_HDR.append(actual_HDR)
        difference.append(actual - alpha)
        difference_HDR.append(actual_HDR - alpha)
    return actual_percentage, difference, actual_percentage_HDR, difference_HDR


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
    ax.set_title("")
    fig.set_size_inches(w=w, h=h)
    filename = example + split_method.replace(" ", "_")
    filename = folder_name + "/" + filename
    fig.savefig(filename + ".svg", bbox_inches="tight", pad_inches=0)
    ax.set_title(title_text)


def plot_split_and_transformed(
    p_split,
    py,
    split_method_str,
    example,
    dims=(0, 1),
    scatter_means=True,
    xf_lim=None,
    yf_lim=None,
    ax_equal=False,
):
    num_contours = 100
    scatter_plt_args = {"marker": "x", "zorder": 2, "color": "k"}
    scatter_plt_overlay_args = {
        "s": 5**2,
        "marker": "x",
        "zorder": 2.1,
        "color": "w",
        "alpha": 0.9,
        "linewidth": 1,
    }
    # Plot the split density
    pp, XX, YY = p_split.pdf_2d(res=300, dimensions=dims)
    plt.figure()
    plt.contour(XX, YY, pp, num_contours)
    plt.title("Original Density, split,  " + split_method_str, wrap=True)
    plt.colorbar()
    if scatter_means:
        plt.scatter(p_split.m[:, dims[0]], p_split.m[:, dims[1]], **scatter_plt_args)
        plt.scatter(
            p_split.m[:, dims[0]], p_split.m[:, dims[1]], **scatter_plt_overlay_args
        )
    plt.grid()
    labels = ["$x$", "$y$", "$z$", r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"]
    plt.xlabel(labels[dims[0]])
    plt.ylabel(labels[dims[1]])

    save_figure(
        example, split_method_str + "before_map" + str(dims), plt.gca(), plt.gcf()
    )

    # Plot the transformed split density
    pp, XX, YY = py.pdf_2d(res=300, dimensions=dims, xbnd=xf_lim, ybnd=yf_lim)
    fig, ax = plt.subplots()
    c = ax.contour(XX, YY, pp, num_contours, linewidths=0.5)
    fig.colorbar(c)
    ax.set_title("Transformed Density, " + split_method_str, wrap=True)
    if scatter_means:
        ax.scatter(py.m[:, dims[0]], py.m[:, dims[1]], **scatter_plt_args)
        ax.scatter(py.m[:, dims[0]], py.m[:, dims[1]], **scatter_plt_overlay_args)
    ax.grid()
    if xf_lim is not None:
        ax.set_xlim(xf_lim)
    if yf_lim is not None:
        ax.set_ylim(yf_lim)
    if ax_equal:
        ax.set_aspect("equal", adjustable="box")

    labels = ["$x$", "$y$", "$z$", r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"]
    ax.set_xlabel(labels[dims[0]])
    ax.set_ylabel(labels[dims[1]])

    save_figure(
        example,
        split_method_str + "_" + str(dims[0]) + "_" + str(dims[1]),
        plt.gca(),
        plt.gcf(),
    )

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
    return GaussianMixture(wy, my, Sy, cov_type="cholesky")


def transform_density_ukf(
    p_split, ny, g, sigma_pt_opts, residual_fun=None, mean_fun=None
):
    # Compute the unscented transform of the Gaussian mixture
    my = np.zeros((len(p_split), ny))
    Sy = np.zeros((len(p_split), ny, ny))
    for i in range(len(p_split)):
        # padding for non-positive definiteness issue with unscented transform of very small covariances
        my[i], Sy[i], _, _, _ = unscented_transform(
            p_split.m[i],
            p_split.Schol[i],
            g,
            sigma_pt_opts=sigma_pt_opts,
            cov_type="cholesky",
            residual_fun=residual_fun,
            mean_fun=mean_fun,
        )
    wy = p_split.w.copy()
    return GaussianMixture(wy, my, Sy, cov_type="cholesky")


# density propagation example in a Cislunar NRHO
def cislunar_example(split_count):
    example = "cislunar"
    # nrho ics
    mu = 1.0 / (81.30059 + 1.0)
    x0 = 1.02202151273581740824714855590570360
    z0 = -0.182096761524240501132977765539282777
    yd0 = -0.103256341062793815791764364248006121
    period = 1.5111111111111111111111111111111111111111
    transfer_time = period * 0.5
    split_time = transfer_time

    x_0 = np.array([x0, 0, z0, 0, yd0, 0])
    ny = 6  # dimension of y
    nx = 6  # dimension of x

    weights = np.array([1])  # single component
    pos_var = (10) ** 2  # meters^2
    vel_var = (0.01) ** 2  # (m/s)^2
    # convert to km, km/s
    pos_uncertainty_km = pos_var / (1000**2)
    vel_uncertainty_kms = vel_var / (1000**2)
    # nondimensionalization
    pos_scale = 3.850e5  # velocity nondimensionalization scaling from Koon Lo Marsden Ross table
    vel_scale = (
        1.025  # velocity nondimensionalization scaling from Koon Lo Marsden Ross table
    )
    temp = [pos_uncertainty_km / (pos_scale**2)] * 3 + [
        vel_uncertainty_kms / (vel_scale**2)
    ] * 3
    cov_0 = np.diag(temp)
    # cov_0 = 0.00001**2 * np.identity(6) + 0.0001**2 * (np.diag([1, 0, 1, 0, 0, 0]))
    p0 = GaussianMixture(weights, np.array([x_0]), np.array([cov_0]))

    # nrho propagator
    integrator = STMint(preset="threeBody", preset_mult=mu, variational_order=2)
    max_integrator_step = period / 2000.0
    int_tol = 1e-16

    # outputs x_f, STM, STT
    def flow_info(x, y, z, vx, vy, vz):
        return integrator.dynVar_int2(
            [0, transfer_time],
            [x, y, z, vx, vy, vz],
            rtol=int_tol,
            atol=int_tol,
            output="final",
        )

    # outputs just the hessian
    def hessian_func(x, y, z, vx, vy, vz):
        return integrator.dynVar_int2(
            [0, transfer_time],
            [x, y, z, vx, vy, vz],
            rtol=int_tol,
            atol=int_tol,
            output="final",
        )[2]

    # outputs just the jacobian
    def jacobian_func(x, y, z, vx, vy, vz):
        return integrator.dynVar_int(
            [0, transfer_time],
            [x, y, z, vx, vy, vz],
            rtol=int_tol,
            atol=int_tol,
            output="final",
        )[1]

    # outputs flow of state only

    def propagation(x_0):
        return integrator.dyn_int(
            [0, transfer_time],
            x_0,
            max_step=max_integrator_step,
            t_eval=[transfer_time],
            rtol=int_tol,
            atol=int_tol,
        ).y[:, -1]

    # apply splitting methods
    split_opts = pygm.GaussSplitOptions(
        L=3, lam=1e-3, recurse_depth=split_count, min_weight=-np.inf
    )  # L^recurse_depth

    # Define the unscented transform parameters
    sigma_pt_opts = SigmaPointOptions(alpha=1e-3, beta=2, kappa=0)

    print("running monte carlo")
    # create/load split cache
    cislunar_mc_cache = Cache(__file__[:-3] + "cislunar_mc_cache")
    # reference Monte Carlo (store points and pdf value at point)
    num_points = int(1e4)
    rng = np.random.default_rng(100)
    if "samples" in cislunar_mc_cache:
        print("cache found, loading samples from cache")
        samples = cislunar_mc_cache["samples"]
        final_samples = cislunar_mc_cache["final_samples"]
        assert len(samples) == num_points
    else:
        print("cache not found, propagating samples")
        samples = rng.multivariate_normal(x_0, cov_0, num_points)

        final_samples = []
        for i, s in enumerate(samples):
            final_samples.append(propagation(s))

            print(f"Propagated {i}")

        cislunar_mc_cache["samples"] = samples
        cislunar_mc_cache["final_samples"] = final_samples

    recursive_split_args = {}
    # use the same number of recursive splits for each mixand
    split_tol = -np.inf
    # settings for the SADL and ALoDT based metrics
    diff_stat_det_sigma_pt_opts = SigmaPointOptions(
        alpha=0.5
    )  # spread sigma points farther

    recursive_split_args["WUSSOLC"] = (
        split.id_wussolc,
        hessian_func,
        jacobian_func,
        split_tol,
    )

    PY_UKF = []
    PY_EKF = []
    methods = []

    # plot the resulting GMM densities propagated
    for split_method, args in recursive_split_args.items():
        # 1. NEW DEDICATED FUNCTIONS FOR SPLITTING (Integrating out to split_time)
        def split_jacobian(x, y, z, vx, vy, vz):
            return integrator.dynVar_int(
                [0, split_time],  # <-- Forces integration to split_time
                [x, y, z, vx, vy, vz],
                rtol=int_tol,
                atol=int_tol,
                output="final",
            )[1]

        def split_hessian(x, y, z, vx, vy, vz):
            return integrator.dynVar_int2(
                [0, split_time],  # <-- Forces integration to split_time
                [x, y, z, vx, vy, vz],
                rtol=int_tol,
                atol=int_tol,
                output="final",
            )[2]

        # Re-pack the arguments for the splitting module using our new split_time functions
        if split_method == "WUSSOLC":
            # WUSSOLC expects: (split.id_wussolc, hessian_func, jacobian_func, split_tol)
            split_args = (args[0], split_hessian, split_jacobian, args[3])
        else:
            split_args = args

        p_split_ekf = split.recursive_split(p0, split_opts, *split_args)
        p_split_ukf = split.recursive_split(p0, split_opts, *split_args)
        # py = transform_density_ekf(p_split, ny, propagation, jacobian_func)

        py_ukf = transform_density_ukf(p_split_ukf, ny, propagation, sigma_pt_opts)
        py_ekf = transform_density_ekf(p_split_ekf, ny, propagation, jacobian_func)

        py = py_ukf

        # for idx_pair in idx_pairs:
        #     _, XX, YY = plot_split_and_transformed(
        #         p_split, py, split_method, example, idx_pair, xf_lim=xlim[idx_pair], yf_lim=ylim[idx_pair])

        PY_UKF.append(py_ukf)
        PY_EKF.append(py_ekf)
        methods.append(split_method)

    # plt.show()
    return PY_UKF, PY_EKF, methods, samples, final_samples, p0, cov_0, x_0


if __name__ == "__main__":
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
    run = input("would you like to generate new results? (y/n): ")
    if run == "y" or len(files) == 0:
        print("running cislunar example")

        # These lists will store results for different numbers of splits. each sublist corresponds to a different number of splits.
        # Each element of the sublist corresponds to a different splitting method. P0, Samples and Final samples are the same for each splitting method so they are not nested lists
        UKF = []
        EKF = []
        P0 = []
        methods_list = []
        Samples = []
        Final_samples = []

        split_count = [0, 1, 2, 3, 4, 5, 6, 7]  # number of splits is 3**split_count;

        for i in range(len(split_count)):
            split_c = split_count[i]
            PY_UKF, PY_EKF, methods, samples, final_samples, p0, cov_0, x_0 = (
                cislunar_example(split_c)
            )
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
        Init_Actual_HDR = []
        Init_Difference_HDR = []
        Final_Actual_UKF_HDR = []
        Final_Difference_UKF_HDR = []
        Final_Actual_EKF_HDR = []
        Final_Difference_EKF_HDR = []
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
            Init_Actual_temp_HDR = []
            Init_Difference_temp_HDR = []
            Final_Actual_UKF_temp_HDR = []
            Final_Difference_UKF_temp_HDR = []
            Final_Actual_EKF_temp_HDR = []
            Final_Difference_EKF_temp_HDR = []
            for j in range(len(methods)):
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
                (
                    final_actual_ukf,
                    final_difference_ukf,
                    final_actual_ukf_HDR,
                    final_difference_ukf_HDR,
                ) = temp2(
                    py_ukf, final_sample, split_method, expected_percentage, cov_0, x_0
                )
                # print('ekf')
                (
                    final_actual_ekf,
                    final_difference_ekf,
                    final_actual_ekf_HDR,
                    final_difference_ekf_HDR,
                ) = temp2(
                    py_ekf, final_sample, split_method, expected_percentage, cov_0, x_0
                )
                # Init_Actual_temp.append(init_actual)
                # Init_Difference_temp.append(init_difference)
                Final_Actual_UKF_temp.append(final_actual_ukf)
                Final_Actual_UKF_temp_HDR.append(final_actual_ukf_HDR)
                Final_Difference_UKF_temp.append(final_difference_ukf)
                Final_Difference_UKF_temp_HDR.append(final_difference_ukf_HDR)
                Final_Actual_EKF_temp.append(final_actual_ekf)
                Final_Actual_EKF_temp_HDR.append(final_actual_ekf_HDR)
                Final_Difference_EKF_temp.append(final_difference_ekf)
                Final_Difference_EKF_temp_HDR.append(final_difference_ekf_HDR)
            Init_Actual.append(Init_Actual_temp)
            Init_Actual_HDR.append(Init_Actual_temp_HDR)
            Init_Difference.append(Init_Difference_temp)
            Init_Difference_HDR.append(Init_Difference_temp_HDR)
            Final_Actual_UKF.append(Final_Actual_UKF_temp)
            Final_Actual_UKF_HDR.append(Final_Actual_UKF_temp_HDR)
            Final_Difference_UKF.append(Final_Difference_UKF_temp)
            Final_Difference_UKF_HDR.append(Final_Difference_UKF_temp_HDR)
            Final_Actual_EKF.append(Final_Actual_EKF_temp)
            Final_Actual_EKF_HDR.append(Final_Actual_EKF_temp_HDR)
            Final_Difference_EKF.append(Final_Difference_EKF_temp)
            Final_Difference_EKF_HDR.append(Final_Difference_EKF_temp_HDR)

        # Saving results for each method to seperate file

        methods_used = methods_list[
            0
        ]  # all methods are the same for each number of splits so just use the first one
        print("saving results to file")
        for i in range(len(methods_used)):
            splitstring = "_".join(
                str(j) for j in split_count if isinstance(j, (int, float))
            )
            fullstring = methods_used[i] + "_" + splitstring
            Final_act_UKF = []
            Final_diff_UKF = []
            Final_act_EKF = []
            Final_diff_EKF = []
            Final_act_UKF_HDR = []
            Final_diff_UKF_HDR = []
            Final_act_EKF_HDR = []
            Final_diff_EKF_HDR = []
            for s in range(len(split_count)):
                Final_act_UKF.append(Final_Actual_UKF[s][i])
                Final_diff_UKF.append(Final_Difference_UKF[s][i])
                Final_act_EKF.append(Final_Actual_EKF[s][i])
                Final_diff_EKF.append(Final_Difference_EKF[s][i])
                Final_act_UKF_HDR.append(Final_Actual_UKF_HDR[s][i])
                Final_diff_UKF_HDR.append(Final_Difference_UKF_HDR[s][i])
                Final_act_EKF_HDR.append(Final_Actual_EKF_HDR[s][i])
                Final_diff_EKF_HDR.append(Final_Difference_EKF_HDR[s][i])
            with open(f"savedresults\{fullstring}.json", "w") as f:
                json.dump(
                    {
                        "Final_Actual_UKF": Final_act_UKF,
                        "Final_Difference_UKF": Final_diff_UKF,
                        "Final_Actual_EKF": Final_act_EKF,
                        "Final_Difference_EKF": Final_diff_EKF,
                        "split_count": [3**s for s in split_count],
                        "methods": methods_used[i],
                        "expected_percentage": expected_percentage.tolist(),
                        "Final_Actual_UKF_HDR": Final_act_UKF_HDR,
                        "Final_Difference_UKF_HDR": Final_diff_UKF_HDR,
                        "Final_Actual_EKF_HDR": Final_act_EKF_HDR,
                        "Final_Difference_EKF_HDR": Final_diff_EKF_HDR,
                    },
                    f,
                    indent=4,
                )

        print("Results saved to savedresults folder")

    else:
        print("the following files already exist in the savedresults folder:")
        for i in range(len(files)):
            print("(" + str(i) + ") " + files[i])
        run = input("Which file would you like to load? (enter number): ")
        with open(f"savedresults\{files[int(run)]}", "r") as f:
            Results = json.load(f)
        Final_Actual_UKF = Results["Final_Actual_UKF"]
        Final_Difference_UKF = Results["Final_Difference_UKF"]
        Final_Actual_EKF = Results["Final_Actual_EKF"]
        Final_Difference_EKF = Results["Final_Difference_EKF"]
        Final_Actual_UKF_HDR = Results["Final_Actual_UKF_HDR"]
        Final_Difference_UKF_HDR = Results["Final_Difference_UKF_HDR"]
        Final_Actual_EKF_HDR = Results["Final_Actual_EKF_HDR"]
        Final_Difference_EKF_HDR = Results["Final_Difference_EKF_HDR"]
        split_count = Results["split_count"]
        methods = Results["methods"]
        expected_percentage = Results["expected_percentage"]

        print("Plotting results as number of splits increases")
        individual = input(
            "Would you like to see individual plots for each confidence interval? (y/n): "
        )

        split_method = methods
        # # UKF plots
        # Fixed percentage, varied number of splits
        plot1 = plt.figure()
        ax1 = plot1.add_subplot(111)  # create an Axes inside the summary figure
        plot1.canvas.manager.set_window_title(
            f"UKF, {split_method}, Confidence Interval at multiple expected percentages"
        )
        for j in range(len(expected_percentage)):
            Actual = [sublist[j] for sublist in Final_Actual_UKF]
            Difference = [(x - expected_percentage[j]) * 100 for x in Actual]

            if individual == "y":
                # Individual plot
                fig, ax = plt.subplots()
                fig.canvas.manager.set_window_title(
                    f"UKF, {split_method}, Confidence Interval at {expected_percentage[j] * 100:.1f}% expected"
                )
                ax.plot(split_count, Difference, "-o")
                ax.axhline(y=0, color="r", linestyle="--", label="difference = 0")
                ax.set_xlabel("Number of splits UKF")
                ax.set_ylabel("Difference between Actual and Expected")
                ax.set_ylim([min(Difference) - 0.05, max(Difference) + 0.05])
                ax.set_title(
                    split_method
                    + ", UKF, Difference between actual and Confidence Interval at "
                    + str(expected_percentage[j] * 100)
                    + "% expected"
                )

            # Plot on the summary figure if condition met
            if (
                j == 0
                or j == len(expected_percentage) - 1
                or j == len(expected_percentage) // 2
                or j == len(expected_percentage) // 4
                or j == 3 * len(expected_percentage) // 4
            ):
                ax1.plot(
                    split_count,
                    Difference,
                    "-o",
                    label=f"Expected = {expected_percentage[j] * 100:.1f}%",
                )

        # finalize the summary plot
        ax1.axhline(y=0, color="r", linestyle="--")
        ax1.set_xlabel("Number of Splits UKF", fontsize=16)
        ax1.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax1.legend(fontsize=16)
        ax1.tick_params(axis="both", labelsize=16)
        # ax1.set_title(f'{split_method}, UKF, Summary of Differences, varying number of splits')

        # Fixed number of splits, varied percentage
        plot2 = plt.figure()
        ax2 = plot2.add_subplot(111)  # create an Axes inside the summary figure
        plot2.canvas.manager.set_window_title(
            f"UKF, {split_method}, Difference between Actual and Expected at varying split counts"
        )
        for j in range(len(split_count)):
            Difference = Final_Difference_UKF[j]
            ax2.plot(
                expected_percentage,
                Difference,
                "-o",
                label=f"Split Count = {split_count[j]}",
            )

        # finalize the summary plot
        ax2.axhline(y=0, color="r", linestyle="--")
        ax2.set_xlabel("Expected Percentage", fontsize=16)
        ax2.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax2.legend(fontsize=16)
        ax2.tick_params(axis="both", labelsize=16)
        # ax2.set_title(f'{split_method}, UKF, Summary of Differences, varying expected percentage')

        # # EKF plots
        # Fixed percentage, varied number of splits
        plot3 = plt.figure()
        ax3 = plot3.add_subplot(111)  # create an Axes inside the summary figure
        plot3.canvas.manager.set_window_title(
            f"EKF, {split_method}, Confidence Interval at multiple expected percentages"
        )
        for j in range(len(expected_percentage)):
            Actual = [sublist[j] for sublist in Final_Actual_EKF]
            Difference = [(x - expected_percentage[j]) * 100 for x in Actual]

            if individual == "y":
                # Individual plot
                fig, ax = plt.subplots()
                fig.canvas.manager.set_window_title(
                    f"EKF, {split_method}, Confidence Interval at {expected_percentage[j] * 100:.1f}% expected"
                )
                ax.plot(split_count, Difference, "-o")
                ax.axhline(y=0, color="r", linestyle="--", label="difference = 0")
                ax.set_xlabel("Number of splits EKF")
                ax.set_ylabel("Difference between Actual and Expected")
                ax.set_ylim([min(Difference) - 0.05, max(Difference) + 0.05])
                ax.set_title(
                    split_method
                    + ", EKF, Difference between actual and Confidence Interval at "
                    + str(expected_percentage[j] * 100)
                    + "% expected"
                )

            # Plot on the summary figure if condition met
            if (
                j == 0
                or j == len(expected_percentage) - 1
                or j == len(expected_percentage) // 2
                or j == len(expected_percentage) // 4
                or j == 3 * len(expected_percentage) // 4
            ):
                ax3.plot(
                    split_count,
                    Difference,
                    "-o",
                    label=f"Expected = {expected_percentage[j] * 100:.1f}%",
                )

        # finalize the summary plot
        ax3.axhline(y=0, color="r", linestyle="--")
        ax3.set_xlabel("Number of splits EKF", fontsize=16)
        ax3.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax3.legend(fontsize=16)
        ax3.tick_params(axis="both", labelsize=16)
        # ax3.set_title(f'{split_method}, EKF, Summary of Differences, varying number of splits')

        # Fixed number of splits, varied percentage
        plot4 = plt.figure()
        ax4 = plot4.add_subplot(111)  # create an Axes inside the summary figure
        plot4.canvas.manager.set_window_title(
            f"EKF, {split_method}, Difference between Actual and Expected at varying split counts"
        )
        for j in range(len(split_count)):
            Difference = Final_Difference_EKF[j]
            # Plot on the summary figure if condition met

            ax4.plot(
                expected_percentage,
                Difference,
                "-o",
                label=f"Split Count = {split_count[j]}",
            )

        # finalize the summary plot
        ax4.axhline(y=0, color="r", linestyle="--")
        ax4.set_xlabel("Expected Percentage", fontsize=16)
        ax4.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax4.legend(fontsize=16)
        ax4.tick_params(axis="both", labelsize=16)
        # ax4.set_title(f'{split_method}, EKF, Summary of Differences, varying expected percentage')

        # # HDR EKF plots
        # Fixed percentage, varied number of splits
        plot5 = plt.figure()
        ax5 = plot5.add_subplot(111)  # create an Axes inside the summary figure
        plot5.canvas.manager.set_window_title(
            f"EKF HDR, {split_method}, Confidence Interval at multiple expected percentages"
        )
        for j in range(len(expected_percentage)):
            Actual = [sublist[j] for sublist in Final_Actual_EKF_HDR]
            Difference = [(x - expected_percentage[j]) * 100 for x in Actual]

            if individual == "y":
                # Individual plot
                fig, ax = plt.subplots()
                fig.canvas.manager.set_window_title(
                    f"EKF HDR, {split_method}, Confidence Interval at {expected_percentage[j] * 100:.1f}% expected"
                )
                ax.plot(split_count, Difference, "-o")
                ax.axhline(y=0, color="r", linestyle="--", label="difference = 0")
                ax.set_xlabel("Number of splits EKF HDR")
                ax.set_ylabel("Difference between Actual and Expected")
                ax.set_ylim([min(Difference) - 0.05, max(Difference) + 0.05])
                ax.set_title(
                    split_method
                    + ", EKF HDR, Difference between actual and Confidence Interval at "
                    + str(expected_percentage[j] * 100)
                    + "% expected"
                )

            # Plot on the summary figure if condition met
            if (
                j == 0
                or j == len(expected_percentage) - 1
                or j == len(expected_percentage) // 2
                or j == len(expected_percentage) // 4
                or j == 3 * len(expected_percentage) // 4
            ):
                ax5.plot(
                    split_count,
                    Difference,
                    "-o",
                    label=f"Expected = {expected_percentage[j] * 100:.1f}%",
                )

        # finalize the summary plot
        ax5.axhline(y=0, color="r", linestyle="--")
        ax5.set_xlabel("Number of splits EKF", fontsize=16)
        ax5.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax5.legend(fontsize=16)
        ax5.tick_params(axis="both", labelsize=16)
        # ax3.set_title(f'{split_method}, EKF, Summary of Differences, varying number of splits')

        # Fixed number of splits, varied percentage
        plot6 = plt.figure()
        ax6 = plot6.add_subplot(111)  # create an Axes inside the summary figure
        plot6.canvas.manager.set_window_title(
            f"EKF HDR, {split_method}, Difference between Actual and Expected at varying split counts"
        )
        for j in range(len(split_count)):
            Difference = Final_Difference_EKF_HDR[j]
            # Plot on the summary figure if condition met

            ax6.plot(
                expected_percentage,
                Difference,
                "-o",
                label=f"Split Count = {split_count[j]}",
            )

        # finalize the summary plot
        ax6.axhline(y=0, color="r", linestyle="--")
        ax6.set_xlabel("Expected Percentage", fontsize=16)
        ax6.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax6.legend(fontsize=16)
        ax6.tick_params(axis="both", labelsize=16)

        # # HDR UKF plots
        # Fixed percentage, varied number of splits
        plot7 = plt.figure()
        ax7 = plot7.add_subplot(111)  # create an Axes inside the summary figure
        plot7.canvas.manager.set_window_title(
            f"UKF HDR, {split_method}, Confidence Interval at multiple expected percentages"
        )
        for j in range(len(expected_percentage)):
            Actual = [sublist[j] for sublist in Final_Actual_UKF_HDR]
            Difference = [(x - expected_percentage[j]) * 100 for x in Actual]

            if individual == "y":
                # Individual plot
                fig, ax = plt.subplots()
                fig.canvas.manager.set_window_title(
                    f"UKF HDR, {split_method}, Confidence Interval at {expected_percentage[j] * 100:.1f}% expected"
                )
                ax.plot(split_count, Difference, "-o")
                ax.axhline(y=0, color="r", linestyle="--", label="difference = 0")
                ax.set_xlabel("Number of splits UKF HDR")
                ax.set_ylabel("Difference between Actual and Expected")
                ax.set_ylim([min(Difference) - 0.05, max(Difference) + 0.05])
                ax.set_title(
                    split_method
                    + ", UKF HDR, Difference between actual and Confidence Interval at "
                    + str(expected_percentage[j] * 100)
                    + "% expected"
                )

            # Plot on the summary figure if condition met
            if (
                j == 0
                or j == len(expected_percentage) - 1
                or j == len(expected_percentage) // 2
                or j == len(expected_percentage) // 4
                or j == 3 * len(expected_percentage) // 4
            ):
                ax7.plot(
                    split_count,
                    Difference,
                    "-o",
                    label=f"Expected = {expected_percentage[j] * 100:.1f}%",
                )

        # finalize the summary plot
        ax7.axhline(y=0, color="r", linestyle="--")
        ax7.set_xlabel("Number of splits UKF HDR", fontsize=16)
        ax7.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax7.legend(fontsize=16)
        ax7.tick_params(axis="both", labelsize=16)

        # Fixed number of splits, varied percentage
        plot8 = plt.figure()
        ax8 = plot8.add_subplot(111)  # create an Axes inside the summary figure
        plot8.canvas.manager.set_window_title(
            f"UKF HDR, {split_method}, Difference between Actual and Expected at varying split counts"
        )
        for j in range(len(split_count)):
            Difference = Final_Difference_UKF_HDR[j]
            # Plot on the summary figure if condition met

            ax8.plot(
                expected_percentage,
                Difference,
                "-o",
                label=f"Split Count = {split_count[j]}",
            )

        # finalize the summary plot
        ax8.axhline(y=0, color="r", linestyle="--")
        ax8.set_xlabel("Expected Percentage", fontsize=16)
        ax8.set_ylabel("Difference (Actual - Expected)", fontsize=16)
        ax8.legend(fontsize=16)
        ax8.tick_params(axis="both", labelsize=16)

        # --- 99% Association Rate Plot (UKF vs EKF) ---
        plot9 = plt.figure()
        ax9 = plot9.add_subplot(111)
        plot9.canvas.manager.set_window_title(
            f"99% Association Rate vs Number of Splits, {split_method}"
        )

        # Index -1 corresponds to 0.99 in expected_percentage
        ukf_99_actual = [sublist[-1] * 100 for sublist in Final_Actual_UKF]
        ekf_99_actual = [sublist[-1] * 100 for sublist in Final_Actual_EKF]

        ax9.plot(split_count, ukf_99_actual, "-o", label="UKF Individual Mixand")
        ax9.plot(split_count, ekf_99_actual, "-s", label="EKF Individual Mixand")

        ukf_hdr_99_actual = [sublist[-1] * 100 for sublist in Final_Actual_UKF_HDR]
        ekf_hdr_99_actual = [sublist[-1] * 100 for sublist in Final_Actual_EKF_HDR]
        ax9.plot(split_count, ukf_hdr_99_actual, "-^", label="UKF HDR")
        ax9.plot(split_count, ekf_hdr_99_actual, "-d", label="EKF HDR")

        # Reference line for the 99% target
        ax9.axhline(y=99.0, color="r", linestyle="--", label="Expected 98.8\%")

        ax9.set_xlabel("Number of Splits", fontsize=16)
        ax9.set_ylabel("Association Rate (\%)", fontsize=16)
        ax9.legend(fontsize=16)
        ax9.tick_params(axis="both", labelsize=16)

        plt.show()
