import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import scipy
import sigpy as sp


def bin_waveform(resp_in, n_bins, resp_min, resp_max, prominence):
    """bin_waveform

    Args:
        resp_in (Array): _description_
        n_bins (Int): _description_
        resp_min (Float): _description_
        resp_max (Float): _description_
        prominence (Float): _description_

    Raises:
        ValueError: _description_

    Returns:
        x (Array): _description_
    """

    if n_bins % 2:
        raise ValueError(
            f"Number of bins should be even: Current value: {n_bins}!")

    # Assumed normalized
    if resp_max is None:
        resp_max = 1
        print("resp_max assumed to be = ", resp_max)
    if resp_min is None:
        resp_min = 0
        print("resp_min assumed to be = ", resp_min)

    # Copy input data
    resp = copy.deepcopy(resp_in)

    # Interpolate resp to spline
    # Find Peaks and Valleys
    if prominence is None:
        prominence = 2000
        print("prominence assumed to be = ", prominence)
    peak_idx, p_prop = find_peaks(resp, prominence=prominence)
    valley_idx, v_prop = find_peaks(resp * -1, prominence=prominence)

    if peak_idx.size < valley_idx.size:
        valley_idx = valley_idx[:-1]

    resp_smol = resp[0:2000]
    peak_idx_smol, _ = find_peaks(resp_smol, prominence=prominence)
    valley_idx_smol, _ = find_peaks(resp_smol * -1, prominence=prominence)
    plt.plot(resp_smol, "#1f77b4")
    plt.scatter(
        peak_idx_smol, resp_smol[peak_idx_smol], color="red", marker="x", label="peaks")
    plt.scatter(valley_idx_smol,
                resp_smol[valley_idx_smol], color="gold", marker="x", label="valleys")
    plt.legend()
    plt.grid()
    plt.show()
    # plt.close()
    bins = n_bins * np.ones_like(resp)

    # Check if first location is a minima or maxima
    if (peak_idx[0] - valley_idx[0]) < 0:
        s_idx = 0
    else:
        s_idx = 1

    # Find the amplitude between peak and the base (minima)
    min_amp = resp_min
    max_amp = resp_max
    for k in range(valley_idx.size - 1):
        # Exclude if amplitude is too small or too big
        if (
            (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] > min_amp)
            & (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] < max_amp)
            & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] < max_amp)
        ):
            amp_left = resp[peak_idx[k]] - resp[valley_idx[k + s_idx]]
            amp_right = resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]]

            # Find the number of data points between peak and the base (minima)
            n_left = valley_idx[k + s_idx] - peak_idx[k]
            n_right = peak_idx[k + 1] - valley_idx[k + s_idx]

            # Select the area of interest to find the intersection point
            resp_left = resp[peak_idx[k]: peak_idx[k] + n_left]
            resp_right = resp[valley_idx[k + s_idx]
                : valley_idx[k + s_idx] + n_right]

            # Intersection points
            bin_amp = amp_left / (n_bins // 2)  # Bin size for each compartment
            y_left = []
            for b in range(n_bins // 2):
                y_left.append(resp[peak_idx[k]] - (0.5 + b) * bin_amp)

            # Bin size for each compartment
            bin_amp = amp_right / (n_bins // 2)
            y_right = []
            for b in range(n_bins // 2):
                y_right.append(
                    resp[valley_idx[k + s_idx]] + (0.5 + b) * bin_amp)

            # Find the indices for each partition (left)
            n = [peak_idx[k]]
            m = [valley_idx[k + s_idx]]
            for b in range(n_bins // 2):
                n.append(
                    np.argmin(np.abs(resp_left - y_left[b])) + peak_idx[k])
                m.append(
                    np.argmin(np.abs(resp_right - y_right[b])) + valley_idx[k + s_idx])
            n.append(n_left + peak_idx[k])
            m.append(n_right + valley_idx[k + s_idx])
            # Bin assignment
            # Left
            for b in range(1 + n_bins // 2):
                bins[n[b]: n[b + 1]] = b

            # Right
            for b in range(n_bins // 2):
                bins[m[b]: m[b + 1]] = b + (n_bins // 2)
            bins[m[-2]: m[-1]] = 0

    plt.rcParams["figure.figsize"] = (9, 5)
    # plt.rcParams["figure.facecolor"] = "black"
    # plt.rcParams["axes.facecolor"] = "black"
    plt.style.use("dark_background")
    colors = plt.cm.rainbow(np.linspace(0, 1, n_bins))
    plt.gca().set_prop_cycle(color=colors)
    resp_sub = resp[12000:15000]
    bins_sub = bins[12000:15000]
    resp_sub = resp
    bins_sub = bins
    for b in range(n_bins):
        resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
        plt.plot(np.arange(resp_sub.size), resp_array, label=f"Bin {b}")
    resp_array = np.ma.masked_where(bins_sub != n_bins, resp_sub)
    plt.plot(np.arange(resp_sub.size), resp_array,
             label=f"Excluded", color="g")
    plt.legend()
    plt.title("Respiratory Binning")
    plt.xlabel("RF Excitation")
    plt.ylabel("Amplitude")
    plt.show()
    # plt.close()

    # Bin Data
    resp_gated = []
    # print(bins.shape)
    for b in range(n_bins):
        idx = bins == b
        # resp_gated.append(resp_in[idx])
        resp_gated.append(idx)
    return resp_gated


# %% Generate binned data
N_bins = 10

folder = "data/floret-neonatal/"
folder = "data/floret-plummer/"
folder = "data/floret-740H-034/"

# Load motion
motion_load = np.array(np.load(folder + "motion.npy"))
motion_load = np.squeeze(motion_load)
if np.size(np.shape(motion_load)) != 2:
    print('Unexpected motion data dimensions.')
waveform = np.reshape(motion_load, (np.shape(motion_load)[
    0]*np.shape(motion_load)[1]))

# Optional, normalize waveform


def normalize_data(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


waveform_normalized = normalize_data(waveform)

# Smooth motion waveform
sos = scipy.signal.iirfilter(4, Wn=[0.1, 2.5], fs=200, btype="bandpass",
                             ftype="butter", output="sos")
waveform_filt = scipy.signal.sosfilt(sos, waveform)
# waveform_filt = scipy.signal.medfilt(waveform,15) # median filter

# Visualize
fig = plt.figure()
plt.plot(sp.to_device(waveform_filt[:np.shape(waveform_filt)[0]//2], -1))
plt.xlabel('Sample number')
plt.ylabel('Motion')
plt.title('Filtered respiratory bellows motion (first 25% projections only)')
plt.show()

# Find the difference waveform
waveform_filt_diff = np.diff(waveform_filt)

# Make binning function


def quantile_bins(array, num_bins=10):
    """quantile_bins

    Args:
        array (Array): Input array (1xN).
        num_bins (int): Number of bins (must be divisible by two).

    Returns:
        out: (Array) (N_bins x N) Indices of output array
    """
    if num_bins % 2:
        raise ValueError(
            f"Number of bins should be even: Current value: {num_bins}!")

    # Find the difference vector to see if increasing/decreasing
    array_diff = np.diff(array)
    array_diff = np.append(array_diff, 0)

    # Calculate the quantile values
    num_bins_halved = num_bins//2
    quantiles = np.linspace(0, 1, num_bins_halved + 1)
    bin_values = np.quantile(array, quantiles)

    # Allocate array elements to bins
    bins = np.zeros_like(array, dtype=int)
    for i in range(num_bins_halved):
        mask = np.logical_and(array >= bin_values[i], array <= bin_values[i+1])
        mask_decreasing = np.array(mask)
        mask_decreasing[array_diff < 0] = 0
        mask_increasing = np.array(mask)
        mask_increasing[array_diff >= 0] = 0
        bins[mask_decreasing] = i
        bins[mask_increasing] = num_bins - i - 1

    plt.rcParams["figure.figsize"] = (9, 5)
    # plt.rcParams["figure.facecolor"] = "black"
    # plt.rcParams["axes.facecolor"] = "black"
    plt.style.use("dark_background")
    colors = plt.cm.rainbow(np.linspace(0, 1, num_bins))
    plt.gca().set_prop_cycle(color=colors)
    resp_sub = array[12000:15000]
    bins_sub = bins[12000:15000]
    # resp_sub = array
    # bins_sub = bins
    for b in range(num_bins):
        resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
        plt.plot(np.arange(resp_sub.size), resp_array, label=f"Bin {b}")
    resp_array = np.ma.masked_where(bins_sub != num_bins, resp_sub)
    plt.plot(np.arange(resp_sub.size), resp_array,
             label=f"Excluded", color="g")
    plt.legend()
    plt.title("Respiratory Binning")
    plt.xlabel("RF Excitation")
    plt.ylabel("Amplitude")
    plt.show()

    # Assign output data
    out = []
    # print(bins.shape)
    for b in range(num_bins):
        idx = bins == b
        # resp_gated.append(resp_in[idx])
        out.append(idx)

    return out


# Bin data
resp_gated = quantile_bins(waveform_filt, num_bins=N_bins)
print("Number of projections per respiratory bin:")
print(np.sum(resp_gated, axis=1))

# Estimate "goodness of breathing"
range_bins = np.ptp(np.sum(resp_gated, axis=1))
range_norm = range_bins/np.max(np.sum(resp_gated, axis=1))
print("Normalized variability of projections in each bin: " +
      str(np.round(range_norm, 3)))
print("(normalized to max number of projections per bin)")
print("(0 = incredible)")
print("(1 = awful)")

# Subset value to have same number proj in each insp exp
k = np.min(np.sum(resp_gated, axis=1))
print("Number of points per bin selected for use: " + str(k))

# Load data
ksp = np.load(folder + "ksp.npy")
ksp = np.reshape(ksp, (np.shape(ksp)[0], np.shape(ksp)[
                 1]*np.shape(ksp)[2], np.shape(ksp)[3]))
print(np.shape(ksp))
coord = np.load(folder + "coord.npy")
coord = coord.reshape(
    (np.shape(coord)[0]*np.shape(coord)[1], np.shape(coord)[2], np.shape(coord)[3]))
dcf = np.load(folder + "dcf.npy")
dcf = dcf.reshape((np.shape(dcf)[0] * np.shape(dcf)[1], np.shape(dcf)[2]))

# Subset
ksp_save = np.zeros(
    (N_bins, np.shape(ksp)[0], k, np.shape(ksp)[2]), dtype="complex")
coord_save = np.zeros((N_bins, k, np.shape(coord)[1], np.shape(coord)[2]))
dcf_save = np.zeros((N_bins, k,  np.shape(dcf)[1]), dtype="complex")


for gate_number in range(N_bins):
    subset = resp_gated[int(gate_number)]

    # Select only a subset of trajectories and data
    ksp_subset = ksp[:, subset, :]
    ksp_subset = ksp_subset[:, :k, :]
    ksp_save[gate_number, :, :, :] = ksp_subset
    coord_subset = coord[subset, ...]
    coord_subset = coord_subset[:k, ...]
    coord_save[gate_number, ...] = coord_subset
    dcf_subset = dcf[subset, ...]
    dcf_subset = dcf_subset[:k, ...]
    dcf_save[gate_number, ...] = dcf_subset

print(np.shape(ksp_save))
print(np.shape(coord_save))
print(np.shape(dcf_save))
np.save(folder + "bksp.npy", ksp_save)
np.save(folder + "bcoord.npy", coord_save)
np.save(folder + "bdcf.npy", dcf_save)
