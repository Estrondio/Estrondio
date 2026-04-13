import os
import re
import csv
import h5py
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from scipy.signal import firwin, filtfilt, find_peaks

# SETTINGS / GLOBAL VARIABLES
FALLBACK_FS = 1000.0
SCORE_EPOCH_SEC = 4.0

WAKE_CHAR = "w"
NREM_CHAR = "n"
REM_CHAR = "r"
TASK_CHAR = "f"

MIN_FRACTION_NREM = 0.8
MIN_CONSOLIDATED_N_EPOCHS = 3

SPINDLE_LOW_HZ = 9.0
SPINDLE_HIGH_HZ = 16.0
FIR_NUMTAPS = 2001
MERGE_GAP_SEC = 0.05

ZT2_SEC = 2 * 3600
VERBOSE = True


def vlog(*args):
    if VERBOSE:
        print("[spindle_detection]", *args)

# GUI FUNCTIONS / Tkinter

def ask_user_for_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing .mat files")
    root.destroy()
    vlog("Selected folder:", folder)
    return folder


def ask_user_for_output_csv():
    root = tk.Tk()
    root.withdraw()
    out_path = filedialog.asksaveasfilename(
        title="Save spindle table as CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()
    vlog("Selected output CSV:", out_path)
    return out_path


def ask_user_for_channel_index():
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askinteger(
        "Channel selection",
        "Enter the 0-based channel index to analyze for all files.\n\n"
        "Examples:\n"
        "0 = first channel\n"
        "7 = eighth channel",
        minvalue=0
    )
    root.destroy()
    vlog("Selected channel index:", value)
    return value


def ask_user_for_window_hours():
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askfloat(
        "Window time",
        "Enter window duration in hours of TOTAL NREM to accumulate (e.g., 6):",
        minvalue=0.01
    )
    root.destroy()
    vlog("Selected window hours:", value)
    return value


def ask_user_for_window_mode():
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askstring(
        "Window anchor",
        "Choose window anchor mode:\n"
        "1 = after_last_f_or_ZT2\n"
        "2 = ZT2_only\n\n"
        "Type 1 or 2:"
    )
    root.destroy()

    if value is None:
        return None
    value = value.strip()
    if value == "1":
        vlog("Selected window mode: after_last_f_or_ZT2")
        return "after_last_f_or_ZT2"
    if value == "2":
        vlog("Selected window mode: ZT2_only")
        return "ZT2_only"
    vlog("Invalid window mode selection:", value)
    return None


def show_info_message(title, text):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(title, text)
    root.destroy()


def show_error_message(title, text):
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, text)
    root.destroy()


# DATA EXTRACTION

def try_extract_scalar(dataset_or_array):
    try:
        arr = np.array(dataset_or_array)
        arr = np.squeeze(arr)
        if arr.size == 1:
            return float(arr)
    except Exception:
        pass
    return None


def try_extract_fs_from_infos(f, fallback_fs):
    if "Infos" not in f:
        return fallback_fs

    infos_obj = f["Infos"]
    possible_keys = ["fs", "Fs", "FS", "sampling_rate", "SamplingRate", "sr", "SR", "freq", "Freq"]

    if isinstance(infos_obj, h5py.Group):
        for key in possible_keys:
            if key in infos_obj:
                value = try_extract_scalar(infos_obj[key][()])
                if value is not None and value > 0:
                    vlog("Sampling rate found in Infos key", key, "=", value)
                    return value

    if isinstance(infos_obj, h5py.Dataset):
        value = try_extract_scalar(infos_obj[()])
        if value is not None and value > 0:
            vlog("Sampling rate found in Infos dataset =", value)
            return value

    vlog("Sampling rate not found in Infos, using fallback:", fallback_fs)
    return fallback_fs


def decode_char_array_from_hdf5_dataset(ds):
    arr = np.array(ds[()])

    if np.issubdtype(arr.dtype, np.integer):
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            return "".join(chr(int(v)) for v in arr)
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                return "".join(chr(int(v)) for v in arr[0])
            if arr.shape[1] == 1:
                return "".join(chr(int(v)) for v in arr[:, 0])
            return "".join(chr(int(v)) for v in arr.ravel())

    flat = arr.ravel()
    chars = []
    for v in flat:
        if isinstance(v, bytes):
            chars.append(v.decode("utf-8"))
        else:
            chars.append(str(v))
    return "".join(chars)


def expand_b_to_sample_hypnogram(b_string, fs, n_samples, score_epoch_sec):
    samples_per_epoch = int(round(score_epoch_sec * fs))
    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be > 0")

    hyp_chars = np.repeat(list(b_string), samples_per_epoch)

    if len(hyp_chars) < n_samples:
        last_val = hyp_chars[-1]
        hyp_chars = np.concatenate([hyp_chars, np.array([last_val] * (n_samples - len(hyp_chars)), dtype=hyp_chars.dtype)])

    return hyp_chars[:n_samples]


def replace_nan_with_zero(x):
    x = np.array(x, dtype=float).ravel()
    x = x - np.nanmean(x)
    x[np.isnan(x)] = 0.0
    return x


def extract_hypnogram_text(f):
    if "b" not in f:
        raise ValueError("No 'b' variable found in the .mat file.")
    b_string = decode_char_array_from_hdf5_dataset(f["b"])
    b_string = b_string.replace("\x00", "").strip().lower()
    if len(b_string) == 0:
        raise ValueError("Decoded 'b' is empty.")
    return b_string


def load_trace_and_hypnogram(mat_file, channel_index, fallback_fs):
    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"File not found: {mat_file}")

    vlog("Loading file:", mat_file)
    with h5py.File(mat_file, "r") as f:
        if "traces" not in f:
            raise KeyError("This file does not contain 'traces'.")

        traces = f["traces"]
        if len(traces.shape) != 2:
            raise ValueError(f"'traces' has unexpected shape: {traces.shape}")

        dim0, dim1 = traces.shape
        fs = try_extract_fs_from_infos(f, fallback_fs)

        vlog("traces shape:", traces.shape)
        if dim0 <= 64 and dim1 > dim0:
            n_channels = dim0
            n_samples = dim1
            if channel_index < 0 or channel_index >= n_channels:
                raise ValueError(f"channel_index must be between 0 and {n_channels - 1}")
            x = np.array(traces[channel_index, :], dtype=float).ravel()
            vlog("Orientation inferred as channels x samples. n_channels =", n_channels, "n_samples =", n_samples)
        elif dim1 <= 64 and dim0 > dim1:
            n_samples = dim0
            n_channels = dim1
            if channel_index < 0 or channel_index >= n_channels:
                raise ValueError(f"channel_index must be between 0 and {n_channels - 1}")
            x = np.array(traces[:, channel_index], dtype=float).ravel()
            vlog("Orientation inferred as samples x channels. n_channels =", n_channels, "n_samples =", n_samples)
        else:
            raise ValueError(f"Could not infer channel/sample orientation from shape {traces.shape}")

        b_string = extract_hypnogram_text(f)
        hyp_samples = expand_b_to_sample_hypnogram(b_string, fs, n_samples, SCORE_EPOCH_SEC)
        vlog("Loaded hypnogram length (epochs):", len(b_string), "sample hypnogram length:", len(hyp_samples))

    return x, hyp_samples, b_string, fs, n_samples, n_channels


# FILE INDEXING / WINDOW BUILDING

def parse_filename_info(path):
    base = os.path.basename(path)
    stem = base[:-4] if base.lower().endswith(".mat") else base
    parts = stem.split("_")
    if len(parts) < 4:
        return None

    date = parts[0]
    name_id = parts[1]
    protocol = parts[2]
    phase = parts[3].upper()
    has_bt = any(p.lower() == "bt" for p in parts[4:])

    if not re.fullmatch(r"\d{6}", date):
        return None
    if phase not in ("P01", "P02"):
        return None

    return {
        "file_path": path,
        "file_name": base,
        "date": date,
        "name_id": name_id,
        "protocol": protocol,
        "phase": phase,
        "has_bt": has_bt,
    }


def collect_bt_p01_p02_files(folder):
    entries = []
    for name in os.listdir(folder):
        if not name.lower().endswith(".mat"):
            continue
        p = os.path.join(folder, name)
        info = parse_filename_info(p)
        if info is None:
            continue
        if not info["has_bt"]:
            continue
        if info["phase"] not in ("P01", "P02"):
            continue
        entries.append(info)

    phase_order = {"P01": 0, "P02": 1}
    entries.sort(key=lambda d: (d["name_id"], int(d["date"]), phase_order[d["phase"]], d["file_name"]))
    vlog("Indexed bt P01/P02 files:", len(entries))
    for e in entries:
        vlog("  -", e["file_name"], "| date:", e["date"], "phase:", e["phase"], "protocol:", e["protocol"])
    return entries


def find_anchor_epoch_index(b_string, fs, mode):
    samples_per_epoch = int(round(SCORE_EPOCH_SEC * fs))
    zt2_epoch = int(round(ZT2_SEC * fs / samples_per_epoch))

    b = np.array(list(b_string), dtype="<U1")

    if mode == "after_last_f_or_ZT2":
        f_idx = np.where(b == TASK_CHAR)[0]
        if len(f_idx) > 0:
            idx = int(f_idx[-1] + 1)
            vlog("Anchor mode after_last_f_or_ZT2 -> last f epoch:", int(f_idx[-1]), "start epoch:", idx)
            return min(max(idx, 0), len(b))

    vlog("Anchor mode", mode, "-> using ZT2 epoch:", zt2_epoch)
    return min(max(zt2_epoch, 0), len(b))


def consolidated_nrem_epoch_mask(b_string, min_run=MIN_CONSOLIDATED_N_EPOCHS):
    b = np.array(list(b_string), dtype="<U1")
    mask = np.zeros(len(b), dtype=bool)
    i = 0
    while i < len(b):
        if b[i] == NREM_CHAR:
            j = i
            while j < len(b) and b[j] == NREM_CHAR:
                j += 1
            if (j - i) >= min_run:
                mask[i:j] = True
            i = j
        else:
            i += 1
    vlog("Consolidated NREM epochs kept:", int(np.sum(mask)), "of", len(mask), "with min run", min_run)
    return mask


def find_end_epoch_for_n_quota(b_string, start_epoch, n_needed):
    b = np.array(list(b_string), dtype="<U1")
    if start_epoch >= len(b):
        return len(b) - 1, 0, False

    n_count = 0
    end_epoch = len(b) - 1
    reached = False

    for idx in range(start_epoch, len(b)):
        if b[idx] == NREM_CHAR:
            n_count += 1
            if n_count >= n_needed:
                end_epoch = idx
                reached = True
                break

    vlog(
        "NREM quota scan from epoch", start_epoch,
        "| needed:", n_needed,
        "| counted:", n_count,
        "| end_epoch:", end_epoch,
        "| reached:", reached
    )
    return end_epoch, n_count, reached


def epoch_mask_to_sample_mask(epoch_mask, n_samples, fs):
    samples_per_epoch = int(round(SCORE_EPOCH_SEC * fs))
    sample_mask = np.repeat(epoch_mask.astype(bool), samples_per_epoch)
    if len(sample_mask) < n_samples:
        sample_mask = np.concatenate([sample_mask, np.zeros(n_samples - len(sample_mask), dtype=bool)])
    return sample_mask[:n_samples]


# SIGNAL PROCESSING + DETECTOR

def make_fir_bandpass(fs, low_hz=SPINDLE_LOW_HZ, high_hz=SPINDLE_HIGH_HZ, numtaps=FIR_NUMTAPS):
    return firwin(numtaps=numtaps, cutoff=[low_hz, high_hz], pass_zero=False, fs=fs)


def zero_phase_fir_filter(x, b):
    return filtfilt(b, [1.0], x)


def compute_troughs_from_signal(x):
    trough_locs, _ = find_peaks(-x)
    return trough_locs


def find_last_n_values_less_than(arr, value, n_keep):
    valid = arr[arr < value]
    if len(valid) < n_keep:
        return None
    return valid[-n_keep:]


def find_first_n_values_greater_than(arr, value, n_keep):
    valid = arr[arr > value]
    if len(valid) < n_keep:
        return None
    return valid[:n_keep]


def merge_close_events(starts, ends, fs, merge_gap_sec=MERGE_GAP_SEC):
    if len(starts) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    merged_starts = [int(starts[0])]
    merged_ends = [int(ends[0])]

    for i in range(1, len(starts)):
        current_start = int(starts[i])
        current_end = int(ends[i])
        previous_end = merged_ends[-1]
        gap_sec = (current_start - previous_end) / fs

        if gap_sec < merge_gap_sec:
            if current_end > merged_ends[-1]:
                merged_ends[-1] = current_end
        else:
            merged_starts.append(current_start)
            merged_ends.append(current_end)

    return np.array(merged_starts, dtype=int), np.array(merged_ends, dtype=int)


def keep_only_events_mostly_in_nrem(starts, ends, hyp, min_fraction=MIN_FRACTION_NREM):
    keep_starts, keep_ends = [], []

    for i in range(len(starts)):
        start = int(starts[i])
        end = int(ends[i])
        seg = hyp[start:end + 1]
        if len(seg) == 0:
            continue
        frac_nrem = np.mean(seg == NREM_CHAR)
        if frac_nrem >= min_fraction:
            keep_starts.append(start)
            keep_ends.append(end)

    return np.array(keep_starts, dtype=int), np.array(keep_ends, dtype=int)


def detect_spindles_lf_style(hyp_samples, signal, fs):
    vlog("Running spindle detection | n_samples:", len(signal), "| fs:", fs)
    x = replace_nan_with_zero(signal)
    hyp = np.array(hyp_samples).astype(str).ravel()

    if len(hyp) != len(x):
        raise ValueError("Hypnogram and signal do not have the same length.")

    b = make_fir_bandpass(fs)
    filtered = zero_phase_fir_filter(x, b)
    filtered_squared = filtered ** 2

    in_nrem = np.where(hyp == NREM_CHAR)[0]
    if len(in_nrem) == 0:
        vlog("No NREM samples after masking. Returning empty result.")
        return build_empty_result(filtered, filtered_squared, np.nan)

    threshold = np.mean(filtered_squared[in_nrem]) + 2.0 * np.std(filtered_squared[in_nrem])
    vlog("Threshold computed:", float(threshold), "| NREM samples for threshold:", len(in_nrem))

    peak_locs_all, _ = find_peaks(filtered_squared)
    peak_vals_all = filtered_squared[peak_locs_all]

    nrem_mask_for_peaks = (hyp[peak_locs_all] == NREM_CHAR)
    peak_locs_nrem = peak_locs_all[nrem_mask_for_peaks]
    peak_vals_nrem = peak_vals_all[nrem_mask_for_peaks]
    all_peaks_pos = peak_locs_nrem.copy()

    if len(peak_locs_nrem) == 0:
        vlog("No NREM peaks found. Returning empty result.")
        return build_empty_result(filtered, filtered_squared, threshold)

    supra_mask = peak_vals_nrem >= threshold
    peak_locs = peak_locs_nrem[supra_mask]

    if len(peak_locs) == 0:
        vlog("No supra-threshold peaks found. Returning empty result.")
        return build_empty_result(filtered, filtered_squared, threshold)

    drop_locs = compute_troughs_from_signal(filtered_squared)
    if len(drop_locs) == 0:
        vlog("No troughs found. Returning empty result.")
        return build_empty_result(filtered, filtered_squared, threshold)

    if peak_locs[0] < drop_locs[0]:
        drop_locs = np.concatenate(([0], drop_locs))

    if len(peak_locs) == 1:
        possible_starts = np.array([peak_locs[0]], dtype=int)
        possible_ends = np.array([peak_locs[0]], dtype=int)
    else:
        time_between_peaks = (peak_locs[1:] - peak_locs[:-1]) / fs
        too_long = np.where(time_between_peaks > ((2.0 * 16.0) / fs))[0]

        possible_ends = np.array([peak_locs[idx] for idx in too_long] + [peak_locs[-1]], dtype=int)
        possible_starts = np.array([peak_locs[0]] + [peak_locs[idx + 1] for idx in too_long], dtype=int)

    real_starts, real_ends = [], []

    for i in range(len(possible_starts)):
        p_start = int(possible_starts[i])
        p_end = int(possible_ends[i])

        left_two = find_last_n_values_less_than(drop_locs, p_start, 2)
        right_two = find_first_n_values_greater_than(drop_locs, p_end, 2)

        if left_two is None or right_two is None:
            continue

        real_start = int(left_two[0])
        real_end = int(right_two[1])

        if real_end > real_start:
            real_starts.append(real_start)
            real_ends.append(real_end)

    real_starts = np.array(real_starts, dtype=int)
    real_ends = np.array(real_ends, dtype=int)

    if len(real_starts) == 0:
        vlog("No candidate events survived start/end assignment.")
        return build_empty_result(filtered, filtered_squared, threshold)

    real_starts, real_ends = merge_close_events(real_starts, real_ends, fs)

    keep_starts, keep_ends = [], []

    for i in range(len(real_starts)):
        start = int(real_starts[i])
        end = int(real_ends[i])

        peaks_in_spindle = all_peaks_pos[(all_peaks_pos > start) & (all_peaks_pos < end)]
        if len(peaks_in_spindle) < 2:
            continue

        speeds_sec = (peaks_in_spindle[1:] - peaks_in_spindle[:-1]) / fs
        if len(speeds_sec) <= 5:
            continue

        mean_freq = np.mean(1.0 / speeds_sec)
        if mean_freq < (9.0 * 2.0) or mean_freq > (16.0 * 2.0):
            continue

        keep_starts.append(start)
        keep_ends.append(end)

    starts = np.array(keep_starts, dtype=int)
    ends = np.array(keep_ends, dtype=int)
    starts, ends = keep_only_events_mostly_in_nrem(starts, ends, hyp)
    vlog("Events after frequency + NREM fraction filters:", len(starts))

    features = extract_spindle_features(starts, ends, filtered, filtered_squared, signal, fs)
    vlog("Feature extraction complete for events:", len(starts))

    return {
        "starts": starts,
        "ends": ends,
        "filtered": filtered,
        "filtered_squared": filtered_squared,
        "threshold": threshold,
        "features": features,
    }


def build_empty_result(filtered, filtered_squared, threshold):
    return {
        "starts": np.array([], dtype=int),
        "ends": np.array([], dtype=int),
        "filtered": filtered,
        "filtered_squared": filtered_squared,
        "threshold": threshold,
        "features": {
            "v_Sp_Amp": np.array([], dtype=float),
            "v_Sp_Speed": np.array([], dtype=float),
            "v_Sp_NCycles": np.array([], dtype=int),
            "v_Sp_Time": np.array([], dtype=float),
            "v_Sp_Pow": np.array([], dtype=float),
            "v_SpindlesLoc": np.array([], dtype=int),
            "v_Sp_PeakFilteredSquared": np.array([], dtype=float),
            "v_Sp_RMS": np.array([], dtype=float),
        },
    }


def extract_spindle_features(starts, ends, filtered, filtered_squared, raw_signal, fs):
    v_sp_amp, v_sp_speed, v_sp_ncycles = [], [], []
    v_sp_time, v_sp_pow, v_sp_center = [], [], []
    v_sp_peakpow, v_sp_rms = [], []

    for i in range(len(starts)):
        start = int(starts[i])
        end = int(ends[i])

        single_spindle = filtered[start:end + 1].copy()
        single_spindle = single_spindle - np.mean(single_spindle)

        amp = np.max(np.abs(single_spindle))
        pos_peak_idx, _ = find_peaks(single_spindle)

        if len(pos_peak_idx) >= 2:
            temp_speed = (pos_peak_idx[1:] - pos_peak_idx[:-1]) / fs
            sp_speed = 1.0 / np.mean(temp_speed)
        else:
            sp_speed = np.nan

        ncycles = len(pos_peak_idx)
        sp_time = (end - start) / fs
        sp_pow = np.sum(single_spindle ** 2)
        center = int(round((start + end) / 2.0))
        peak_pow = np.max(filtered_squared[start:end + 1])
        rms_val = np.sqrt(np.mean(single_spindle ** 2))

        v_sp_amp.append(amp)
        v_sp_speed.append(sp_speed)
        v_sp_ncycles.append(ncycles)
        v_sp_time.append(sp_time)
        v_sp_pow.append(sp_pow)
        v_sp_center.append(center)
        v_sp_peakpow.append(peak_pow)
        v_sp_rms.append(rms_val)

    return {
        "v_Sp_Amp": np.array(v_sp_amp, dtype=float),
        "v_Sp_Speed": np.array(v_sp_speed, dtype=float),
        "v_Sp_NCycles": np.array(v_sp_ncycles, dtype=int),
        "v_Sp_Time": np.array(v_sp_time, dtype=float),
        "v_Sp_Pow": np.array(v_sp_pow, dtype=float),
        "v_SpindlesLoc": np.array(v_sp_center, dtype=int),
        "v_Sp_PeakFilteredSquared": np.array(v_sp_peakpow, dtype=float),
        "v_Sp_RMS": np.array(v_sp_rms, dtype=float),
    }


# CSV EXPORT

def build_rows_for_file(file_meta, channel_index, fs, starts, ends, features, extra):
    rows = []
    file_path = file_meta["file_path"]
    file_name = file_meta["file_name"]

    for i in range(len(starts)):
        start_sample = int(starts[i])
        end_sample = int(ends[i])
        center_sample = int(features["v_SpindlesLoc"][i])

        rows.append({
            "anchor_file": extra["anchor_file"],
            "file_name": file_name,
            "file_path": file_path,
            "date": file_meta["date"],
            "phase": file_meta["phase"],
            "protocol": file_meta["protocol"],
            "channel_index": channel_index,
            "spindle_index_in_file": i,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "center_sample": center_sample,
            "start_sec": start_sample / fs,
            "end_sec": end_sample / fs,
            "center_sec": center_sample / fs,
            "timestamp_relative_to_recording_start_sec": center_sample / fs,
            "spindle_duration_sec": float(features["v_Sp_Time"][i]),
            "intra_spindle_frequency_hz": float(features["v_Sp_Speed"][i]),
            "spindle_amplitude": float(features["v_Sp_Amp"][i]),
            "cycle_count": int(features["v_Sp_NCycles"][i]),
            "spindle_power": float(features["v_Sp_Pow"][i]),
            "spindle_rms": float(features["v_Sp_RMS"][i]),
            "peak_filtered_squared": float(features["v_Sp_PeakFilteredSquared"][i]),
            "window_time_hours": extra["window_time_hours"],
            "window_mode": extra["window_mode"],
            "window_start_epoch": extra["window_start_epoch"],
            "window_end_epoch": extra["window_end_epoch"],
            "nrem_epochs_selected_in_file": extra["nrem_epochs_selected_in_file"],
            "nrem_seconds_selected_in_file": extra["nrem_seconds_selected_in_file"],
            "window_target_reached": extra["window_target_reached"],
            "continuity_flag": extra["continuity_flag"],
        })

    return rows


def save_rows_to_csv(rows, csv_path):
    headers = [
        "anchor_file", "file_name", "file_path", "date", "phase", "protocol", "channel_index",
        "spindle_index_in_file", "start_sample", "end_sample", "center_sample",
        "start_sec", "end_sec", "center_sec", "timestamp_relative_to_recording_start_sec",
        "spindle_duration_sec", "intra_spindle_frequency_hz", "spindle_amplitude", "cycle_count",
        "spindle_power", "spindle_rms", "peak_filtered_squared",
        "window_time_hours", "window_mode", "window_start_epoch", "window_end_epoch",
        "nrem_epochs_selected_in_file", "nrem_seconds_selected_in_file", "window_target_reached", "continuity_flag",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# MAIN PROCESSING

def process_anchor_sequence(entries, start_idx, channel_index, window_hours, window_mode):
    target_nrem_epochs = int(round((window_hours * 3600.0) / SCORE_EPOCH_SEC))
    remaining = target_nrem_epochs

    all_rows = []
    issues = []

    anchor = entries[start_idx]
    anchor_file = anchor["file_name"]
    vlog("============================================================")
    vlog("Starting anchor sequence:", anchor_file, "| window_hours:", window_hours, "| target NREM epochs:", target_nrem_epochs)

    idx = start_idx
    reached = False

    while idx < len(entries) and remaining > 0:
        meta = entries[idx]
        vlog("Processing continuity file:", meta["file_name"], "| remaining NREM epochs:", remaining)

        try:
            signal, hyp_samples, b_string, fs, n_samples, _ = load_trace_and_hypnogram(
                meta["file_path"], channel_index, FALLBACK_FS
            )

            if idx == start_idx:
                start_epoch = find_anchor_epoch_index(b_string, fs, window_mode)
            else:
                start_epoch = 0
                vlog("Non-anchor continuity file; start_epoch set to 0")

            end_epoch, n_count_seen, local_reached = find_end_epoch_for_n_quota(b_string, start_epoch, remaining)

            epoch_mask_window = np.zeros(len(b_string), dtype=bool)
            if len(b_string) > 0 and start_epoch <= end_epoch:
                epoch_mask_window[start_epoch:end_epoch + 1] = True

            nrem_epochs_selected = int(np.sum((np.array(list(b_string)) == NREM_CHAR) & epoch_mask_window))
            remaining -= nrem_epochs_selected
            vlog(
                "File window selected epochs:", int(np.sum(epoch_mask_window)),
                "| NREM epochs selected in file:", nrem_epochs_selected,
                "| remaining after file:", remaining
            )
            if remaining <= 0:
                reached = True
                vlog("Target NREM reached within file:", meta["file_name"])

            epoch_mask_consolidated = consolidated_nrem_epoch_mask(b_string, min_run=MIN_CONSOLIDATED_N_EPOCHS)

            sample_mask_window = epoch_mask_to_sample_mask(epoch_mask_window, n_samples, fs)
            sample_mask_consolidated = epoch_mask_to_sample_mask(epoch_mask_consolidated, n_samples, fs)
            sample_mask_final = sample_mask_window & sample_mask_consolidated

            hyp_for_detection = np.array(hyp_samples).astype(str)
            hyp_for_detection[~sample_mask_final] = WAKE_CHAR

            result = detect_spindles_lf_style(hyp_for_detection, signal, fs)

            extra = {
                "anchor_file": anchor_file,
                "window_time_hours": window_hours,
                "window_mode": window_mode,
                "window_start_epoch": int(start_epoch),
                "window_end_epoch": int(end_epoch),
                "nrem_epochs_selected_in_file": int(nrem_epochs_selected),
                "nrem_seconds_selected_in_file": float(nrem_epochs_selected * SCORE_EPOCH_SEC),
                "window_target_reached": bool(reached),
                "continuity_flag": "ok",
            }

            rows = build_rows_for_file(
                meta, channel_index, fs,
                result["starts"], result["ends"], result["features"], extra
            )
            all_rows.extend(rows)
            vlog("Spindle rows generated for file:", len(rows))

        except Exception as e:
            issues.append(f"{meta['file_name']}: {str(e)}")
            vlog("Error in file:", meta["file_name"], "|", str(e))
            break

        idx += 1

    if not reached:
        if idx >= len(entries):
            issues.append(f"{anchor_file}: target NREM not reached (end of available P01/P02 bt chain).")
        else:
            issues.append(f"{anchor_file}: target NREM not reached due to processing interruption.")
        vlog("Anchor ended without reaching target NREM:", anchor_file)
    else:
        vlog("Anchor completed with target reached:", anchor_file)

    return all_rows, issues


def main():
    vlog("Script started.")
    folder = ask_user_for_folder()
    if not folder:
        print("No folder selected. Exiting.")
        return

    channel_index = ask_user_for_channel_index()
    if channel_index is None:
        print("No channel selected. Exiting.")
        return

    window_hours = ask_user_for_window_hours()
    if window_hours is None:
        print("No window time selected. Exiting.")
        return

    window_mode = ask_user_for_window_mode()
    if window_mode is None:
        print("No valid window mode selected. Exiting.")
        return

    output_csv = ask_user_for_output_csv()
    if not output_csv:
        print("No output CSV selected. Exiting.")
        return

    entries = collect_bt_p01_p02_files(folder)
    if len(entries) == 0:
        raise ValueError("No .mat files matching *_bt.mat with P01/P02 naming were found.")

    # Anchor analyses on P01 files only; P02 files are continuity support.
    anchor_indices = [i for i, e in enumerate(entries) if e["phase"] == "P01"]
    if len(anchor_indices) == 0:
        raise ValueError("No P01 *_bt.mat files were found to anchor analysis.")

    all_rows = []
    all_issues = []

    for start_idx in anchor_indices:
        vlog("Launching processing for anchor index:", start_idx, "| file:", entries[start_idx]["file_name"])
        rows, issues = process_anchor_sequence(
            entries=entries,
            start_idx=start_idx,
            channel_index=channel_index,
            window_hours=window_hours,
            window_mode=window_mode,
        )
        all_rows.extend(rows)
        all_issues.extend(issues)

    save_rows_to_csv(all_rows, output_csv)
    vlog("CSV writing complete. Rows written:", len(all_rows))

    summary_lines = [
        f"CSV saved:\n{output_csv}",
        "",
        f"Folder: {folder}",
        f"Total bt P01/P02 files indexed: {len(entries)}",
        f"Total P01 anchors processed: {len(anchor_indices)}",
        f"Total spindle rows exported: {len(all_rows)}",
        f"Issues flagged: {len(all_issues)}",
    ]

    if len(all_issues) > 0:
        summary_lines.append("")
        summary_lines.append("Flags:")
        for msg in all_issues:
            summary_lines.append(f"- {msg}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    show_info_message("Finished", summary_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", str(e))
        show_error_message("Fatal error", str(e))
