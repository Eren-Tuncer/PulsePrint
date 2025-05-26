import numpy as np
import wfdb
import pywt
from scipy.signal import find_peaks
from scipy import signal as sig_proc
import os
from pathlib import Path
from prediction import read_from_file



"""### Globals"""

# DATA GLOBALS
MIT_BIH_PATH = 'mit-bih-arrhythmia-database-1.0.0'

# SIGNAL GLOBALS
MIT_BIH_SAMPLING_RATE = 360
TARGET_SAMPLING_RATE = 360
SEQUENCE_LENGTH_SEC = 10
TARGET_SEQUENCE_LENGTH = SEQUENCE_LENGTH_SEC * TARGET_SAMPLING_RATE

def load_file(file_path,suffix):

    file = Path(file_path)
    if not file.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if suffix == '.dat':
        try:
            # For .dat files, use wfdb to read the record
            record = wfdb.rdrecord(file_path)
            # Assuming the ECG signal is in the first channel
            signal = record.p_signal[:, 0]
            return signal
        except Exception as e:
            raise IOError(f"Error reading WFDB .dat file {file_path}: {e}")
    elif suffix == '.txt' or suffix == '.asc':
        try:
            return read_from_file(file_path)
        except Exception as e:
            raise IOError(f"Error reading text file {file_path}: {e}")
    else:
        raise ValueError(f"Unsupported file type: {file.suffix}")

def load_mit_bih_samples():

    record_files = [f for f in os.listdir(MIT_BIH_PATH) if f.endswith('.dat')]
    record_names = [f.split('.')[0] for f in record_files]
    record_names = list(set(record_names))

    samples = {}

    for record_name in record_names:

        try:
            record_path = os.path.join(MIT_BIH_PATH, record_name)
            record = wfdb.rdrecord(record_path)

            signal = record.p_signal[:, 0]


            segment_length = 10 * MIT_BIH_SAMPLING_RATE
            segment = signal[:segment_length]


            resampled_segment = resample_signal(segment, MIT_BIH_SAMPLING_RATE, TARGET_SAMPLING_RATE)


            if len(resampled_segment) > TARGET_SEQUENCE_LENGTH:
                resampled_segment = resampled_segment[:TARGET_SEQUENCE_LENGTH]
            elif len(resampled_segment) < TARGET_SEQUENCE_LENGTH:
                resampled_segment = np.pad(resampled_segment, (0, TARGET_SEQUENCE_LENGTH - len(resampled_segment)))


            normalized_segment = normalize_signal(resampled_segment)

            samples[record_name] = normalized_segment



        except Exception as e:
            print(f"Error processing record {record_name}: {e}")

    return samples

def normalize_signal(signal):

    if np.std(signal) == 0:
        return signal
    return (signal - np.mean(signal)) / np.std(signal)

def resample_signal(signal, from_rate, to_rate=360):

    num_samples = int(len(signal) * to_rate / from_rate)

    resampled = sig_proc.resample(signal, num_samples)

    return resampled

def detect_r_peaks(signal, fs=360):

    min_distance = int(0.2 * fs)
    r_peaks, _ = find_peaks(signal, height=0.5, distance=min_distance)

    return r_peaks

def segment_around_r_peaks(signal, r_peaks, window_size=250):

    half_window = window_size // 2
    windows = []

    for peak in r_peaks:
        if peak - half_window >= 0 and peak + half_window < len(signal):
            window = signal[peak - half_window:peak + half_window]
            windows.append(window)

    return np.array(windows)


def extract_wdm_template(windows, wavelet='db3', level=5):

    avg_window = np.mean(windows, axis=0)

    coeffs = pywt.wavedec(avg_window, wavelet, level=level)

    detail_coeffs = np.concatenate(coeffs[1:])

    return detail_coeffs


def compute_wdm_distance(template1, template2):

    min_len = min(len(template1), len(template2))
    template1 = template1[:min_len]
    template2 = template2[:min_len]

    distance = np.sum(np.abs(template1 - template2))

    return distance

"""### Authentication"""

def authenticate(probe_template, gallery_templates, threshold):

    min_distance = float('inf')
    claimed_id = None

    for subject_id, gallery_template in gallery_templates.items():
        distance = compute_wdm_distance(probe_template, gallery_template)

        if distance < min_distance:
            min_distance = distance
            claimed_id = subject_id

    is_authenticated = min_distance < threshold

    return is_authenticated, claimed_id, min_distance




