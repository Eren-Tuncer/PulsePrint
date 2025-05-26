import numpy as np
from scipy import signal
from tensorflow.keras.models import load_model
import numpy as np



# Constants from the original code
TARGET_SAMPLE_RATE = 500  # Hz
SEGMENT_LENGTH_SECONDS = 10  # seconds
SEGMENT_LENGTH_SAMPLES = SEGMENT_LENGTH_SECONDS * TARGET_SAMPLE_RATE

def preprocess_ecg_signal(ecg_signal, original_sample_rate=None):
    """
    Preprocess a single ECG signal to be compatible with the trained model.

    Parameters:
    -----------
    ecg_signal : numpy.ndarray
        The raw ECG signal
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.

    Returns:
    --------
    processed_signal : numpy.ndarray
        The preprocessed signal ready for model prediction
    """
    # Check for NaN or inf values
    if np.isnan(ecg_signal).any() or np.isinf(ecg_signal).any():
        print("Warning: NaN or Inf values found in signal. Replacing with zeros.")
        ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=1.0, neginf=-1.0)

    # Resample to 500 Hz if needed
    if original_sample_rate is not None and original_sample_rate != TARGET_SAMPLE_RATE:
        # Calculate number of samples after resampling
        num_samples = int(len(ecg_signal) * TARGET_SAMPLE_RATE / original_sample_rate)
        ecg_signal = signal.resample(ecg_signal, num_samples)
        print(f"Resampled signal from {original_sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")

    # Check if signal is long enough
    if len(ecg_signal) < SEGMENT_LENGTH_SAMPLES:
        print(f"Warning: Signal is shorter than required length ({len(ecg_signal)} < {SEGMENT_LENGTH_SAMPLES}).")
        print("Padding with zeros to reach required length.")
        # Pad with zeros
        pad_length = SEGMENT_LENGTH_SAMPLES - len(ecg_signal)
        ecg_signal = np.pad(ecg_signal, (0, pad_length), 'constant')

    # If signal is too long, take the first 10 seconds
    if len(ecg_signal) > SEGMENT_LENGTH_SAMPLES:
        print(f"Signal is longer than required. Taking first {SEGMENT_LENGTH_SECONDS} seconds.")
        ecg_signal = ecg_signal[:SEGMENT_LENGTH_SAMPLES]

    # Z-score normalization
    ecg_signal = (ecg_signal - np.mean(ecg_signal)) / np.std(ecg_signal)

    # Reshape for CNN input [batch, time steps, features]
    processed_signal = ecg_signal.reshape(1, SEGMENT_LENGTH_SAMPLES, 1)

    return processed_signal

def segment_long_ecg(long_ecg, original_sample_rate=None, overlap=0.5):
    """
    Segment a long ECG recording into 10-second segments with overlap.

    Parameters:
    -----------
    long_ecg : numpy.ndarray
        The long ECG recording
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.
    overlap : float
        Overlap between segments (0.5 = 50% overlap)

    Returns:
    --------
    segments : list of numpy.ndarray
        List of preprocessed segments ready for model prediction
    """
    # Resample if needed
    if original_sample_rate is not None and original_sample_rate != TARGET_SAMPLE_RATE:
        # Calculate number of samples after resampling
        num_samples = int(len(long_ecg) * TARGET_SAMPLE_RATE / original_sample_rate)
        resampled_ecg = signal.resample(long_ecg, num_samples)
        print(f"Resampled signal from {original_sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
    else:
        resampled_ecg = long_ecg

    # Clean the signal
    if np.isnan(resampled_ecg).any() or np.isinf(resampled_ecg).any():
        print("Warning: NaN or Inf values found in signal. Replacing with zeros.")
        resampled_ecg = np.nan_to_num(resampled_ecg, nan=0.0, posinf=1.0, neginf=-1.0)

    # Segment the recording into 10-second windows
    window_size = SEGMENT_LENGTH_SAMPLES
    step = int(window_size * (1 - overlap))

    segments = []
    for i in range(0, len(resampled_ecg) - window_size + 1, step):
        segment = resampled_ecg[i:i + window_size]

        # Z-score normalization
        segment = (segment - np.mean(segment)) / np.std(segment)

        # Reshape for CNN input [batch, time steps, features]
        segment = segment.reshape(1, SEGMENT_LENGTH_SAMPLES, 1)
        segments.append(segment)

    return segments

def predict_ecg_authenticity(model, ecg_signal, original_sample_rate=None):
    """
    Predict whether an ECG signal is real or fake.

    Parameters:
    -----------
    model : tensorflow.keras.Model
        The trained classifier model
    ecg_signal : numpy.ndarray
        The raw ECG signal
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.

    Returns:
    --------
    prediction : float
        The model's prediction (0-1, where closer to 1 means more likely fake)
    """
    # Preprocess the signal
    processed_signal = preprocess_ecg_signal(ecg_signal, original_sample_rate)

    # Make prediction
    prediction = model.predict(processed_signal)[0][0]

    return prediction

def classify_ecg_recording(model_path, ecg_signal, original_sample_rate=None, threshold=0.5):
    """
    Complete pipeline to classify an ECG recording as real or fake.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    ecg_signal : numpy.ndarray
        The ECG signal to classify
    original_sample_rate : int or None
        The sampling rate of the input signal. If None, assumed to be already 500Hz.
    threshold : float
        Classification threshold (default 0.5)

    Returns:
    --------
    result : dict
        Dictionary containing classification results
    """
    # Load the model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Determine if it's a long recording or a short segment
    is_long_recording = len(ecg_signal) > (SEGMENT_LENGTH_SAMPLES * 1.5)

    if is_long_recording:
        print(f"Processing long recording ({len(ecg_signal)} samples)")
        # Segment the recording
        segments = segment_long_ecg(ecg_signal, original_sample_rate)

        # Make predictions for each segment
        segment_predictions = []
        for i, segment in enumerate(segments):
            pred = model.predict(segment)[0][0]
            segment_predictions.append(pred)
            print(f"Segment {i+1}/{len(segments)}: {pred:.4f} ({'Fake' if pred > threshold else 'Real'})")

        # Aggregate results
        avg_prediction = np.mean(segment_predictions)
        max_prediction = np.max(segment_predictions)

        result = {
            'classification': 0 if avg_prediction > threshold else 1, # 0 for Fake, 1 for Real
            'confidence': abs(avg_prediction - 0.5) * 2,  # Scale to 0-1
            'avg_score': avg_prediction,
            'max_score': max_prediction,
            'segment_predictions': segment_predictions,
            'num_segments': len(segments)
        }
    else:
        # Process as a single segment
        print("Processing single segment")
        processed_signal = preprocess_ecg_signal(ecg_signal, original_sample_rate)
        prediction = model.predict(processed_signal)[0][0]

        result = {
            'classification': 0 if prediction > threshold else 1, # 0 for Fake, 1 for Real
            'confidence': abs(prediction - 0.5) * 2,  # Scale to 0-1
            'score': prediction
        }

    return result



def read_from_file(file_path) -> np.ndarray:
    """
    Read ECG data from a file.

    Parameters:
    -----------
    file_path : str
        Path to the ECG data file

    Returns:
    --------
    ecg_signal : numpy.ndarray
        The ECG signal read from the file
    """
    try:
        # Try to read the file as a simple text file with one value per line
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # Convert each line to float, handling potential whitespace and empty lines
            ecg_values = []
            for line in lines:
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        ecg_values.append(float(line))
                    except ValueError:
                        # If a line contains multiple values separated by spaces or commas
                        try:
                            parts = line.replace(',', ' ').split()
                            ecg_values.extend([float(part) for part in parts if part])
                        except ValueError as e:
                            print(f"Warning: Skipping invalid line: {line}, error: {e}")
        
        if not ecg_values:
            raise ValueError("No valid ECG values found in the file")
        
        return np.array(ecg_values)
    
    except Exception as e:
        print(f"Error reading ECG file: {e}")
        raise


