from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
import tempfile
from prediction import classify_ecg_recording, read_from_file
from auth import *
app = Flask(__name__)
CORS(app)

# Model path - update this to point to your actual model file
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/ecg_classifier.h5')

@app.route('/api/predict_and_auth', methods=['POST'])
def predict_and_authenticate():
    try:
        if 'ecg_file' not in request.files:
            return jsonify({'error': 'No ECG file uploaded'}), 400

        file = request.files['ecg_file']
        original_filename = file.filename
        suffix = os.path.splitext(original_filename)[1].lower()
        print(original_filename)
        print(suffix)
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        sample_rate = request.form.get('sample_rate', 500)
        try:
            sample_rate = int(sample_rate)
        except ValueError:
            return jsonify({'error': 'Sample rate must be a number'}), 400

        # Handle .dat files which require .hea files
        if suffix == '.dat':
            # Check if .hea file is also uploaded
            hea_file = request.files.get('hea_file')
            if not hea_file:
                return jsonify(
                    {'error': '.dat files require a corresponding .hea file. Please upload both files.'}), 400

            # Create temporary files with the same base name
            base_name = os.path.splitext(original_filename)[0]
            temp_dir = tempfile.mkdtemp()

            dat_path = os.path.join(temp_dir, f"{base_name}.dat")
            hea_path = os.path.join(temp_dir, f"{base_name}.dat.hea")

            file.save(dat_path)
            hea_file.save(hea_path)
            print(dat_path)
            print(hea_path)

            temp_path = os.path.join(temp_dir, dat_path)
            temp_paths_to_cleanup = [dat_path, hea_path, temp_dir]
        else:
            # For other file types, use single temp file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                file.save(temp_file.name)
                temp_path = temp_file.name
            temp_paths_to_cleanup = [temp_path]

        try:
            # Predict
            ecg_signal = load_file(temp_path,suffix)
            result = classify_ecg_recording(MODEL_PATH, ecg_signal, original_sample_rate=sample_rate)

            if result is None:
                return jsonify({'error': 'Error processing ECG'}), 500

            classification = int(result['classification'])
            confidence = float(result['confidence'])

            response = {
                'classification': classification,
                'confidence': confidence
            }

            if classification == 1:  # Only authenticate if ECG is real
                records = load_mit_bih_samples()
                gallery_templates = {}

                for signal_id, signal in records.items():
                    if signal is not None:
                        processed = normalize_signal(signal)
                        r_peaks = detect_r_peaks(processed)
                        windows = segment_around_r_peaks(processed, r_peaks)
                        if len(windows) > 0:
                            template = extract_wdm_template(windows)
                            gallery_templates[signal_id] = template

                ecg_signal = resample_signal(ecg_signal,sample_rate,360)
                processed_signal = normalize_signal(ecg_signal)
                r_peaks = detect_r_peaks(processed_signal)
                windows = segment_around_r_peaks(processed_signal, r_peaks)

                if len(windows) > 0:
                    template = extract_wdm_template(windows)
                    is_auth, claimed_id, distance = authenticate(template, gallery_templates,100)

                    response.update({
                        'authenticated': bool(is_auth),
                        'claimed_id': claimed_id,
                        'distance': float(distance)
                    })
                else:
                    response.update({'authenticated': False, 'error': 'No valid windows for authentication'})

            return jsonify(response), 200

        finally:
            os.unlink(temp_path)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

