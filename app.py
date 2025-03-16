from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import glob
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'vid_input'
OUTPUT_FOLDER = 'runs/detect'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to get the latest YOLOv5 output folder
def get_latest_output_folder():
    exp_folders = sorted(glob.glob(f"{OUTPUT_FOLDER}/exp*"), key=os.path.getctime, reverse=True)
    return exp_folders[0] if exp_folders else None

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run YOLOv5 detection
    os.system(f"python detect.py --weights best.pt --img 416 --conf 0.5 --source {filepath}")

    # Get the latest output folder
    latest_exp = get_latest_output_folder()
    if not latest_exp:
        return jsonify({'error': 'No output folder found'}), 500

    # Find processed images in the latest folder
    processed_images = sorted(glob.glob(f"{latest_exp}/*.jpg"), key=os.path.getctime, reverse=True)
    if not processed_images:
        return jsonify({'error': 'Detection failed'}), 500

    processed_image = os.path.basename(processed_images[0])
    
    # **Return full image URL for frontend**
    return jsonify({
        'processed_image': processed_image,
        'image_url': f"http://localhost:5000/output/{processed_image}"
    })

@app.route('/output/<filename>')
def get_output_image(filename):
    latest_exp = get_latest_output_folder()
    if latest_exp:
        return send_from_directory(latest_exp, filename)
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
