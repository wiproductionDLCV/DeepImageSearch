import os
from flask import Flask, render_template, request, jsonify
import time
import threading
import tkinter as tk
from tkinter import filedialog

# Point to the correct template folder
template_path = os.path.join(os.path.dirname(__file__), 'ui', 'templates')
app = Flask(__name__, template_folder=template_path)

download_status = {"status": "idle"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-json', methods=['POST'])
def upload_json():
    json_file = request.files['jsonFile']
    if json_file:
        json_file.save('uploaded_config.json')
        return jsonify({"message": "File uploaded successfully!"}), 200
    return jsonify({"message": "Upload failed!"}), 400

@app.route('/start-download', methods=['POST'])
def start_download():
    def simulate_download():
        download_status["status"] = "in_progress"
        time.sleep(5)  # Simulate time taken to download from S3
        download_status["status"] = "completed"

    threading.Thread(target=simulate_download).start()
    return jsonify({"message": "Download started"}), 202

@app.route('/check-status')
def check_status():
    return jsonify(download_status)

@app.route('/select-folder')
def select_folder():
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        root.attributes('-topmost', True)  # Bring dialog to front
        folder_path = filedialog.askdirectory()
        root.destroy()
        return jsonify({"folderPath": folder_path})
    except Exception as e:
        print("Error selecting folder:", e)
        return jsonify({"folderPath": "", "error": str(e)})

if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0', debug=True)