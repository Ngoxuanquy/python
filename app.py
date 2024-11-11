import os
import cv2
import numpy as np
from flask import Flask, request, redirect, render_template, url_for, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def read_image(filepath):
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img_bgr is None or img_gray is None:
        print(f"Error: Unable to load image at {filepath}")
        return None, None
    return img_bgr, img_gray

def resize_and_preprocess(img_gray, img_bgr):
    height, width = img_gray.shape
    if height > 900:
        resized_dim = (width // 10, height // 10)
        factor = 300
    elif height < 200:
        resized_dim = (width * 2, height * 2)
        factor = 300
    else:
        resized_dim = (width, height)
        factor = 500

    img_gray = cv2.resize(img_gray, resized_dim, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.medianBlur(img_gray, 5)
    img_bgr = cv2.resize(img_bgr, resized_dim, interpolation=cv2.INTER_CUBIC)
    return img_bgr, img_gray, factor

def detect_and_draw_circles(img_gray, img_bgr, factor):
    # Adjust minRadius and maxRadius to detect eyes better
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, factor, param1=50, param2=30, minRadius=20, maxRadius=80)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            xc, yc, r = circle
            # Draw the outer circle (green)
            cv2.circle(img_bgr, (xc, yc), r, (0, 255, 0), 2)
            # Draw the center of the circle (red)
            cv2.circle(img_bgr, (xc, yc), 2, (0, 0, 255), 3)
    return circles

def evaluate_cataract(img_gray, xc, yc, r):
    # Calculate the region around the eye to assess cataract severity
    y, x = np.ogrid[:img_gray.shape[0], :img_gray.shape[1]]
    mask = (x - xc) ** 2 + (y - yc) ** 2 > r ** 2
    inside = np.ma.masked_where(mask, img_gray)
    average_color = inside.mean()
    
    # Determine cataract severity based on average color
    if average_color <= 60:
        return "Không bị"
    elif 60 < average_color <= 100:
        return "Bị nhẹ"
    elif 100 < average_color <= 150:
        return "Bị mức độ trung bình"
    else:
        return "Bị nặng"

def detect_cataract(filepath):
    img_bgr, img_gray = read_image(filepath)
    if img_bgr is None or img_gray is None:
        return "Error loading image", None
    img_bgr, img_gray, factor = resize_and_preprocess(img_gray, img_bgr)
    circles = detect_and_draw_circles(img_gray, img_bgr, factor)

    if circles is not None:
        if len(circles[0]) > 0:  # Handle any number of eyes detected
            results = []
            for i, (xc, yc, r) in enumerate(circles[0]):
                eye_name = f"Mắt {i+1}"  # Label each detected eye
                results.append(f"{eye_name}: {evaluate_cataract(img_gray, xc, yc, r)}")
            message = " - ".join(results)
        else:
            message = "Không phát hiện mắt nào."
        return message, img_bgr
    else:
        return "Không phát hiện mắt nào", img_bgr

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        message, img_bgr = detect_cataract(filename)
        if img_bgr is not None:
            output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
            cv2.imwrite(output_filename, img_bgr)
            processed_image_url = url_for('uploaded_file', filename='result_' + file.filename)
            return render_template('result.html', message=message, image_url=processed_image_url)
        else:
            return 'Error processing the image'
    else:
        return 'Invalid file type. Please upload a JPG, JPEG, PNG, or BMP file.'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
