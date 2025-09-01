import io
import uuid
import os
import math
import pandas as pd
import openpyxl
import numpy as np
from math import acos, degrees
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
from colorthief import ColorThief
from color_harmony import get_harmony
from flask_cors import CORS

app = Flask(__name__)

# Konfigurasi CORS untuk mengizinkan akses dari Blogger
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://*.blogspot.com",  # Semua subdomain Blogger
            "http://localhost:8000",   # Untuk development lokal
            "http://127.0.0.1:8000"    # Untuk development lokal
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Handle preflight requests untuk semua routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://yourblogname.blogspot.com')  # Ganti dengan URL blog Anda
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Semua upload (App1 & App2) dalam folder yang sama
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

############################################
# ============== COLOR DB ==================
############################################

COLOR_DB = []
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # lokasi file main.py
    excel_path = os.path.join(BASE_DIR, "color.xlsx")      # path absolut

    df = pd.read_excel(excel_path, engine="openpyxl")
    for _, row in df.iterrows():
        r = int(float(row["Red"]) * 255)
        g = int(float(row["Green"]) * 255)
        b = int(float(row["Blue"]) * 255)
        COLOR_DB.append({
            "name": str(row["Name"]),
            "rgb": (r, g, b),
            "hex": str(row["Hex"])
        })
    print(f"Loaded {len(COLOR_DB)} colors from {excel_path}")
except Exception as e:
    print(f"Failed to load color.xlsx: {e}")
    df = pd.DataFrame()

def dist(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def closest_color_name(rgb):
    return min(COLOR_DB, key=lambda c: dist(rgb, c["rgb"])) if COLOR_DB else {"name": "Unknown"}

def rgb_to_hex(rgb):
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"

############################################
# ============== APP SELECTOR ==============
############################################

@app.route("/")
def main_index():
    # Kembalikan info API sederhana
    return jsonify({
        "message": "Flask Color Tools API",
        "version": "1.0",
        "endpoints": {
            "dominant_color": "/dom-color-app/analyze",
            "color_at_point": "/dom-color-app/color_at",
            "angle_finder": "/angle-finder-app/calculate_angle",
            "color_harmony": "/get_color_harmony",
            "color_info": "/get_color_info"
        }
    })

############################################
# ============== APP 1 =====================
############################################

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp"}

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/dom-color-app/upload", methods=["POST", "OPTIONS"])
def upload_app1():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["image"]
    if f.filename == "" or not allowed(f.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        img = Image.open(f.stream)
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Failed to open image: {e}"}), 400

    img_id = f"app1_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(app.config["UPLOAD_FOLDER"], img_id)
    img.save(out_path, format="JPEG", quality=92, optimize=True)

    w, h = img.size
    return jsonify({
        "image_id": img_id,
        "image_url": f"/dom-color-app/uploads/{img_id}",
        "width": w,
        "height": h
    })

@app.route("/dom-color-app/uploads/<path:fname>")
def serve_upload_app1(fname):
    return send_from_directory(app.config["UPLOAD_FOLDER"], fname)

@app.route("/dom-color-app/color_at", methods=["POST", "OPTIONS"])
def color_at_app1():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    data = request.get_json(silent=True) or {}
    img_id = data.get("image_id")
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))

    if not img_id or x < 0 or y < 0:
        return jsonify({"error": "Missing or invalid parameters"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], img_id)
    if not os.path.isfile(path):
        return jsonify({"error": "Image not found"}), 404

    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        W, H = img.size
        if x >= W or y >= H:
            return jsonify({"error": "Coordinates out of bounds"}), 400

        rgb = img.getpixel((x, y))
        hex_val = rgb_to_hex(rgb)
        color_info = closest_color_name(rgb)

        return jsonify({
            "rgb": rgb,
            "hex": hex_val,
            "name": color_info["name"]
        })
    except Exception as e:
        return jsonify({"error": f"Failed to get color: {e}"}), 500

@app.route("/dom-color-app/analyze", methods=["POST", "OPTIONS"])
def analyze_app1():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    data = request.get_json(silent=True) or {}
    img_id = data.get("image_id")
    rect = data.get("rect") or {}
    if not img_id or not rect:
        return jsonify({"error": "Missing image_id or rect"}), 400

    path = os.path.join(app.config["UPLOAD_FOLDER"], img_id)
    if not os.path.isfile(path):
        return jsonify({"error": "Image not found"}), 404

    try:
        x = int(rect.get("x", 0))
        y = int(rect.get("y", 0))
        w = int(rect.get("w", 0))
        h = int(rect.get("h", 0))
    except Exception:
        return jsonify({"error": "Invalid rect"}), 400

    if w <= 1 or h <= 1:
        return jsonify({"error": "Selection too small"}), 400

    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        W, H = img.size

        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        crop = img.crop((x, y, x + w, y + h))

        bio = io.BytesIO()
        crop.save(bio, format="PNG")
        bio.seek(0)

        ct = ColorThief(bio)
        dominant = ct.get_color(quality=1)
        palette = ct.get_palette(color_count=5) or []

        result = {
            "dominant": {
                "rgb": dominant,
                "hex": rgb_to_hex(dominant),
                "name": closest_color_name(dominant)["name"]
            },
            "palette": [
                {
                    "rgb": p,
                    "hex": rgb_to_hex(p),
                    "name": closest_color_name(p)["name"]
                }
                for p in palette
            ],
            "rect_clamped": {"x": x, "y": y, "w": w, "h": h},
            "image_size": {"w": W, "h": H}
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {e}"}), 500

############################################
# ============== APP 2 =====================
############################################

@app.route("/angle-finder-app/upload", methods=["POST", "OPTIONS"])
def upload_app2():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    file = request.files.get('image')
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400
        
    # Validasi ekstensi file
    if not allowed(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
        
    fname = f"app2_{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    
    try:
        file.save(filepath)
        return jsonify({
            'filename': fname, 
            'url': f"/angle-finder-app/uploads/{fname}"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500

@app.route("/angle-finder-app/uploads/<path:fname>")
def serve_upload_app2(fname):
    return send_from_directory(app.config["UPLOAD_FOLDER"], fname)

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = degrees(acos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)

@app.route("/angle-finder-app/calculate_angle", methods=["POST", "OPTIONS"])
def angle_app2():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    data = request.get_json()
    if not data or 'p1' not in data or 'p2' not in data or 'p3' not in data:
        return jsonify({"error": "Missing required points"}), 400
        
    try:
        p1 = data['p1']
        p2 = data['p2']
        p3 = data['p3']
        angle = calculate_angle(p1, p2, p3)
        return jsonify({'angle': angle})
    except Exception as e:
        return jsonify({"error": f"Failed to calculate angle: {str(e)}"}), 500

############################################
# ============== APP 3 =====================
############################################

@app.route("/get_color_harmony", methods=["POST", "OPTIONS"])
def api_color_harmony():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    data = request.get_json()
    if not data or 'color' not in data:
        return jsonify({"error": "No color provided"}), 400
        
    base_color = data.get('color')
    try:
        harmony = get_harmony(base_color)
        return jsonify(harmony)
    except Exception as e:
        return jsonify({"error": f"Failed to generate color harmony: {str(e)}"}), 500

@app.route("/get_color_info", methods=["POST", "OPTIONS"])
def api_color_info():
    if request.method == "OPTIONS":
        return jsonify({}), 200
        
    data = request.get_json(silent=True) or {}
    rgb = data.get("rgb")
    hex_val = data.get("hex")

    if rgb:
        try:
            # Handle berbagai format RGB
            if isinstance(rgb, str):
                rgb = tuple(map(int, rgb.strip('()').split(',')))
            else:
                rgb = tuple(map(int, rgb))
        except Exception:
            return jsonify({"error": "Invalid rgb format"}), 400
    elif hex_val:
        try:
            # Hilangkan # jika ada
            hex_val = hex_val.lstrip('#')
            # Pastikan panjang hex valid
            if len(hex_val) not in [3, 6]:
                return jsonify({"error": "Invalid hex format"}), 400
            # Konversi ke RGB
            if len(hex_val) == 3:
                hex_val = ''.join([c*2 for c in hex_val])
            rgb = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
        except Exception:
            return jsonify({"error": "Invalid hex format"}), 400
    else:
        return jsonify({"error": "No color provided"}), 400

    match = closest_color_name(rgb)
    if not match:
        return jsonify({"error": "Color database empty"}), 500

    return jsonify({
        "input": {
            "rgb": rgb,
            "hex": rgb_to_hex(rgb)
        },
        "closest": {
            "name": match["name"],
            "rgb": match["rgb"],
            "hex": match["hex"]
        }
    })

############################################
# ============== ERROR HANDLERS ============
############################################

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({"error": "File too large"}), 413

############################################
# ============== MAIN RUN ==================
############################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
