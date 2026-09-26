import os
import cv2
import time
import base64
import numpy as np
import datetime
import logging
from flask import Flask, Response, render_template_string, request, jsonify

import config

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ==============================================================================
# SHARED STATE CONTAINER (BRIDGE)
# Acts as a shared memory bridge between the Flask threads and the AI processing loop.
# ==============================================================================
class WebState:
    shared_state = None
    frame_buffer = None
    frame_lock = None
    kinematics = None
    target_waypoints = []

# ==============================================================================
# FLASK ROUTES
# ==============================================================================
@app.route('/')
def index(): 
    return render_template_string(config.HTML_PAGE)

@app.route('/graphs')
def graphs(): 
    return render_template_string(config.HTML_PAGE_GRAPHS)

@app.route('/api/save_graphs', methods=['POST'])
def save_graphs():
    try:
        data = request.json
        
        # 1. Tạo folder mới có chứa Ngày_Giờ
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(config.BASE_DIR, "DATA", f"graphs_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        saved_files = []
        
        # 2. Tách từng đồ thị ra và lưu thành các file PNG riêng biệt
        for chart_name, b64_data in data.items():
            if b64_data and "," in b64_data:
                # Tách phần header "data:image/png;base64," ra khỏi dữ liệu thực
                header, encoded = b64_data.split(",", 1)
                img_data = base64.b64decode(encoded)
                
                file_path = os.path.join(save_dir, f"{chart_name}.png")
                with open(file_path, "wb") as f:
                    f.write(img_data)
                saved_files.append(f"{chart_name}.png")

        print(f"[INFO] Saved {len(saved_files)} graphs to {save_dir}")
        return jsonify({"status": "success", "folder": save_dir, "files": saved_files})
        
    except Exception as e:
        print(f"[ERROR] Saving graphs failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """ Streams the processed frames to the web UI. """
    def generate():
        while True:
            # Safely access the frame buffer through the lock
            if WebState.frame_lock:
                with WebState.frame_lock:
                    if WebState.frame_buffer is None:
                        time.sleep(0.05)
                        continue
                    ret, jpeg = cv2.imencode('.jpg', WebState.frame_buffer)
                if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/command', methods=['POST'])
def command():
    """ Handles incoming commands from the web UI (Start/Stop, Speed, Light, LLIE). """
    data = request.json
    c_type = data.get('type')
    
    if c_type == 'run': 
        WebState.shared_state['is_running'] = data.get('state', False)
        # Reset kinematic tracking map when vehicle starts a new run
        if WebState.shared_state['is_running'] and WebState.kinematics:
            WebState.kinematics.reset_mapping()
            
    elif c_type == 'light': 
        WebState.shared_state['light_pwm'] = data.get('value', 255)
        
    elif c_type == 'speed_slider': 
        WebState.shared_state['max_speed'] = data.get('value', 100)
        
    elif c_type == 'speed':
        if data.get('key') == 'j': 
            WebState.shared_state['max_speed'] = min(255, WebState.shared_state['max_speed'] + 5)
        elif data.get('key') == 'k': 
            WebState.shared_state['max_speed'] = max(0, WebState.shared_state['max_speed'] - 5)
            
    # =========================================================
    # THE MISSING FIX: Handle LLIE Toggle Command from Web
    # =========================================================
    elif c_type == 'llie':
        # Default to True (ON) if state is missing from payload
        WebState.shared_state['use_llie'] = data.get('state', True)
        
    return jsonify({"status": "ok"})

@app.route('/api/telemetry', methods=['GET'])
def telemetry():
    """ Delivers real-time data to the dashboard and graph UI. """
    if WebState.shared_state is None:
        return jsonify({"status": "loading"})
        
    return jsonify({
        'is_running': WebState.shared_state.get('is_running', False),
        
        # =========================================================
        # THE MISSING FIX: Sync LLIE State with Web UI
        # =========================================================
        'use_llie': WebState.shared_state.get('use_llie', True),
        
        'max_speed': WebState.shared_state.get('max_speed', 0),
        'current_speed': WebState.shared_state.get('current_speed', 0),
        'target_angle': WebState.shared_state.get('target_angle', 85),
        'ai_fps': WebState.shared_state.get('ai_fps', 0.0),
        't_llie': WebState.shared_state.get('t_llie', 0.0),
        't_yolo': WebState.shared_state.get('t_yolo', 0.0),
        'pitch': WebState.shared_state.get('pitch', 0.0),
        'yaw_rate': WebState.shared_state.get('yaw_rate', 0.0),
        'acc_x': WebState.shared_state.get('acc_x', 0.0),
        'drive_state': WebState.shared_state.get('drive_state', 'IDLE'),
        'history': WebState.kinematics.history if WebState.kinematics else [],
        'waypoints': WebState.target_waypoints,
        'current_classes': WebState.shared_state.get('current_classes', [])
    })