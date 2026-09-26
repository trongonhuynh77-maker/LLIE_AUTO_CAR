import cv2
import time
import os
import math
import torch
import numpy as np
import collections
import threading
from queue import Queue
import tensorrt as trt
import config

# ==============================================================================
# MULTI-THREADED CAMERA STREAM
# ==============================================================================
class CameraStream:
    def __init__(self, src=0, width=960, height=640):
        self.src = src
        self.width = width
        self.height = height
        self.queue = Queue(maxsize=3)
        self.stopped = False
        self.reconnect()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def reconnect(self):
        if hasattr(self, 'stream') and self.stream is not None:
            self.stream.release()
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)

    def update(self):
        fail_count = 0
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.stream.read()
                if not ret: 
                    fail_count += 1
                    time.sleep(0.05) 
                    if fail_count > 10:
                        self.reconnect()
                        fail_count = 0
                    continue
                fail_count = 0
                self.queue.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=2.0)
        if hasattr(self, 'stream'): 
            self.stream.release()


# ==============================================================================
# TENSORRT ENGINE WRAPPER (High-Performance Inference)
# ==============================================================================
class TRTEngineWrapper:
    def __init__(self, engine_path, device):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.device = device
        self.inputs, self.outputs = [], []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            shape = tuple([1 if s == -1 else s for s in shape])
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            t_dtype = torch.float16 if dtype == np.float16 else torch.float32
            dev_tensor = torch.empty(shape, dtype=t_dtype, device=self.device)
            self.context.set_tensor_address(name, dev_tensor.data_ptr())
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(dev_tensor)
            else:
                self.outputs.append(dev_tensor)

    def __call__(self, x):
        x = x.half() if self.inputs[0].dtype == torch.float16 else x.float()
        self.inputs[0].copy_(x)
        self.context.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return [out.clone() for out in self.outputs]
    

# ==============================================================================
# VISION TRACKERS & ESTIMATORS
# ==============================================================================
class TrafficSignTracker:
    def __init__(self):
        # Target area threshold to trigger a turn (Must be close enough)
        self.TARGET_AREA = 4500.0 
        self.current_area = 0.0 

    def update_and_evaluate(self, bbox, frame_width=960, direction='LEFT', sign_placement='RIGHT'):
        """
        Instant evaluation based on Bounding Box Area.
        No history buffer needed since false positives rarely exceed the massive 4500px^2 threshold.
        """
        if bbox is None:
            self.current_area = 0.0
            return False
        
        # Calculate current BBox Area (Width * Height)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        self.current_area = w * h
        
        # Trigger turn IMMEDIATELY if the sign is large enough in a single frame
        if self.current_area >= self.TARGET_AREA:
            return True
                
        return False
    
class StopSignTracker:
    def __init__(self, history_size=5):
        self.history = collections.deque(maxlen=history_size)
        self.missed_frames = 0
        
        # Define the Region of Interest (ROI) for a valid STOP sign.
        # It must be vertically centered and horizontally within the vehicle's path.
        self.roi_x_min = config.IMG_W * 0.3  # 30% from left
        self.roi_x_max = config.IMG_W * 0.7  # 70% from left
        self.roi_y_min = config.IMG_H * 0.1  # 10% from top
        self.roi_y_max = config.IMG_H * 0.8  # 80% from top

    def update_and_evaluate(self, bbox):
        # If YOLO misses the sign, tolerate up to 2 frames before clearing buffer
        if bbox is None:
            self.missed_frames += 1
            if self.missed_frames > 2:
                self.history.clear()
            return False
        
        self.missed_frames = 0
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        # Spatial Filter: Check if the sign's center is strictly inside the ROI
        if self.roi_x_min < cx < self.roi_x_max and self.roi_y_min < cy < self.roi_y_max:
            self.history.append((cx, cy))
        else:
            # Clear history if a sign is detected but is outside the strict ROI
            self.history.clear()
            
        # Trigger Condition: Sign has been stable inside ROI for exactly 'history_size' frames
        if len(self.history) == self.history.maxlen:
            self.history.clear() # Reset immediately to prevent multiple triggers
            return True
            
        return False
    
class DynamicScaleEstimator:
    def __init__(self, initial_ppm=150.0):
        self.ppm = initial_ppm
        self.alpha = 0.15 
        self.last_feature_y = None
        self.last_time = None

    def update_scale(self, current_y, v_x, current_time):
        """
        Dynamically adjusts Pixels-Per-Meter based on the empirical constant velocity (v_x).
        This guarantees highly accurate map scaling during steady driving.
        """
        if self.last_feature_y is None or self.last_time is None:
            self.last_feature_y = current_y
            self.last_time = current_time
            return self.ppm

        dt = current_time - self.last_time
        s_real = v_x * dt # Physical distance moved (meters)

        if s_real < 0.05: 
            self.last_feature_y = current_y
            self.last_time = current_time
            return self.ppm

        s_pixel = current_y - self.last_feature_y # Downward pixel movement
          
        if 0 < s_pixel < 150:
            instant_ppm = s_pixel / s_real
            # Clamp limits to prevent extreme scaling errors
            instant_ppm = max(50.0, min(300.0, instant_ppm))
            # Exponential Moving Average for smooth scale transitions
            self.ppm = (self.alpha * instant_ppm) + ((1.0 - self.alpha) * self.ppm)
            
        self.last_feature_y = current_y
        self.last_time = current_time
        return self.ppm


# ==============================================================================
# VIRTUAL TRAJECTORY GENERATORS
# ==============================================================================
def generate_virtual_curve(direction, current_ppm):
    """ Generates an arc trajectory for sharp left/right turns. """
    waypoints = []
    radius_p = 0.6 * current_ppm
    angles = np.linspace(0, math.pi / 2, 20)
    for theta in angles:
        dx = radius_p * math.sin(theta)
        dy = radius_p * (1 - math.cos(theta))
        if direction == 'LEFT':
            waypoints.append((int(config.IMG_W / 2 - dx), int(config.IMG_H - dy)))
        else:
            waypoints.append((int(config.IMG_W / 2 + dx), int(config.IMG_H - dy)))
            
    return waypoints[::-1]

def generate_straight_curve(current_ppm, length_m=0.5):
    """ Generates a straight virtual trajectory for crossing intersections. """
    waypoints = []
    length_p = length_m * current_ppm
    num_points = 20
    for i in range(num_points):
        # Generate points from bottom of screen (car position) pointing straight up
        dy = (length_p / num_points) * i
        waypoints.append((int(config.IMG_W / 2), int(config.IMG_H - dy)))
        
    # Reverse to follow bottom-to-top logic matching Pure Pursuit expectations
    return waypoints[::-1]