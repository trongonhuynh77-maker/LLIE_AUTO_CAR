import math
import numpy as np
import config
# ==============================================================================
# KINEMATIC TRACKER (100% Fixed Rotation & Polarity)
# ==============================================================================
class KinematicTracker:
    def __init__(self):
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = math.pi / 2.0  
        self.v_x = 0.0 
        self.CONSTANT_V_X = 0.2529 
        self.previous_yaw_rad = None
        self.history = []

    def reset_mapping(self):
        self.history.clear()
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = math.pi / 2.0
        self.v_x = 0.0
        self.previous_yaw_rad = None

    def update(self, dt, absolute_yaw_deg, is_running):
        if is_running:
            self.v_x = self.CONSTANT_V_X
        else:
            self.v_x = 0.0
            
        current_yaw_rad = math.radians(absolute_yaw_deg)
        
        if self.previous_yaw_rad is None:
            self.previous_yaw_rad = current_yaw_rad
            
        # =====================================================================
        # EXACT POLARITY: BNO055 increases CW, Math needs CCW. We INVERT it.
        # So Left Turn = Positive delta_yaw
        # =====================================================================
        raw_delta = current_yaw_rad - self.previous_yaw_rad
        raw_delta = (raw_delta + math.pi) % (2 * math.pi) - math.pi
        
        delta_yaw = -raw_delta 
        
        self.previous_yaw_rad = current_yaw_rad
        
        if is_running:
            self.yaw += delta_yaw
            v_global_x = self.v_x * math.cos(self.yaw)
            v_global_y = self.v_x * math.sin(self.yaw)
            
            self.pose_x += v_global_x * dt
            self.pose_y += v_global_y * dt
            
            self.history.append({'x': self.pose_x, 'y': self.pose_y})
            if len(self.history) > 5000:  # Tăng bộ nhớ lên gấp 10 lần
                self.history.pop(0)
            
        return self.v_x, delta_yaw

    def shift_waypoints(self, waypoints, dt, delta_yaw, ppm):
        delta_px = 0
        delta_py = int((self.v_x * dt) * ppm)
        
        shifted = []
        ox, oy = config.IMG_W / 2.0, config.IMG_H
        for (wx, wy) in waypoints:
            new_x = wx - delta_px
            new_y = wy + delta_py
            
            dx = new_x - ox
            dy = new_y - oy
            
            # =================================================================
            # THE FIX: REMOVED NEGATIVE SIGNS!
            # Since delta_yaw is already mathematically correct (CCW is Positive),
            # this standard matrix naturally counter-rotates the camera points 
            # to the Right when the car turns Left.
            # =================================================================
            rot_x = ox + (dx * math.cos(delta_yaw) - dy * math.sin(delta_yaw))
            rot_y = oy + (dx * math.sin(delta_yaw) + dy * math.cos(delta_yaw))
            
            if rot_y < config.IMG_H:
                shifted.append((int(rot_x), int(rot_y)))
                
        return shifted

    def export_to_chart(self, waypoints, ppm):
        chart_wpts = []
        cos_y = math.cos(self.yaw)
        sin_y = math.sin(self.yaw)
        for pt in waypoints:
            dx_m = (pt[0] - (config.IMG_W / 2.0)) / ppm  
            dy_m = (config.IMG_H - pt[1]) / ppm          
            global_x = self.pose_x + (dy_m * cos_y) + (dx_m * sin_y)
            global_y = self.pose_y + (dy_m * sin_y) - (dx_m * cos_y)
            chart_wpts.append({'x': global_x, 'y': global_y})
        return chart_wpts

# ==============================================================================
# STEERING CONTROLLER (Exact Logic from Web Debugger)
# ==============================================================================
class SteeringController:
    @staticmethod
    def compute_angle(waypoints, current_state, ppm):
        if len(waypoints) == 0:
            return config.ANGLE_CENTER, 0.0, None

        wpts_sorted = sorted(waypoints, key=lambda pt: pt[1], reverse=True)
        
        if current_state == "LANE_KEEPING":
            roi_top = config.IMG_H * (2.0 / 3.0)
            roi_wpts = [pt for pt in wpts_sorted if pt[1] >= roi_top]
            
            if len(roi_wpts) > 0:
                avg_x = sum(pt[0] for pt in roi_wpts) / len(roi_wpts)
                avg_y = sum(pt[1] for pt in roi_wpts) / len(roi_wpts)
                target_pt = (int(avg_x), int(avg_y))
            else:
                target_pt = wpts_sorted[0] 
            
            dev = target_pt[0] - (config.IMG_W / 2.0)
            
            K_p = 0.035
            adjustment = math.copysign(1, dev) * (abs(dev) ** 1.3) * K_p
            target_angle = config.ANGLE_CENTER + adjustment
            
        else:
            # =================================================================
            # 🔥 PURE PURSUIT (Copied exactly from execute_trajectory) 🔥
            # =================================================================
            origin_x = config.IMG_W / 2.0
            origin_y = config.IMG_H
            lookahead_pixels = config.LOOKAHEAD * ppm
            
            # get_lookahead_point logic
            target_pt = wpts_sorted[-1] 
            for pt in wpts_sorted:
                dist = math.hypot(pt[0] - origin_x, pt[1] - origin_y)
                if dist >= lookahead_pixels:
                    target_pt = pt
                    break
                    
            dev = target_pt[0] - origin_x
            
            # Convert to physical meters relative to the car (car is at 0,0)
            target_x_m = dev / ppm
            target_y_m = max((origin_y - target_pt[1]) / ppm, 0.001)
            
            # alpha = math.atan2(y, x) - yaw 
            # (In camera frame, the car always faces strictly forward at PI/2)
            alpha = math.atan2(target_y_m, target_x_m) - (math.pi / 2.0)
            alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
            
            # delta_deg = math.degrees(math.atan2(2.0 * L * math.sin(alpha), L_D))
            delta_rad = math.atan2(2.0 * config.WHEELBASE * math.sin(alpha), config.LOOKAHEAD)
            delta_deg = math.degrees(delta_rad)
            
            # servo_cmd = SERVO_NEUTRAL - (delta_deg * STEERING_RATIO)
            target_angle = config.ANGLE_CENTER - (delta_deg * config.STEERING_RATIO)
        
        final_angle = int(max(35.0, min(130.0, target_angle)))
        
        return final_angle, dev, target_pt


# ==============================================================================
# BRIDGE STATE MACHINE
# ==============================================================================
class BridgeStateMachine:
    def __init__(self):
        self.state = "FLAT"
        self.t_up = 0.0
        self.t_down = 0.0
        self.PITCH_UP = 4.0      
        self.PITCH_DOWN = -4.0   
        self.PITCH_FLAT = 1.5    

    def update(self, current_pitch, dt, is_turning, bridge_detected_in_frame):
        target_speed = 100 
        if self.state == "FLAT":
            if current_pitch > self.PITCH_UP and bridge_detected_in_frame:
                self.state = "UP"
        elif self.state == "UP":
            target_speed = 180 
            if current_pitch < self.PITCH_FLAT:
                self.state = "TOP"
        elif self.state == "TOP":
            target_speed = 80 
            if current_pitch < self.PITCH_DOWN:
                self.state = "DOWN"
        elif self.state == "DOWN":
            target_speed = 50
            if current_pitch > -self.PITCH_FLAT:
                self.state = "FLAT"
        return self.state, target_speed, self.t_up, self.t_down


# ==============================================================================
# OBSTACLE AVOIDANCE TRACKER & GENERATOR
# ==============================================================================
class ObstacleTracker:
    def __init__(self, min_area_thresh=3500, history_size=3):
        self.min_area = min_area_thresh
        self.history = []
        self.history_size = history_size
        self.cooldown_timeout = 0.0 

    def update_and_evaluate(self, bbox, current_time):
        if current_time < self.cooldown_timeout:
            return False
            
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            area = (bx2 - bx1) * (by2 - by1)
            self.history.append(area > self.min_area)
        else:
            self.history.append(False)
            
        if len(self.history) > self.history_size:
            self.history.pop(0)
            
        if len(self.history) == self.history_size and all(self.history):
            self.history.clear()
            self.cooldown_timeout = current_time + 5.0 
            return True
        return False

def generate_avoidance_waypoints(ppm):
    """
    Generated from the Web Debugger 'test_obstacle' parameters.
    Ensures safe lane change to the LEFT in Vietnam traffic.
    """
    waypoints_m = []
    
    # Exact variables from your test code
    LATERAL_SHIFT = 0.20
    FORWARD_SHIFT = 0.30
    PASSING_DIST = 0.30 
    NUM_PTS = 20
    
    # Negative shift forces the car LEFT
    LATERAL_SHIFT = -LATERAL_SHIFT 
    
    # Segment 1: Leave lane
    for y_m in np.linspace(0, FORWARD_SHIFT, NUM_PTS):
        x_m = (y_m / FORWARD_SHIFT) * LATERAL_SHIFT
        waypoints_m.append((x_m, y_m))
        
    # Segment 2: Pass straight
    current_y = FORWARD_SHIFT
    for y_m in np.linspace(current_y, current_y + PASSING_DIST, NUM_PTS)[1:]:
        waypoints_m.append((LATERAL_SHIFT, y_m))
        
    # Segment 3: Return to lane
    current_y += PASSING_DIST
    for y_m in np.linspace(current_y, current_y + FORWARD_SHIFT, NUM_PTS)[1:]:
        ratio = (y_m - current_y) / FORWARD_SHIFT
        x_m = LATERAL_SHIFT - (ratio * LATERAL_SHIFT)
        waypoints_m.append((x_m, y_m))
        
    # Segment 4: Stabilize
    current_y += FORWARD_SHIFT
    for y_m in np.linspace(current_y, current_y + 0.3, 10)[1:]:
        waypoints_m.append((0.0, y_m))
        
    # Convert real-world coordinates (meters) to Image Pixels
    pixel_waypoints = []
    origin_x = config.IMG_W / 2.0
    origin_y = config.IMG_H
    
    for (x_m, y_m) in waypoints_m:
        px = int(origin_x + (x_m * ppm))
        py = int(origin_y - (y_m * ppm))
        pixel_waypoints.append((px, py))
        
    return pixel_waypoints