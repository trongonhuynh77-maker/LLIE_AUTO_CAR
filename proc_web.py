import os
import cv2
import time
import math
import numpy as np
import threading
import datetime
from queue import Empty

import config
from utils import TrafficSignTracker, DynamicScaleEstimator, StopSignTracker, generate_straight_curve
from navigator import KinematicTracker, BridgeStateMachine, SteeringController, ObstacleTracker, generate_avoidance_waypoints

import web_server

# ==============================================================================
# HELPER: TRAJECTORY SPLICING (BRIDGE & LANE MERGER)
# ==============================================================================
def merge_lane_and_bridge_waypoints(lane_waypoints, bridge_bbox, img_height):
    if not bridge_bbox or len(lane_waypoints) == 0:
        return lane_waypoints

    bx1, by1, bx2, by2 = bridge_bbox
    bridge_center_x = int((bx1 + bx2) / 2.0)
    bridge_entrance_y = int(by2) 
    
    merged_waypoints = []
    
    for pt in lane_waypoints:
        if pt[1] > bridge_entrance_y:
            merged_waypoints.append(pt)
            
    merged_waypoints = sorted(merged_waypoints, key=lambda p: p[1], reverse=True)
    merged_waypoints.append((bridge_center_x, bridge_entrance_y))
    merged_waypoints.append((bridge_center_x, max(0, bridge_entrance_y - 40)))
    
    return merged_waypoints

# ==============================================================================
# MAIN WEB MASK & DECISION LOOP
# ==============================================================================
def web_mask_process(data_queue, shared_state):
    data_queue.cancel_join_thread() 
    print("[PROCESS 2] Decision & Mask Rendering Online.")
    obstacle_tracker = ObstacleTracker(min_area_thresh=10000)
    
    web_server.WebState.shared_state = shared_state
    web_server.WebState.frame_lock = threading.Lock()
    web_server.WebState.kinematics = KinematicTracker()
    web_server.WebState.target_waypoints = []
    
    kinematics = web_server.WebState.kinematics
    web_target_waypoints = web_server.WebState.target_waypoints
    web_frame_lock = web_server.WebState.frame_lock
    
    threading.Thread(target=lambda: web_server.app.run(host='0.0.0.0', port=config.FLASK_PORT, debug=False, use_reloader=False), daemon=True).start()
    
    colors = np.random.randint(0, 255, (len(config.CLASS_NAMES), 3), dtype=np.uint8)
    sign_tracker = TrafficSignTracker()
    stop_tracker = StopSignTracker(history_size=5)
    scale_estimator = DynamicScaleEstimator(initial_ppm=config.PIXELS_PER_METER) 
    
    bridge_fsm = BridgeStateMachine()
    
    cached_waypoints = []
    current_state = "LANE_KEEPING"
    current_ppm = config.PIXELS_PER_METER 
    
    post_turn_cooldown = 0.0 
    turn_start_yaw = 0.0  
    intersection_timeout = 0.0
    stop_sign_timeout = 0.0
    
    intersection_intention = "NONE"
    
    red_light_timeout = 0.0
    no_red_start_time = 0.0
    
    # THE FIX 1: Restore both X and Y smoothing variables
    smooth_sign_cx = None
    smooth_sign_cy = None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(config.BASE_DIR, "DATA"), exist_ok=True)
    video_filename = os.path.join(config.BASE_DIR, f"DATA/XE_ENHANCE_dataset_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out_video = cv2.VideoWriter(video_filename, fourcc, 30.0, (config.IMG_W, config.IMG_H))
    
    try:
        while shared_state['global_run_flag']:
            is_blind_frame = False
            try:
                raw_frame, frame, valid_boxes, proto_masks, dt_gpu = data_queue.get(timeout=0.08)
            except Empty:
                is_blind_frame = True
                dt_gpu = 0.08 
                frame = np.zeros((config.IMG_H, config.IMG_W, 3), dtype=np.uint8)
                cv2.putText(frame, "BLIND FRAME", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            shared_state['drive_state'] = current_state
            
            car_is_moving = shared_state.get('is_running', False) and shared_state.get('max_speed', 0) > 0

            vehicle_v_x, delta_yaw = kinematics.update(
                dt_gpu, shared_state.get('yaw', 0.0), car_is_moving
            )

            target_pt = None
            current_time = time.time()
            waypoints_updated_from_vision = False
            
            if not is_blind_frame:
                if out_video.isOpened(): out_video.write(frame)
                
                color_mask = np.zeros_like(frame)
                nav_waypoints = [] 
                
                left_bbox, stop_bbox, bridge_bbox, obstacle_bbox, best_feature_y = None, None, None, None, None
                straight_sign_bbox = None 
                
                bridge_detected_in_frame, red_light_detected, green_light_detected, yellow_light_detected = False, False, False, False
                cross_detected = False 

                if len(valid_boxes) > 0:
                    x1, y1, x2, y2 = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
                    scores = valid_boxes[:, 4]
                    w, h = x2 - x1, y2 - y1
                    boxes_for_nms = [[int(x1[k]), int(y1[k]), int(w[k]), int(h[k])] for k in range(len(x1))]
                    indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores.tolist(), config.CONF_THRESH, 0.45)

                    if len(indices) > 0:
                        valid_indices = indices.flatten()
                        final_boxes = valid_boxes[valid_indices]
                        
                        current_detected_classes = [config.CLASS_NAMES[int(box[5])] for box in final_boxes]
                        shared_state['current_classes'] = current_detected_classes
                        
                        for i in range(len(final_boxes)):
                            bx1, by1, bx2, by2 = final_boxes[i, :4]
                            conf, cls_id = final_boxes[i, 4], int(final_boxes[i, 5])
                            
                            if cls_id == config.LEFT_SIGN_ID:
                                left_bbox = [bx1, by1, bx2, by2]
                                best_feature_y = (by1 + by2) / 2.0 
                            elif cls_id == config.STOP_SIGN_ID: 
                                stop_bbox = [bx1, by1, bx2, by2]
                            elif cls_id == config.BRIDGE_CLASS_ID and (bx2-bx1)*(by2-by1) > 2000: 
                                bridge_detected_in_frame = True
                                bridge_bbox = [bx1, by1, bx2, by2]
                            elif cls_id == config.RED_LIGHT_CLASS_ID: red_light_detected = True
                            elif cls_id == config.GREEN_LIGHT_CLASS_ID: green_light_detected = True
                            elif cls_id == getattr(config, 'YELLO_LIGHT_ID', 11): yellow_light_detected = True
                            elif cls_id == config.CROSS_CLASS_ID: cross_detected = True
                            elif cls_id == config.OBSTACLE_CLASS_ID:
                                obstacle_bbox = [bx1, by1, bx2, by2]
                                
                            elif getattr(config, 'STRAIGHT_SIGN_ID', 9) == cls_id:
                                intersection_intention = "STRAIGHT"
                                straight_sign_bbox = [bx1, by1, bx2, by2]
                                cv2.putText(frame, "MEM: STRAIGHT", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                            elif getattr(config, 'RIGHT_SIGN_ID', 10) == cls_id:
                                intersection_intention = "RIGHT"
                                cv2.putText(frame, "MEM: RIGHT", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                            if cls_id in [config.LANE_CLASS_ID, config.BRIDGE_CLASS_ID]:
                                coeffs = final_boxes[i, 6:38]
                                
                                bx1_c, by1_c = max(0, int(bx1)), max(0, int(by1))
                                bx2_c, by2_c = min(config.IMG_W, int(bx2)), min(config.IMG_H, int(by2))
                                
                                target_w, target_h = bx2_c - bx1_c, by2_c - by1_c
                                
                                if target_w > 0 and target_h > 0:
                                    scale_x = 240.0 / config.IMG_W
                                    scale_y = 160.0 / config.IMG_H
                                    px1, py1 = int(bx1_c * scale_x), int(by1_c * scale_y)
                                    px2, py2 = int(bx2_c * scale_x), int(by2_c * scale_y)
                                    
                                    if px2 > px1 and py2 > py1:
                                        cropped_proto = proto_masks[:, py1:py2, px1:px2].reshape(32, -1)
                                        mask_logit = np.dot(coeffs, cropped_proto)
                                        mask_sig = 1.0 / (1.0 + np.exp(-mask_logit))
                                        mask_sig = mask_sig.reshape(py2 - py1, px2 - px1)
                                        
                                        mask_patch = cv2.resize(mask_sig, (target_w, target_h))
                                        binary_patch = mask_patch > 0.5
                                        
                                        if time.time() < post_turn_cooldown and cls_id == config.LANE_CLASS_ID:
                                            erase_w = int(target_w * 0.1)
                                            binary_patch[:, :erase_w] = False
                                            
                                        for r_y in range(config.IMG_H - 10, 0, -30):
                                            if by1_c <= r_y <= by2_c:
                                                local_y = int(r_y - by1_c)
                                                if local_y < target_h:
                                                    x_indices = np.nonzero(binary_patch[local_y, :])[0]
                                                    if len(x_indices) > 0:
                                                        nav_waypoints.append((int(np.mean(x_indices) + bx1_c), r_y))
                                        
                                        color_mask[by1_c:by2_c, bx1_c:bx2_c][binary_patch] = colors[cls_id]

                            box_color = [int(c) for c in colors[cls_id]]
                            cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, 2)
                            label = f"{config.CLASS_NAMES[cls_id]} {conf:.2f}"
                            cv2.putText(frame, label, (int(bx1), max(int(by1) - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    else:
                        shared_state['current_classes'] = []
                else:
                    shared_state['current_classes'] = []
                        
                frame = cv2.addWeighted(frame, 1.0, color_mask, 0.4, 0)

                b_state, b_speed, t_up, t_down = bridge_fsm.update(
                    shared_state['pitch'], dt_gpu, current_state in ["TURNING_LEFT", "TURNING_RIGHT"], bridge_detected_in_frame
                )
                
                # ==============================================================================
                # CENTRALIZED SPEED MANAGER
                # ==============================================================================
                if current_state in ["STOPPED_AT_RED", "STOPPED_AT_SIGN"]:
                    shared_state['max_speed'] = 0
                elif current_state in ["TURNING_LEFT", "TURNING_RIGHT", "CROSSING_INTERSECTION"]:
                    shared_state['max_speed'] = 120
                elif current_state == "AVOIDING_OBSTACLE":
                    shared_state['max_speed'] = 100
                else:
                    shared_state['max_speed'] = b_speed

                # 1. STOP SIGN LOGIC
                if stop_tracker.update_and_evaluate(stop_bbox) and current_state == "LANE_KEEPING": 
                    current_state = "STOPPED_AT_SIGN"
                    shared_state['max_speed'] = 0
                    stop_sign_timeout = current_time + 3.0 
                    print("[INFO] STOP Sign detected. Stopping for 3s.")

                if current_state == "STOPPED_AT_SIGN":
                    if current_time > stop_sign_timeout:
                        current_state = "LANE_KEEPING"
                        shared_state['max_speed'] = 100
                        print("[INFO] STOP timeout reached. Resuming.")

                # ==============================================================================
                # 2. TRAFFIC LIGHT STOP LOGIC (Red / Yellow)
                # ==============================================================================
                if current_state == "LANE_KEEPING":
                    if (red_light_detected or yellow_light_detected) and (intersection_intention == "STRAIGHT" or cross_detected):
                        current_state = "STOPPED_AT_RED"
                        shared_state['max_speed'] = 0
                        red_light_timeout = current_time + 10.0 
                        no_red_start_time = current_time
                        color_str = "Red" if red_light_detected else "Yellow"
                        print(f"[INFO] {color_str} Light. Intention: {intersection_intention}. Stopped.")

                # ==============================================================================
                # 3. INTERSECTION EXECUTION LOGIC (WITH AUTO-CLEAR)
                # ==============================================================================
                proceed_intersection = False
                
                if current_state == "STOPPED_AT_RED":
                    if red_light_detected or yellow_light_detected:
                        no_red_start_time = current_time
                    
                    if green_light_detected or (current_time - no_red_start_time > 2.0) or (current_time > red_light_timeout):
                        proceed_intersection = True
                        
                elif current_state == "LANE_KEEPING":
                    if green_light_detected and cross_detected:
                        proceed_intersection = True
                    elif intersection_intention == "STRAIGHT" and not red_light_detected and not yellow_light_detected:
                        proceed_intersection = True

                if proceed_intersection:
                   
                    if intersection_intention == "RIGHT":
                        shared_state['max_speed'] = 120
                        current_state = "TURNING_RIGHT"
                        turn_start_yaw = shared_state.get('yaw', 0.0) 
                        cached_waypoints = []
                        
                        A, B, C, D = -6.25, 0.0, 0.0, 0.0
                        MAX_FORWARD_DIST = 0.4
                        NUM_POINTS = 40
                        
                        for y_m in np.linspace(0, MAX_FORWARD_DIST, NUM_POINTS):
                            x_m_left = (A * (y_m ** 3)) + (B * (y_m ** 2)) + (C * y_m) + D
                            x_m_right = -x_m_left 
                            
                            dy_pixel = y_m * current_ppm
                            dx_pixel = x_m_right * current_ppm
                            pixel_x = int(config.IMG_W / 2.0 + dx_pixel)
                            pixel_y = int(config.IMG_H - dy_pixel)
                            cached_waypoints.append((pixel_x, pixel_y))
                        print("[INFO] Executing RIGHT turn.")
                        
                    else: 
                        shared_state['max_speed'] = 120
                        current_state = "CROSSING_INTERSECTION"
                        
                        if straight_sign_bbox is not None:
                            # THE FIX 2: Restore tracking for both X and Y
                            raw_cx = int((straight_sign_bbox[0] + straight_sign_bbox[2]) / 2.0)
                            raw_cy = int(straight_sign_bbox[3])
                            
                            if smooth_sign_cx is None:
                                smooth_sign_cx = raw_cx
                                smooth_sign_cy = raw_cy
                            else:
                                smooth_sign_cx = int(0.15 * raw_cx + 0.85 * smooth_sign_cx)
                                smooth_sign_cy = int(0.15 * raw_cy + 0.85 * smooth_sign_cy)
                                
                            cached_waypoints = []
                            for i in range(20):
                                ratio = i / 19.0
                                # Generate waypoints towards the overhead sign's X center
                                px = int(config.IMG_W / 2.0 + (smooth_sign_cx - config.IMG_W / 2.0) * ratio)
                                py = int(config.IMG_H - (config.IMG_H - smooth_sign_cy) * ratio)
                                cached_waypoints.append((px, py))
                            waypoints_updated_from_vision = True
                            print("[INFO] Proceeding STRAIGHT (Overhead Sign Tracking).")
                        else:
                            cached_waypoints = generate_straight_curve(current_ppm, length_m=0.8)
                            print("[INFO] Proceeding STRAIGHT (Kinematic).")
                            
                        intersection_timeout = current_time + 8.0 
                        
                    intersection_intention = "NONE"

                # ==============================================================================
                # 4. CROSSING INTERSECTION EXIT & DYNAMIC TRACKING LOGIC
                # ==============================================================================
                if current_state == "CROSSING_INTERSECTION":
                    
                    sign_reached_top = False
                    
                    if straight_sign_bbox is not None:
                        raw_cx = int((straight_sign_bbox[0] + straight_sign_bbox[2]) / 2.0)
                        raw_cy = int(straight_sign_bbox[3])
                        
                        # THE FIX 3: Check if the TOP of the sign box is within 50px of the frame top (Y = 0)
                        sign_top_y = int(straight_sign_bbox[1])
                        if sign_top_y < 50:
                            sign_reached_top = True
                            cv2.putText(frame, "SIGN THRESHOLD REACHED!", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if smooth_sign_cx is None:
                            smooth_sign_cx = raw_cx
                            smooth_sign_cy = raw_cy
                        else:
                            smooth_sign_cx = int(0.15 * raw_cx + 0.85 * smooth_sign_cx)
                            smooth_sign_cy = int(0.15 * raw_cy + 0.85 * smooth_sign_cy)
                            
                        cached_waypoints = []
                        for i in range(20):
                            ratio = i / 19.0
                            px = int(config.IMG_W / 2.0 + (smooth_sign_cx - config.IMG_W / 2.0) * ratio)
                            py = int(config.IMG_H - (config.IMG_H - smooth_sign_cy) * ratio)
                            cached_waypoints.append((px, py))
                        waypoints_updated_from_vision = True
                    
                    lower_half_points = sum(1 for wp in nav_waypoints if wp[1] > config.IMG_H // 2)
                    found_reliable_lane = lower_half_points >= 3
                    
                    # THE FIX 4: Immediately exit to LANE_KEEPING if sign crosses the 50px threshold
                    if sign_reached_top or (current_time > intersection_timeout) or found_reliable_lane:
                        current_state = "LANE_KEEPING"
                        smooth_sign_cx = None
                        smooth_sign_cy = None
                        print("[INFO] 50px Threshold reached or lane detected! Resuming LANE_KEEPING.")

                if best_feature_y is not None:
                    current_ppm = scale_estimator.update_scale(best_feature_y, vehicle_v_x, current_time)

                should_turn = sign_tracker.update_and_evaluate(left_bbox, config.IMG_W, direction='LEFT', sign_placement='RIGHT')
                
                # ==============================================================================
                # LEFT TURN LOGIC
                # ==============================================================================
                if should_turn and current_state == "LANE_KEEPING":
                    current_state = "TURNING_LEFT"
                    turn_start_yaw = shared_state.get('yaw', 0.0) 
                    shared_state['max_speed'] = 120 
                    
                    cached_waypoints = []
                    
                    A, B, C, D = -6.25, 0.0, 0.0, 0.0
                    MAX_FORWARD_DIST = 0.4 
                    NUM_POINTS = 40
                    
                    for y_m in np.linspace(0, MAX_FORWARD_DIST, NUM_POINTS):
                        x_m = (A * (y_m ** 3)) + (B * (y_m ** 2)) + (C * y_m) + D
                        
                        dy_pixel = y_m * current_ppm
                        dx_pixel = x_m * current_ppm
                        pixel_x = int(config.IMG_W / 2.0 + dx_pixel)
                        pixel_y = int(config.IMG_H - dy_pixel)
                        cached_waypoints.append((pixel_x, pixel_y))
                        
                # OBSTACLE AVOIDANCE LOGIC
                should_avoid = obstacle_tracker.update_and_evaluate(obstacle_bbox, current_time)

                if should_avoid and current_state == "LANE_KEEPING":
                    current_state = "AVOIDING_OBSTACLE"
                    shared_state['max_speed'] = 100 
                    cached_waypoints = generate_avoidance_waypoints(current_ppm)
                    
                if current_state == "AVOIDING_OBSTACLE":
                    lower_half_points = sum(1 for wp in nav_waypoints if wp[1] > config.IMG_H // 2)
                    found_reliable_lane = lower_half_points >= 3
                    
                    is_returning_phase = len(cached_waypoints) < 35
                    
                    if (len(cached_waypoints) == 0 or cached_waypoints[-1][1] > config.IMG_H) or \
                       (is_returning_phase and found_reliable_lane):
                        current_state = "LANE_KEEPING"
                        print("[INFO] Obstacle cleared or lane re-acquired. Resuming LANE_KEEPING.")

                # LANE KEEPING & TRAJECTORY SPLICING
                if current_state == "LANE_KEEPING":
                    spliced_waypoints = merge_lane_and_bridge_waypoints(nav_waypoints, bridge_bbox, config.IMG_H)
                    cached_waypoints = spliced_waypoints
                    
            # ==============================================================================
            # GLOBAL KINEMATIC WAYPOINT SHIFTING
            # ==============================================================================
            if is_blind_frame or current_state in ["TURNING_LEFT", "TURNING_RIGHT", "CROSSING_INTERSECTION", "AVOIDING_OBSTACLE"]:
                if not waypoints_updated_from_vision:
                    cached_waypoints = kinematics.shift_waypoints(cached_waypoints, dt_gpu, delta_yaw, current_ppm)
                
            # TURN EXIT LOGIC
            if current_state in ["TURNING_LEFT", "TURNING_RIGHT"]:
                current_yaw = shared_state.get('yaw', 0.0)
                yaw_diff = (current_yaw - turn_start_yaw + 180) % 360 - 180
                turn_yaw_accumulated = abs(yaw_diff)
                
                if turn_yaw_accumulated >= 75.0:
                    current_state = "LANE_KEEPING"
                    post_turn_cooldown = time.time() + 2.0
                    shared_state['max_speed'] = 100 
                else:
                    cv2.putText(frame, f"TURN YAW: {turn_yaw_accumulated:.1f}/75.0", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

            # ==============================================================================
            # COMPUTE STEERING ANGLE
            # ==============================================================================
            if len(cached_waypoints) > 0:
                ctrl_mode = "LANE_KEEPING" if current_state == "CROSSING_INTERSECTION" else current_state
                target_angle, dev, target_pt = SteeringController.compute_angle(cached_waypoints, ctrl_mode, current_ppm)
                    
                shared_state['dev'] = dev 
                shared_state['target_angle'] = target_angle
                
                web_target_waypoints.clear()
                web_target_waypoints.extend(kinematics.export_to_chart(cached_waypoints, current_ppm))
                
            if not is_blind_frame or current_state in ["TURNING_LEFT", "TURNING_RIGHT", "CROSSING_INTERSECTION", "AVOIDING_OBSTACLE"]:
                for wp in cached_waypoints: cv2.circle(frame, wp, 4, (255, 0, 0), -1)
                
                if target_pt is not None:
                    cv2.circle(frame, target_pt, 10, (0, 0, 255), -1)
                    cv2.line(frame, (int(config.IMG_W/2), config.IMG_H), target_pt, (0, 255, 255), 2)
                
                cv2.putText(frame, f"Scale: {current_ppm:.1f} px/m", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                mode_text = "NON-LINEAR" if current_state in ["LANE_KEEPING", "CROSSING_INTERSECTION"] else "PURE PURSUIT"
                
                if current_state in ["TURNING_LEFT", "TURNING_RIGHT", "AVOIDING_OBSTACLE"]:
                    speed_text = "Vx: LOCKED (Maneuver)"
                else:
                    speed_text = f"Vx: {vehicle_v_x:.2f} m/s"
                    
                cv2.putText(frame, f"CTRL: {mode_text} | STATE: {current_state}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"V: {speed_text}", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"BRIDGE: {b_state}", (20, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            with web_frame_lock: 
                web_server.WebState.frame_buffer = frame.copy()

    except KeyboardInterrupt: pass
    finally: 
        if out_video: out_video.release()
        print("[PROCESS 2] Offline.")