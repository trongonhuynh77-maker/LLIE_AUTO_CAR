import time
import serial
import threading
import math
import config

def imu_serial_reader(ser, shared_state):
    """ Background thread to parse IMU data from Arduino """
    
    # --- CALIBRATION & FILTER SETTINGS ---
    PITCH_OFFSET = 2.94
    ALPHA = 0.2 
    
    filtered_pitch = 0.0
    first_read = True

    while shared_state['global_run_flag']:
        if ser and ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8').strip()
                if line.startswith("IMU:"):
                    parts = line.replace("IMU:", "").split(",")
                    
                    # MODIFIED: Expecting at least 3 parameters. 
                    # Using parts[-1] to safely grab Absolute Yaw regardless of Arduino firmware version (4 or 5 params).
                    if len(parts) >= 3:
                        raw_pitch = float(parts[0])
                        yaw_rate_deg = float(parts[1]) 
                        absolute_yaw = float(parts[-1]) # Safely grab the last element
                        
                        calibrated_pitch = raw_pitch - PITCH_OFFSET
                        
                        if first_read:
                            filtered_pitch = calibrated_pitch
                            first_read = False
                        else:
                            filtered_pitch = (ALPHA * calibrated_pitch) + ((1.0 - ALPHA) * filtered_pitch)
                        
                        # Share ONLY the reliable rotational data across processes
                        shared_state['pitch'] = filtered_pitch
                        shared_state['yaw_rate'] = yaw_rate_deg
                        shared_state['yaw'] = absolute_yaw 
                        
                        # Removed acc_x and acc_y completely!
            except: 
                pass
        time.sleep(0.005)
def hardware_control_process(shared_state):
    print("[PROCESS 3] Hardware Control Online.")
    ser = None
    try:
        ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.1)
        time.sleep(2)
        
        # =========================================================
        # THE FIX IS HERE: Changed L255 to L10
        # This turns the headlight to level 10 (low beam) upon startup
        # =========================================================
        ser.write(b"A85 V0 L10\n")
        
    except Exception as e:
        print(f"[PROCESS 3 WARNING] Serial error: {e}")

    threading.Thread(target=imu_serial_reader, args=(ser, shared_state), daemon=True).start()

    prev_angle, prev_speed, prev_light = -1, -1, -1
    curr_angle, curr_speed = float(config.ANGLE_CENTER), 0.0
    
    try:
        while shared_state['global_run_flag']:
            loop_start = time.time()
            
            is_running = shared_state['is_running']
            max_speed = shared_state['max_speed']
            
            # The light_pwm value is pulled from shared_state
            # Make sure you also change shared_state['light_pwm'] = 10 in your main script!
            light_pwm = shared_state['light_pwm'] 
            
            target_angle = shared_state['target_angle'] if is_running else config.ANGLE_CENTER
            target_speed = max_speed if is_running else 0

            # --- SLEW RATE LIMITER (Smooth Interpolation) ---
            # Maximum 3 degrees per 0.05s loop (60 deg/sec) to prevent mechanical shock
            MAX_STEP = 3.0 
            if curr_angle < target_angle: 
                curr_angle = min(curr_angle + MAX_STEP, float(target_angle))
            elif curr_angle > target_angle: 
                curr_angle = max(curr_angle - MAX_STEP, float(target_angle))
            
            # Speed Interpolation
            if curr_speed < target_speed: 
                curr_speed = min(curr_speed + 15.0, float(target_speed))
            elif curr_speed > target_speed: 
                curr_speed = max(curr_speed - 15.0, float(target_speed))
            
            shared_state['current_speed'] = int(curr_speed)

            s_ang, s_spd, s_lht = int(curr_angle), int(curr_speed), light_pwm
            
            if (s_ang != prev_angle or s_spd != prev_speed or s_lht != prev_light):
                if ser: 
                    ser.write(f"A{s_ang} V{s_spd} L{s_lht}\n".encode('utf-8'))
                prev_angle, prev_speed, prev_light = s_ang, s_spd, s_lht

            elapsed = time.time() - loop_start
            if elapsed < 0.05: 
                time.sleep(0.05 - elapsed)

    except KeyboardInterrupt: 
        pass
    finally:
        if ser: 
            ser.write(b"A85 V0 L0\n")
            ser.close()
        print("[PROCESS 3] Offline.")