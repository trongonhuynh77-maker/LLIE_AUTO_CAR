import os
import signal
import time
import multiprocessing as mp

import config
from proc_vision import camera_process, gpu_inference_process
from proc_web import web_mask_process
from proc_hardware import hardware_control_process

if __name__ == '__main__':
    # Initialize multi-processing setup
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    
    # Shared memory dictionary for inter-process communication
    shared_state = manager.dict({
        'global_run_flag': True,  
        'is_running': False,      
        'max_speed': 100,
        'light_pwm': 10,
        'ai_fps': 0.0,
        't_llie': 0.0,
        't_yolo': 0.0,
        'target_angle': config.ANGLE_CENTER,
        'current_speed': 0,
        'dev': 0.0,
        
        # Drive state for Web UI telemetry
        'drive_state': 'IDLE',    
        
        # IMU Telemetry
        'acc_x': 0.0,
        'yaw_rate': 0.0,
        'pitch': 0.0
    })

    # Data pipes between processes
    cam_queue = mp.Queue(maxsize=1)   # Pipeline for pure raw frames (P4 -> P1)
    data_queue = mp.Queue(maxsize=3)  # Pipeline for processed AI data (P1 -> P2)
    
    # Process declaration
    p4_cam = mp.Process(target=camera_process, args=(cam_queue, shared_state))
    p1_gpu = mp.Process(target=gpu_inference_process, args=(cam_queue, data_queue, shared_state))
    p2_web = mp.Process(target=web_mask_process, args=(data_queue, shared_state))
    p3_hw  = mp.Process(target=hardware_control_process, args=(shared_state,))

    def graceful_exit(sig, frame):
        print("\n[SYSTEM] Ctrl+C Received. Initiating Kill Sequence...")
        
        # 1. Signal all processes to stop their loops
        shared_state['global_run_flag'] = False
        
        # 2. Give Hardware Process a 0.5s window to send the "Stop Engine" command safely
        time.sleep(0.5) 
        
        # 3. Force kill all processes to bypass any deadlocks
        print("[SYSTEM] Executing Force Terminate...")
        try:
            if p4_cam.is_alive(): p4_cam.terminate()
            if p1_gpu.is_alive(): p1_gpu.terminate()
            if p2_web.is_alive(): p2_web.terminate()
            if p3_hw.is_alive(): p3_hw.terminate()
        except:
            pass
            
        print("[SYSTEM] Teardown Complete. Goodbye.")
        os._exit(0) # Aggressive exit, bypassing Python's garbage collector
        
    # Catch Ctrl+C signal
    signal.signal(signal.SIGINT, graceful_exit)

    print("="*50)
    print("🚀 MULTI-PROCESS MODULAR SYSTEM ONLINE 🚀")
    print(f"Web Dashboard: http://<JETSON_IP>:{config.FLASK_PORT}")
    print("="*50)

    # Start all parallel processes
    p4_cam.start()
    p1_gpu.start()
    p2_web.start()
    p3_hw.start()

    # Keep main thread alive waiting for subprocesses
    p4_cam.join()
    p1_gpu.join()
    p2_web.join()
    p3_hw.join()