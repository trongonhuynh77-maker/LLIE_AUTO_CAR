import os
import cv2
import time
import torch
import numpy as np
import gc
from queue import Empty
import config
from utils import TRTEngineWrapper

# ====================================================================
# PROCESS 4: DEDICATED CAMERA NODE
# ====================================================================
def camera_process(cam_queue, shared_state):
    try:
        os.sched_setaffinity(0, {1})  # Pin to CPU Core 1
        os.nice(-5)
    except Exception:
        pass

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.IMG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.IMG_H)
    
    # THE FIX 1: Strictly force hardware to 30 FPS (prevents low-light frame drops)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    print("[PROCESS 4] Camera Node Online. Locked to Core 1.")
    
    last_time = time.time()
    hardware_frames = 0

    try:
        while shared_state.get('global_run_flag', True):
            ret, frame = cap.read() 
            if not ret:
                continue
            
            # --------------------------------------------------------
            # HARDWARE FPS MONITOR: Proves physical camera limits
            # --------------------------------------------------------
            hardware_frames += 1
            if time.time() - last_time >= 1.0:
                print(f"[PROCESS 4] Hardware Camera Captured: {hardware_frames} FPS")
                hardware_frames = 0
                last_time = time.time()
            
            if cam_queue.full():
                try:
                    cam_queue.get_nowait()
                except Empty:
                    pass
            
            try:
                cam_queue.put_nowait(frame)
            except Exception:
                pass
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("[PROCESS 4] Offline.")

# ====================================================================
# PROCESS 1: GPU INFERENCE NODE
# ====================================================================
def gpu_inference_process(cam_queue, data_queue, shared_state):
    data_queue.cancel_join_thread() 
    cam_queue.cancel_join_thread()
    
    try:
        os.sched_setaffinity(0, {2})  
        os.nice(-10)                  
    except Exception as e:
        print(f"[WARN] Could not set CPU affinity/priority: {e}. Run with 'sudo'.")

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(1) 
    torch.set_num_threads(1) 
    gc.disable()  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inference_stream = torch.cuda.Stream(device=device)
    
    llie_engine = TRTEngineWrapper(config.LLIE_ENGINE_PATH, device)
    yolo_engine = TRTEngineWrapper(config.YOLO_ENGINE_PATH, device)
    
    img_rgb = np.zeros((config.IMG_H, config.IMG_W, 3), dtype=np.uint8)
    pinned_out_buffer = torch.zeros((config.IMG_H, config.IMG_W, 3), dtype=torch.uint8).pin_memory()
    tensor_in_static = torch.zeros((1, 3, config.IMG_H, config.IMG_W), dtype=torch.float32, device=device)
    
    if 'use_llie' not in shared_state:
        shared_state['use_llie'] = True
        
    start_llie = torch.cuda.Event(enable_timing=True)
    end_llie = torch.cuda.Event(enable_timing=True)
    start_yolo = torch.cuda.Event(enable_timing=True)
    end_yolo = torch.cuda.Event(enable_timing=True)
        
    frame_count = 0
    num_classes = 12  

    with torch.cuda.stream(inference_stream):
        for _ in range(5):
            _ = llie_engine(tensor_in_static)
            _ = yolo_engine(tensor_in_static)
    inference_stream.synchronize()
        
    try:
        while shared_state.get('global_run_flag', True):
            loop_start = time.time()
            
            # ====================================================================
            # THE FIX 2: Separation of "Wait" (Hardware limits) and "Pre" (CPU work)
            # ====================================================================
            t_wait_start = time.perf_counter()
            try:
                frame = cam_queue.get(timeout=0.1)
            except Empty:
                continue
            t_wait = (time.perf_counter() - t_wait_start) * 1000.0
            
            t_start_pre = time.perf_counter()
            raw_frame = cv2.resize(frame, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_NEAREST)
            cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB, dst=img_rgb)
            t_pre = (time.perf_counter() - t_start_pre) * 1000.0
            
            t_start_dispatch = time.perf_counter()
            with torch.cuda.stream(inference_stream):
                tensor_in_tmp = torch.from_numpy(img_rgb).to(device, non_blocking=True)
                tensor_in_static.copy_(tensor_in_tmp.permute(2, 0, 1).unsqueeze(0))
                tensor_in_static.div_(255.0)
                
                with torch.no_grad():
                    if shared_state.get('use_llie', True):
                        start_llie.record(stream=inference_stream)
                        llie_outs = llie_engine(tensor_in_static)
                        end_llie.record(stream=inference_stream)
                        
                        tensor_enh = next((out for out in llie_outs if len(out.shape) == 4 and out.shape[1] == 3), llie_outs[0])
                        torch.clamp(tensor_enh, 0.0, 1.0, out=tensor_enh)
                    else:
                        tensor_enh = tensor_in_static
                        shared_state['t_llie'] = 0.0 
                    
                    start_yolo.record(stream=inference_stream)
                    yolo_outs = yolo_engine(tensor_enh)
                    end_yolo.record(stream=inference_stream)
                
                img_gpu_uint8 = (tensor_enh.squeeze(0) * 255.0).to(torch.uint8).permute(1, 2, 0)
                pinned_out_buffer.copy_(img_gpu_uint8, non_blocking=True)
                
                raw_boxes_out = next((out for out in yolo_outs if len(out.shape) == 3), None)
                raw_masks_out = next((out for out in yolo_outs if len(out.shape) == 4), None)
                
            t_dispatch = (time.perf_counter() - t_start_dispatch) * 1000.0

            t_start_sync = time.perf_counter()
            inference_stream.synchronize() 
            t_sync = (time.perf_counter() - t_start_sync) * 1000.0
            
            if shared_state.get('use_llie', True):
                shared_state['t_llie'] = start_llie.elapsed_time(end_llie)
            shared_state['t_yolo'] = start_yolo.elapsed_time(end_yolo)
            
            t_start_post = time.perf_counter()
            combined_boxes = np.empty((0, 38), dtype=np.float32)
            
            if raw_boxes_out is not None:
                preds = raw_boxes_out[0].cpu().numpy()
                if preds.shape[0] < preds.shape[1]:
                    preds = preds.transpose()

                boxes_xywh = preds[:, :4]
                class_scores = preds[:, 4:4 + num_classes]
                mask_coeffs = preds[:, 4 + num_classes:]

                max_scores = np.max(class_scores, axis=1)
                class_ids = np.argmax(class_scores, axis=1)
                
                conf_mask = max_scores > config.CONF_THRESH
                
                if np.any(conf_mask):
                    boxes_xywh = boxes_xywh[conf_mask]
                    max_scores = max_scores[conf_mask]
                    class_ids = class_ids[conf_mask]
                    mask_coeffs = mask_coeffs[conf_mask]

                    half_w = boxes_xywh[:, 2] * 0.5
                    half_h = boxes_xywh[:, 3] * 0.5
                    x1 = boxes_xywh[:, 0] - half_w
                    y1 = boxes_xywh[:, 1] - half_h
                    x2 = boxes_xywh[:, 0] + half_w
                    y2 = boxes_xywh[:, 1] + half_h

                    combined_boxes = np.column_stack((x1, y1, x2, y2, max_scores, class_ids, mask_coeffs)).astype(np.float32)

            img_enh_np = pinned_out_buffer.numpy()
            final_frame = cv2.cvtColor(img_enh_np, cv2.COLOR_RGB2BGR)
            t_post = (time.perf_counter() - t_start_post) * 1000.0

            dt_gpu = time.time() - loop_start
            shared_state['ai_fps'] = 1.0 / (dt_gpu + 1e-6)

            t_start_queue = time.perf_counter()
            if not data_queue.full():
                mask_proto_np = raw_masks_out[0].cpu().numpy() if raw_masks_out is not None else None
                data_queue.put((raw_frame, final_frame, combined_boxes, mask_proto_np, dt_gpu))
            t_queue = (time.perf_counter() - t_start_queue) * 1000.0
                
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[PROFILER] Wait: {t_wait:4.1f}ms | Pre: {t_pre:4.1f}ms | Disp: {t_dispatch:4.1f}ms | Sync: {t_sync:4.1f}ms | Post: {t_post:4.1f}ms | Queue: {t_queue:4.1f}ms | FPS: {shared_state['ai_fps']:4.1f}")
                gc.collect()

    except KeyboardInterrupt: 
        pass
    finally: 
        gc.enable()
        print("[PROCESS 1] Offline.")