import os

# ========================================================================
# GLOBAL PATHS & ENVIRONMENT
# ========================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLIE_ENGINE_PATH = os.path.join(BASE_DIR, "liteIE.engine")
YOLO_ENGINE_PATH = os.path.join(BASE_DIR, "best10.engine")

# ========================================================================
# AI & CAMERA CONFIGURATION
# ========================================================================
CLASS_NAMES = ['bridge', 'car',  'cross', 'green_light', 'lane', 'lane-1', 'left', 'red_light', 'right', 'stop', 'straight', 'yellow_light']
BRIDGE_CLASS_ID = 0
LANE_CLASS_ID = 4  
STOP_SIGN_ID = 9
LEFT_SIGN_ID = 6
CROSS_CLASS_ID = 2 
GREEN_LIGHT_CLASS_ID = 3
RED_LIGHT_CLASS_ID = 7
OBSTACLE_CLASS_ID = 1
CONF_THRESH = 0.5
IMG_W, IMG_H = 960, 640
STRAIGHT_SIGN_ID = 10
RIGHT_SIGN_ID = 8
YELLO_LIGHT_ID = 11

# ========================================================================
# HARDWARE & CONTROL CONSTANTS
# ========================================================================
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 115200

ANGLE_CENTER = 85.0
PIXELS_PER_METER = 187.05
WHEELBASE = 0.19          
LOOKAHEAD = 0.35          
STEERING_RATIO = 2.5      

FLASK_PORT = 5000

# ========================================================================
# 1. MAIN WEB DASHBOARD TEMPLATE (ALL-IN-ONE)
# ========================================================================
HTML_PAGE = """
<html>
    <head>
        <title>CAR HMI Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-track { background: #1e1e1e; border-radius: 4px; }
            ::-webkit-scrollbar-thumb { background: #555; border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: #777; }

            body { background: #121212; color: #fff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 15px; height: 100vh; box-sizing: border-box; overflow: hidden; }
            .header { text-align: center; margin-bottom: 15px; height: 35px; }
            .header h2 { margin: 0; color: #ddd; letter-spacing: 2px; font-size: 24px; }
            
            .main-layout { display: grid; grid-template-columns: 300px 1fr 680px; gap: 15px; height: calc(100vh - 65px); }
            
            .panel { background: #1e1e1e; padding: 15px; border-radius: 12px; border: 1px solid #333; display: flex; flex-direction: column; gap: 15px; box-sizing: border-box; overflow-y: auto; }
            .video-container { background: #000; border-radius: 12px; border: 3px solid #444; display: flex; justify-content: center; align-items: center; overflow: hidden; box-shadow: 0 8px 16px rgba(0,0,0,0.6); height: 100%; }
            .video-container img { width: 100%; height: 100%; object-fit: contain; }
            
            .right-panel { 
                background: #1e1e1e; padding: 10px; border-radius: 12px; border: 1px solid #333; 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                grid-template-rows: 1fr 1fr 1fr; 
                gap: 10px; 
                overflow: hidden; 
            }
            .chart-box { 
                background: #2a2a2a; border-radius: 8px; padding: 8px; border: 1px solid #444; 
                display: flex; flex-direction: column; 
                height: 100%; min-height: 0; box-sizing: border-box;
            }
            .chart-box h4 { margin: 0 0 5px 0; color: #ccc; font-size: 11px; text-align: center; text-transform: uppercase; letter-spacing: 1px;}
            
            .canvas-container { position: relative; width: 100%; flex: 1; min-height: 0; }
            
            .telemetry-box { background: #2a2a2a; padding: 10px; border-radius: 8px; text-align: left; font-size: 13px; line-height: 1.4; border-left: 4px solid #007bff;}
            .telemetry-box.imu { border-left-color: #ffc107; }
            .telemetry-box h4 { margin: 0 0 6px 0; font-size: 13px; color: #aaa; text-transform: uppercase;}
            .val { color: #00ff00; font-weight: bold; font-family: monospace; font-size: 15px;}
            .sub-val-container { font-size: 11px; color: #888; padding-left: 10px; border-left: 2px solid #444; margin-left: 6px; margin-bottom: 4px; }
            .sub-val { font-size: 12px; color: #f39c12; font-family: monospace; font-weight: bold; }
            
            .slider-container h4 { margin: 0 0 5px 0; font-size: 12px; color: #aaa; text-transform: uppercase;}
            input[type=range] { width: 100%; cursor: pointer;}
            
            .btn { padding: 10px; font-size: 15px; font-weight: bold; cursor: pointer; border: none; border-radius: 8px; color: white; width: 100%; transition: 0.2s;}
            .btn-run { background: #28a745; box-shadow: 0 4px 0 #1e7e34;} 
            .btn-stop { background: #dc3545; box-shadow: 0 4px 0 #bd2130;}
            .btn-save { background: #2980b9; box-shadow: 0 4px 0 #1c5980;}
            .btn-llie { background: #8e44ad; box-shadow: 0 4px 0 #6c3483; margin-top: 5px; font-size: 13px;}
            .btn-llie-off { background: #7f8c8d; box-shadow: 0 4px 0 #606b6d; margin-top: 5px; font-size: 13px;}
            .btn:active { transform: translateY(4px); box-shadow: none;}
            
            .status-text { text-align: center; font-size: 12px; font-weight: bold; margin-top: 5px; min-height: 16px;}
            .hotkeys { font-size: 11px; color:#888; text-align: center; margin-top: -5px; display: block; }
        </style>
    </head>
    <body>
        <div class="header"><h2>AUTONOMOUS CAR DASHBOARD</h2></div>
        
        <div class="main-layout">
            <div class="panel">
                <div>
                    <button id="toggleBtn" class="btn btn-run" onclick="toggleRun()">START ENGINE</button>
                    <span class="hotkeys">Hotkeys: [J] Spd Up | [K] Spd Down</span>
                </div>
                
                <div>
                    <button class="btn btn-save" onclick="autoSaveGraphs()">💾 SAVE GRAPHS</button>
                    <button id="llieBtn" class="btn btn-llie" onclick="toggleLLIE()">🌙 LLIE: ON (NIGHT)</button>
                    <div id="save-status" class="status-text"></div>
                </div>
                
                <div class="telemetry-box">
                    <h4>Vehicle Status</h4>
                    <div>Sys Mode: <span id="val_status" style="color: #dc3545; font-weight:bold;">STOPPED</span></div>
                    <div>State: <span id="val_drive_state" class="val" style="color: #00d2ff;">LANE_KEEPING</span></div>
                    <div>Vision FPS: <span id="val_fps" class="val">0.0</span></div>
                    <div class="sub-val-container">
                        &#9500; LLIE: <span id="val_t_llie" class="sub-val">0.0</span> ms<br>
                        &#9492; YOLO: <span id="val_t_yolo" class="sub-val">0.0</span> ms
                    </div>
                    <div>Steering: A<span id="val_cmd" class="val">85</span></div>
                    <div>Motor PWM: <span id="val_speed" class="val">0</span> / <span id="val_max">100</span></div>
                </div>
                
                <div class="telemetry-box imu">
                    <h4>IMU Sensor</h4>
                    <div>Pitch: <span id="val_pitch" class="val">0.00</span>&deg;</div>
                    <div>Yaw Rate: <span id="val_yaw" class="val">0.00</span>&deg;/s</div>
                </div>
                
                <hr style="border-color: #333; width: 100%; margin: 0;">
                <div class="slider-container">
                    <h4>Headlight Power</h4>
                    <input type="range" id="lightSlider" min="0" max="255" value="10" oninput="updateLight()">
                </div>
                <div class="slider-container">
                    <h4>Max Speed Limit</h4>
                    <input type="range" id="speedSlider" min="0" max="255" value="100" oninput="updateSpeed()">
                </div>
            </div>
            
            <div class="video-container"><img src="/video_feed" alt="Camera Feed"></div>
            
            <div class="right-panel">
                <div class="chart-box">
                    <h4>Kinematic Trajectory</h4>
                    <div class="canvas-container"><canvas id="trajectoryChart"></canvas></div>
                </div>
                <div class="chart-box">
                    <h4>Steering Yaw Angle (deg)</h4>
                    <div class="canvas-container"><canvas id="targetYawChart"></canvas></div>
                </div>
                <div class="chart-box">
                    <h4>Yaw Rate (deg/s)</h4>
                    <div class="canvas-container"><canvas id="yawChart"></canvas></div>
                </div>
                <div class="chart-box">
                    <h4>Pitch Angle (deg)</h4>
                    <div class="canvas-container"><canvas id="pitchChart"></canvas></div>
                </div>
                <div class="chart-box">
                    <h4>AI Detection Timeline</h4>
                    <div class="canvas-container"><canvas id="classChart"></canvas></div>
                </div>
                <div class="chart-box">
                    <h4>Model Latency (ms)</h4>
                    <div class="canvas-container"><canvas id="latencyChart"></canvas></div>
                </div>
            </div>
        </div>
        
        <script>
            // ================= CHART CONFIGURATIONS =================
            let localIsRunning = false;
            let wasRunning = false; 
            let localLLIEEnabled = true; 
            
            const MAX_POINTS = 5000; 
            const timeLabels = [];
            
            const bgPlugin = {
                id: 'customCanvasBackgroundColor',
                beforeDraw: (chart, args, options) => {
                    const {ctx} = chart;
                    ctx.save();
                    ctx.globalCompositeOperation = 'destination-over';
                    ctx.fillStyle = options.color || '#1e1e1e';
                    ctx.fillRect(0, 0, chart.width, chart.height);
                    ctx.restore();
                }
            };

            function createLineConfig(color, yMin, yMax) {
                let yAxisConfig = { min: yMin, max: yMax, ticks: { font: { size: 9 } } };
                
                return {
                    type: 'line',
                    data: { labels: timeLabels, datasets: [{ data: [], borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 0, tension: 0.2 }] },
                    options: {
                        responsive: true, maintainAspectRatio: false, animation: false,
                        layout: { padding: { left: 0, right: 0, top: 0, bottom: 0 } },
                        plugins: { legend: { display: false }, customCanvasBackgroundColor: { color: '#2a2a2a' } },
                        scales: { 
                            x: { display: true, ticks: { maxTicksLimit: 10, font: {size: 8}, color: '#777' }, grid: { color: 'rgba(255,255,255,0.05)' } }, 
                            y: yAxisConfig 
                        }
                    },
                    plugins: [bgPlugin]
                };
            }

            let trajectoryChart, targetYawChart, yawChart, pitchChart, classChart, latencyChart;
            
""" + f"            // THE FIX: Dynamically inject Python's CLASS_NAMES into JS!\n            const ALL_CLASSES = {CLASS_NAMES};" + """

            function initCharts() {
                trajectoryChart = new Chart(document.getElementById('trajectoryChart').getContext('2d'), {
                    type: 'scatter',
                    data: {
                        datasets: [
                            { label: 'IMU Est', data: [], borderColor: '#ff0000', backgroundColor: '#ff0000', showLine: true, fill: false, tension: 0.1, borderWidth: 2, pointRadius: 0 },
                            { label: 'Target', data: [], borderColor: '#0000ff', backgroundColor: 'transparent', showLine: true, fill: false, borderDash: [5, 5], pointRadius: 2, borderWidth: 1 }
                        ]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false, 
                        scales: { 
                            x: { min: -5.0, max: 5.0, title: { display: false }, ticks: { font: { size: 8 }, color: '#777' }, grid: { color: 'rgba(255,255,255,0.05)' } }, 
                            y: { min: -2.0, max: 8.0, title: { display: false }, ticks: { font: { size: 8 }, color: '#777' }, grid: { color: 'rgba(255,255,255,0.05)' } } 
                        },
                        animation: false, 
                        plugins: { 
                            legend: { display: true, position: 'top', labels: { color: '#ccc', boxWidth: 10, font: {size: 9}, padding: 5 } },
                            customCanvasBackgroundColor: { color: '#2a2a2a' }
                        }
                    },
                    plugins: [bgPlugin] 
                });

                targetYawChart = new Chart(document.getElementById('targetYawChart').getContext('2d'), createLineConfig('#f1c40f', 35, 135));
                yawChart = new Chart(document.getElementById('yawChart').getContext('2d'), createLineConfig('#3498db', -50, 50));
                pitchChart = new Chart(document.getElementById('pitchChart').getContext('2d'), createLineConfig('#e74c3c', -15, 15));
                
                // THE FIX: Expanded color palette to handle theoretically infinite YOLO classes
                const classDatasets = ALL_CLASSES.map((cls, index) => {
                    const colors = ['#f1c40f', '#e67e22', '#1abc9c', '#3498db', '#9b59b6', '#e74c3c', '#c0392b', '#e84393', '#00cec9', '#ffeaa7', '#a29bfe', '#fd79a8', '#55efc4', '#fab1a0', '#00b894'];
                    return { label: cls, data: [], backgroundColor: colors[index % colors.length], borderColor: colors[index % colors.length], pointRadius: 4, pointStyle: 'circle', showLine: false };
                });

                classChart = new Chart(document.getElementById('classChart').getContext('2d'), {
                    type: 'line',
                    data: { labels: timeLabels, datasets: classDatasets },
                    options: {
                        responsive: true, maintainAspectRatio: false, animation: false,
                        layout: { padding: { left: 0, right: 0, top: 0, bottom: 0 } },
                        plugins: { legend: { display: false }, customCanvasBackgroundColor: { color: '#2a2a2a' } },
                        scales: { 
                            x: { display: true, ticks: { maxTicksLimit: 10, font: {size: 8}, color: '#777' }, grid: { color: 'rgba(255,255,255,0.05)' } }, 
                            y: { type: 'category', labels: ALL_CLASSES, offset: true, ticks: {font: {size: 9}} } 
                        }
                    },
                    plugins: [bgPlugin]
                });
                
                latencyChart = new Chart(document.getElementById('latencyChart').getContext('2d'), {
                    type: 'line',
                    data: { 
                        labels: timeLabels, 
                        datasets: [
                            { label: 'LLIE', data: [], borderColor: '#9b59b6', backgroundColor: '#9b59b6', borderWidth: 2, pointRadius: 0, tension: 0.2 },
                            { label: 'YOLO', data: [], borderColor: '#e67e22', backgroundColor: '#e67e22', borderWidth: 2, pointRadius: 0, tension: 0.2 }
                        ] 
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false, animation: false,
                        layout: { padding: { left: 0, right: 0, top: 0, bottom: 0 } },
                        plugins: { legend: { display: true, position: 'top', labels: {color: '#ccc', boxWidth: 10, font: {size: 9}, padding: 2} }, customCanvasBackgroundColor: { color: '#2a2a2a' } },
                        scales: { 
                            x: { display: true, ticks: { maxTicksLimit: 10, font: {size: 8}, color: '#777' }, grid: { color: 'rgba(255,255,255,0.05)' } }, 
                            y: { min: 0, ticks: { font: { size: 9 } } } 
                        } 
                    },
                    plugins: [bgPlugin]
                });
            }

            // ================= CONTROL LOGIC =================
            function toggleRun() {
                let newState = !localIsRunning;
                
                if (newState === true) {
                    timeLabels.length = 0;
                    targetYawChart.data.datasets[0].data.length = 0;
                    yawChart.data.datasets[0].data.length = 0;
                    pitchChart.data.datasets[0].data.length = 0;
                    latencyChart.data.datasets[0].data.length = 0;
                    latencyChart.data.datasets[1].data.length = 0;
                    classChart.data.datasets.forEach(ds => ds.data.length = 0);
                    
                    targetYawChart.update();
                    yawChart.update();
                    pitchChart.update();
                    latencyChart.update();
                    classChart.update();
                }
                
                fetch('/api/command', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type: 'run', state: newState}) });
            }
            
            function toggleLLIE() {
                let newState = !localLLIEEnabled;
                fetch('/api/command', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type: 'llie', state: newState}) });
            }
            
            function updateLight() { fetch('/api/command', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type: 'light', value: parseInt(document.getElementById("lightSlider").value)})}); }
            function updateSpeed() { fetch('/api/command', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type: 'speed_slider', value: parseInt(document.getElementById("speedSlider").value)})}); }
            
            document.addEventListener('keydown', (e) => {
                if (['j', 'k'].includes(e.key.toLowerCase())) fetch('/api/command', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({type: 'speed', key: e.key.toLowerCase()})});
            });

            // ==============================================================================
            // HELPER: FORCE WHITE BACKGROUND & AXES BEFORE CAPTURE
            // ==============================================================================
            function captureChartWhite(chart) {
                if (!chart) return "";
                
                const origBg = chart.options.plugins?.customCanvasBackgroundColor?.color || '#1e1e1e';
                const origXDisplay = chart.options.scales?.x?.display || false;
                const origXColor = chart.options.scales?.x?.ticks?.color || '#777';
                const origLegendColor = chart.options.plugins?.legend?.labels?.color || '#ccc';

                if (!chart.options.plugins) chart.options.plugins = {};
                if (!chart.options.plugins.customCanvasBackgroundColor) chart.options.plugins.customCanvasBackgroundColor = {};
                
                chart.options.plugins.customCanvasBackgroundColor.color = '#ffffff';
                
                if (chart.options.scales && chart.options.scales.x) {
                    chart.options.scales.x.display = true;
                    if(chart.options.scales.x.ticks) chart.options.scales.x.ticks.color = '#000000';
                }
                
                if (chart.options.plugins?.legend?.labels) {
                    chart.options.plugins.legend.labels.color = '#000000';
                }
                
                Chart.defaults.color = '#000000'; 
                
                if (chart.options.scales?.y?.grid) chart.options.scales.y.grid.color = 'rgba(0,0,0,0.1)';
                if (chart.options.scales?.x?.grid) chart.options.scales.x.grid.color = 'rgba(0,0,0,0.1)';

                chart.update('none'); 
                
                const imgData = chart.toBase64Image();

                chart.options.plugins.customCanvasBackgroundColor.color = origBg;
                if (chart.options.scales && chart.options.scales.x) {
                    chart.options.scales.x.display = origXDisplay;
                    if(chart.options.scales.x.ticks) chart.options.scales.x.ticks.color = origXColor;
                }
                
                if (chart.options.plugins?.legend?.labels) {
                    chart.options.plugins.legend.labels.color = origLegendColor;
                }
                
                Chart.defaults.color = '#cccccc'; 
                
                if (chart.options.scales?.y?.grid) chart.options.scales.y.grid.color = 'rgba(255,255,255,0.05)';
                if (chart.options.scales?.x?.grid) chart.options.scales.x.grid.color = 'rgba(255,255,255,0.05)';

                chart.update('none');
                
                return imgData;
            }

            function autoSaveGraphs() {
                const statusDiv = document.getElementById('save-status');
                statusDiv.innerText = "⏳ Saving graphs...";
                statusDiv.style.color = "#f39c12";

                const payload = {
                    kinematic_trajectory: captureChartWhite(trajectoryChart), 
                    steering_target_yaw: captureChartWhite(targetYawChart),
                    yaw_rate: captureChartWhite(yawChart),
                    pitch_angle: captureChartWhite(pitchChart),
                    model_latency: captureChartWhite(latencyChart),
                    ai_detection_timeline: captureChartWhite(classChart)
                };

                fetch('/api/save_graphs', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
                }).then(res => res.json()).then(resData => {
                    if(resData.status === "success") {
                        statusDiv.innerText = "✓ Saved 6 files to DATA folder";
                        statusDiv.style.color = "#2ecc71";
                    } else {
                        statusDiv.innerText = "❌ Save Error";
                        statusDiv.style.color = "#e74c3c";
                    }
                    setTimeout(() => statusDiv.innerText = "", 4000);
                }).catch(err => {
                    statusDiv.innerText = "❌ Network Error";
                    statusDiv.style.color = "#e74c3c";
                    setTimeout(() => statusDiv.innerText = "", 4000);
                });
            }

            // ================= TELEMETRY POLLING =================
            setInterval(() => {
                fetch('/api/telemetry').then(res => res.json()).then(data => {
                    document.getElementById('val_fps').innerText = data.ai_fps.toFixed(1);
                    document.getElementById('val_cmd').innerText = data.target_angle;
                    document.getElementById('val_speed').innerText = data.current_speed;
                    document.getElementById('val_max').innerText = data.max_speed;
                    document.getElementById('val_drive_state').innerText = data.drive_state;
                    document.getElementById('val_t_llie').innerText = (data.t_llie || 0.0).toFixed(1);
                    document.getElementById('val_t_yolo').innerText = (data.t_yolo || 0.0).toFixed(1);
                    document.getElementById('val_pitch').innerText = data.pitch.toFixed(2);
                    document.getElementById('val_yaw').innerText = data.yaw_rate.toFixed(2);
                    
                    if (document.activeElement !== document.getElementById('speedSlider')) {
                        document.getElementById('speedSlider').value = data.max_speed;
                    }

                    localIsRunning = data.is_running;
                    let stat = document.getElementById('val_status');
                    stat.innerText = localIsRunning ? "ACTIVE" : "STOPPED";
                    stat.style.color = localIsRunning ? "#28a745" : "#dc3545";
                    
                    let btn = document.getElementById("toggleBtn");
                    btn.className = localIsRunning ? "btn btn-stop" : "btn btn-run";
                    btn.innerText = localIsRunning ? "EMERGENCY STOP" : "START ENGINE";
                    
                    localLLIEEnabled = data.use_llie;
                    let llieBtn = document.getElementById("llieBtn");
                    llieBtn.innerText = localLLIEEnabled ? "🌙 LLIE: ON (NIGHT)" : "☀️ LLIE: OFF (DAY)";
                    llieBtn.className = localLLIEEnabled ? "btn btn-llie" : "btn btn-llie-off";

                    if (wasRunning === true && data.is_running === false) { autoSaveGraphs(); }
                    wasRunning = data.is_running;

                    if (trajectoryChart && (data.waypoints || data.history)) {
                        trajectoryChart.data.datasets[0].data = data.history || [];
                        trajectoryChart.data.datasets[1].data = data.waypoints || [];
                        trajectoryChart.update();
                    }

                    if (!data.is_running) return; 

                    const now = new Date();
                    const timeStr = now.getSeconds() + '.' + Math.floor(now.getMilliseconds()/100);
                    
                    timeLabels.push(timeStr);
                    targetYawChart.data.datasets[0].data.push(data.target_angle);
                    yawChart.data.datasets[0].data.push(data.yaw_rate);
                    pitchChart.data.datasets[0].data.push(data.pitch);
                    
                    latencyChart.data.datasets[0].data.push(data.t_llie || 0.0);
                    latencyChart.data.datasets[1].data.push(data.t_yolo || 0.0);
                    
                    // THE FIX: Directly map backend detected classes to the graph row!
                    // No more manual mapping strings (e.g. 'green' -> 'green_light').
                    const detectedList = data.current_classes || [];
                    ALL_CLASSES.forEach((cls, idx) => {
                        if (detectedList.includes(cls)) {
                            classChart.data.datasets[idx].data.push(cls);
                        } else {
                            classChart.data.datasets[idx].data.push(null);
                        }
                    });

                    if (timeLabels.length > MAX_POINTS) {
                        timeLabels.shift();
                        targetYawChart.data.datasets[0].data.shift();
                        yawChart.data.datasets[0].data.shift();
                        pitchChart.data.datasets[0].data.shift();
                        classChart.data.datasets.forEach(ds => ds.data.shift());
                        latencyChart.data.datasets[0].data.shift();
                        latencyChart.data.datasets[1].data.shift();
                    }

                    targetYawChart.update();
                    yawChart.update();
                    pitchChart.update();
                    classChart.update();
                    latencyChart.update(); 

                }).catch(e => console.log("Telemetry wait..."));
            }, 100);

            window.onload = initCharts;
        </script>
    </body>
</html>
"""

# ========================================================================
# 2. SENSOR ANALYTICS DASHBOARD (DEPRECATED - MERGED ABOVE)
# ========================================================================
HTML_PAGE_GRAPHS = """
<html>
    <head>
        <title>Redirecting...</title>
        <meta http-equiv="refresh" content="0; url=/" />
    </head>
    <body style="background: #121212; color: #fff; text-align: center; margin-top: 50px;">
        <h3>Graphs have been integrated into the main dashboard. Redirecting...</h3>
    </body>
</html>
"""