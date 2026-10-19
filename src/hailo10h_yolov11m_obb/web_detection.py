import os
import sys
import cv2
import argparse
import time
import subprocess
import numpy as np
import threading
from fastapi import FastAPI, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
import uvicorn
import shutil
from typing import Optional, List

from py_utils.coco_utils import COCO_test_helper
import sys; sys.path.insert(0, os.path.dirname(__file__))

stop_event = threading.Event()

try:
    from py_utils.hailo_executor import HailoInfer
    HAILO_AVAILABLE = True
except ImportError as e:
    HAILO_AVAILABLE = False
    print(f"Warning: HailoRT not available ({e}), inference will fail")

# Demo defaults. YOLOv11-OBB has no on-chip NMS: the host decodes the nine raw
# heads and runs the rotated probabilistic-IoU NMS, so both sliders are live.
# The Model Zoo eval uses score_threshold=0.01 / nms_iou_thresh=0.7
# (base/yolo_obb.yaml + base/yolov8.yaml); 0.25 gives a cleaner live preview.
OBJ_THRESH = 0.25
NMS_THRESH = 0.7
IMG_SIZE = (640, 640)  # (width, height) — overridden at runtime from the .hef

# YOLOv11-OBB outputs the 15 DOTA v1 classes (base/yolo_obb.yaml: classes=15,
# labels_offset=0, host-side rotated NMS). cls_id from the host decode indexes
# this tuple directly; the names match the Model Zoo's dotav1_names.json.
DEFAULT_CLASSES = (
    "plane",               # 0
    "ship",                # 1
    "storage-tank",        # 2
    "baseball-diamond",    # 3
    "tennis-court",        # 4
    "basketball-court",    # 5
    "ground-track-field",  # 6
    "harbor",              # 7
    "bridge",              # 8
    "large-vehicle",       # 9
    "small-vehicle",       # 10
    "helicopter",          # 11
    "roundabout",          # 12
    "soccer-ball-field",   # 13
    "swimming-pool",       # 14
)

CLASSES = DEFAULT_CLASSES
_DET_OUTPUT_LOGGED = True  # Disable verbose raw NMS diagnostics in production.

def load_classes(path):
    global CLASSES
    if not path or not os.path.exists(path):
        CLASSES = DEFAULT_CLASSES
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        import re
        items = re.findall(r'"([^"]*)"', content)
        if items:
            CLASSES = tuple(items)
            print(f"Successfully loaded {len(CLASSES)} classes from {path}")
        else:
            items = [item.strip().strip('"') for item in content.split(',') if item.strip()]
            if items:
                CLASSES = tuple(items)
                print(f"Loaded {len(CLASSES)} classes from {path} (fallback parsing)")
            else:
                print(f"Warning: No classes found in {path}, using default COCO classes")
                CLASSES = DEFAULT_CLASSES
    except Exception as e:
        print(f"Error loading classes from {path}: {e}. Using default COCO classes")
        CLASSES = DEFAULT_CLASSES

class DetectionConfig:
    def __init__(self):
        self.obj_thresh = OBJ_THRESH
        self.nms_thresh = NMS_THRESH
        self.lock = threading.Lock()

    def update(self, obj_thresh, nms_thresh):
        with self.lock:
            self.obj_thresh = obj_thresh
            self.nms_thresh = nms_thresh

    def get(self):
        with self.lock:
            return self.obj_thresh, self.nms_thresh

det_config = DetectionConfig()

UPLOAD_DIR = "workspace/uploads"
OUTPUT_DIR = "workspace/outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VideoAnalyzer:
    def __init__(self, model=None, co_helper=None):
        self.model = model
        self.co_helper = co_helper
        self.is_processing = False
        self.progress = 0
        self.current_video = ""
        self.error_msg = ""
        self._stop_event = threading.Event()
        self._thread = None

    def set_engine(self, model, co_helper):
        self.model = model
        self.co_helper = co_helper

    def start_analysis(self, input_path, output_path):
        if self.is_processing:
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._process_video, args=(input_path, output_path))
        self._thread.daemon = True
        self._thread.start()
        return True

    @staticmethod
    def _open_writer(output_path, width, height, fps):
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', f'{fps}', '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-threads', '0',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output_path,
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"[VideoAnalyzer] Using ffmpeg libx264 ultrafast", flush=True)
            return proc, 'ffmpeg'
        except FileNotFoundError:
            pass
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"[VideoAnalyzer] Using cv2 mp4v (slower; install ffmpeg for 5x speedup)", flush=True)
            return out, 'mp4v'
        out.release()
        return None, None

    def _process_video(self, input_path, output_path):
        self.is_processing = True
        self.progress = 0
        self.error_msg = ""
        self.current_video = os.path.basename(input_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            self.error_msg = f"Error: Cannot open video {input_path}"
            self.is_processing = False
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            self.error_msg = "Error: Invalid total frames"
            self.is_processing = False
            cap.release()
            return
        out, kind = self._open_writer(output_path, width, height, fps)
        if out is None:
            self.error_msg = "Error: No usable video writer (ffmpeg + cv2 mp4v both failed)"
            self.is_processing = False
            cap.release()
            return
        frame_idx = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if self.model and self.co_helper:
                    processed_img, lb_info = preprocess_frame(frame, self.co_helper)
                    outputs = self.model.run(processed_img)
                    if outputs is not None:
                        obj, nms = det_config.get()
                        boxes, classes, scores = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                        if boxes is not None and len(boxes) > 0:
                            real_boxes = unletterbox_obb(boxes, lb_info)
                            h, w = frame.shape[:2]
                            real_boxes[:, 0] = np.clip(real_boxes[:, 0], 0, w - 1)
                            real_boxes[:, 1] = np.clip(real_boxes[:, 1], 0, h - 1)
                            draw_obb(frame, real_boxes, scores, classes)
                if kind == 'ffmpeg':
                    out.stdin.write(frame.tobytes())
                else:
                    out.write(frame)
                frame_idx += 1
                self.progress = int((frame_idx / total_frames) * 100)
        except Exception as e:
            self.error_msg = f"Process error: {str(e)}"
        finally:
            cap.release()
            if kind == 'ffmpeg':
                try:
                    out.stdin.close()
                except Exception:
                    pass
                try:
                    out.wait(timeout=30)
                except Exception:
                    out.kill()
            else:
                out.release()
            self.is_processing = False
            if not self.error_msg:
                self.progress = 100

    def stop(self):
        self._stop_event.set()

video_analyzer = VideoAnalyzer()

app = FastAPI(title="reComputer YOLOv11m_obb Hailo-10H")

@app.get("/api/config")
async def get_config():
    obj, nms = det_config.get()
    return {"obj_thresh": obj, "nms_thresh": nms}

@app.post("/api/config")
async def update_config(config: dict):
    det_config.update(config.get("obj_thresh", OBJ_THRESH), config.get("nms_thresh", NMS_THRESH))
    return {"status": "success"}

@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "uploaded"}

@app.get("/api/video/list")
async def list_videos():
    uploads = os.listdir(UPLOAD_DIR)
    outputs = os.listdir(OUTPUT_DIR)
    return {"uploads": uploads, "outputs": outputs}

@app.post("/api/video/analyze")
async def analyze_video(filename: str = Form(...)):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Video not found")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot open video file")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    name_base = os.path.splitext(filename)[0]
    output_filename = f"{name_base}_{width}x{height}_results.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    success = video_analyzer.start_analysis(input_path, output_path)
    if success:
        return {"status": "started", "output": output_filename}
    else:
        return {"status": "error", "message": "Already processing another video"}

@app.get("/api/video/status")
async def get_analysis_status():
    return {
        "is_processing": video_analyzer.is_processing,
        "progress": video_analyzer.progress,
        "current_video": video_analyzer.current_video,
        "error": video_analyzer.error_msg
    }

@app.get("/api/video/download/{filename}")
async def download_video(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type='video/mp4', filename=filename)

_global_model = None
_global_co_helper = None

@app.post("/api/models/yolov11m_obb/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    timestamp: Optional[float] = Form(None),
    realtime: Optional[bool] = Form(False),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None)
):
    if _global_model is None or _global_co_helper is None:
        return {"success": False, "message": "Model not initialized"}
    try:
        img = None
        source_info = ""
        if file:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            source_info = "uploaded image"
        elif video:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await video.read())
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                if timestamp is not None:
                    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = cap.read()
                if ret:
                    img = frame
                    source_info = f"video frame at {timestamp if timestamp else 0}s"
                cap.release()
            os.unlink(tmp_path)
        if img is None:
            img = frame_buffer.get_raw_frame()
            source_info = "realtime camera frame"
        if img is None:
            return {"success": False, "message": "No valid input source found (image, video, or camera)"}
        h, w = img.shape[:2]
        input_img, lb_info = preprocess_frame(img, _global_co_helper)
        outputs = _global_model.run(input_img)
        current_obj_thresh, current_nms_thresh = det_config.get()
        target_conf = conf if conf is not None else current_obj_thresh
        target_iou = iou if iou is not None else current_nms_thresh
        boxes, classes, scores = post_process_hailo(outputs, target_conf, target_iou, IMG_SIZE[1], IMG_SIZE[0])
        predictions = []
        if boxes is not None and len(boxes) > 0:
            real_boxes = unletterbox_obb(boxes, lb_info)
            real_boxes[:, 0] = np.clip(real_boxes[:, 0], 0, w - 1)
            real_boxes[:, 1] = np.clip(real_boxes[:, 1], 0, h - 1)
            for box, score, cl in zip(real_boxes, scores, classes):
                cl = int(cl)
                if cl < 0 or cl >= len(CLASSES) or CLASSES[cl] == "N/A":
                    continue
                cx, cy, bw, bh, theta = [float(v) for v in box]
                predictions.append({
                    "class": CLASSES[cl],
                    "confidence": float(score),
                    "obb": {
                        "x_center": round(cx, 1),
                        "y_center": round(cy, 1),
                        "width": round(bw, 1),
                        "height": round(bh, 1),
                        "angle": round(float(np.degrees(theta)), 2),
                    },
                })
        return {"success": True, "source": source_info, "predictions": predictions, "image": {"width": w, "height": h}}
    except Exception as e:
        return {"success": False, "message": str(e)}

class FrameBuffer:
    def __init__(self):
        self.raw = None
        self.annotated = None
        self.annotated_version = 0
        self.jpeg = None
        self.jpeg_version = 0
        self.cond = threading.Condition()
    def push_annotated(self, frame):
        with self.cond:
            self.raw = frame
            self.annotated = frame
            self.annotated_version += 1
            self.cond.notify_all()
    def wait_annotated(self, last_version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.annotated_version > last_version, timeout=timeout)
            return self.annotated, self.annotated_version
    def push_jpeg(self, jpeg_bytes):
        with self.cond:
            self.jpeg = jpeg_bytes
            self.jpeg_version += 1
            self.cond.notify_all()
    def wait_jpeg(self, last_version, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.jpeg_version > last_version, timeout=timeout)
            return self.jpeg, self.jpeg_version
    def get_raw_frame(self):
        with self.cond:
            return self.raw.copy() if self.raw is not None else None

frame_buffer = FrameBuffer()

class LatestFrameReader:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.version = 0
        self._last_read_version = 0
        self._stopped = False
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._loop, daemon=True)
    def start(self):
        self._thread.start()
        return self
    def _loop(self):
        while not stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._cond:
                self.frame = frame
                self.version += 1
                self._cond.notify_all()
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
    def read(self, timeout=1.0):
        with self._cond:
            self._cond.wait_for(
                lambda: self.version > self._last_read_version or self._stopped, timeout=timeout)
            if self.frame is None:
                return False, None
            self._last_read_version = self.version
            return True, self.frame.copy()
    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=2)

@app.get("/api/video_feed")
async def video_feed():
    def generate():
        last_v = -1
        while True:
            jpeg, last_v = frame_buffer.wait_jpeg(last_v, timeout=1.0)
            if jpeg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def index():
    return Response(content="""
    <html>
      <head>
        <title>reComputer YOLOv11m_obb · Hailo-10H</title>
        <style>
          body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px; }
          .container { max-width: 1200px; margin: 0 auto; }
          .video-box { margin: 20px auto; display: inline-block; border: 5px solid #333; border-radius: 10px; overflow: hidden; background: #000; width: 100%; max-width: 800px; }
          .controls { background: #2a2a2a; padding: 20px; border-radius: 10px; display: inline-block; text-align: left; min-width: 400px; vertical-align: top; margin: 10px; }
          .control-group { margin-bottom: 15px; }
          .control-group label { display: block; margin-bottom: 5px; font-weight: bold; }
          .slider-container { display: flex; align-items: center; gap: 15px; }
          input[type=range] { flex-grow: 1; cursor: pointer; }
          .value-display { min-width: 50px; font-family: monospace; background: #444; padding: 2px 8px; border-radius: 4px; text-align: center; }
          h1 { color: #00e676; }
          .tabs { display: flex; justify-content: center; margin-bottom: 20px; border-bottom: 2px solid #333; }
          .tab { padding: 10px 30px; cursor: pointer; border-bottom: 3px solid transparent; transition: 0.3s; font-weight: bold; }
          .tab.active { border-bottom-color: #00e676; color: #00e676; }
          .tab-content { display: none; }
          .tab-content.active { display: block; }
          .video-analysis { text-align: left; background: #2a2a2a; padding: 20px; border-radius: 10px; margin: 10px; }
          .btn { background: #00e676; color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; font-weight: bold; margin: 5px; }
          .btn:hover { background: #00c853; }
          .btn:disabled { background: #555; cursor: not-allowed; }
          .progress-container { width: 100%; background: #444; border-radius: 10px; margin: 15px 0; height: 20px; position: relative; overflow: hidden; }
          .progress-bar { height: 100%; background: #00e676; width: 0%; transition: 0.3s; }
          .progress-text { position: absolute; width: 100%; text-align: center; top: 0; left: 0; line-height: 20px; font-size: 12px; font-weight: bold; color: #fff; text-shadow: 1px 1px 2px #000; }
          table { width: 100%; border-collapse: collapse; margin-top: 15px; }
          th, td { text-align: left; padding: 10px; border-bottom: 1px solid #444; }
          th { color: #888; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>YOLOv11m_obb · CM5 + Hailo-10H</h1>
          <div class="tabs">
            <div class="tab active" onclick="showTab('realtime')">Real-time Detection</div>
            <div class="tab" onclick="showTab('analysis')">Local Video Analysis</div>
          </div>
          <div id="realtime" class="tab-content active">
            <div class="video-box">
              <img id="streamImg" src="/api/video_feed" style="max-width: 100%; height: auto;">
            </div>
            <div class="controls">
              <div class="control-group">
                <label>Confidence Threshold</label>
                <div class="slider-container">
                  <input type="range" id="confSlider" min="0.01" max="1.0" step="0.01" value="0.25">
                  <span id="confValue" class="value-display">0.25</span>
                </div>
              </div>
              <div class="control-group">
                <label>IOU Threshold (NMS on-chip — slider has no effect)</label>
                <div class="slider-container">
                  <input type="range" id="iouSlider" min="0.01" max="1.0" step="0.01" value="0.45">
                  <span id="iouValue" class="value-display">0.45</span>
                </div>
              </div>
            </div>
          </div>
          <div id="analysis" class="tab-content">
            <div class="video-analysis">
              <h3>Analyze Local Video</h3>
              <div class="control-group">
                <label>Upload New Video (.mp4)</label>
                <input type="file" id="videoUpload" accept=".mp4">
                <button class="btn" onclick="uploadVideo()">Upload</button>
              </div>
              <div id="processingArea" style="display: none;">
                <p id="statusText">Processing: <span id="currentFileName">-</span></p>
                <div class="progress-container">
                  <div id="progressBar" class="progress-bar"></div>
                  <div id="progressText" class="progress-text">0%</div>
                </div>
                <p id="errorText" style="color: #ff5252;"></p>
              </div>
              <div class="control-group">
                <label>File Management</label>
                <button class="btn" onclick="refreshFileList()">Refresh List</button>
                <table>
                  <thead><tr><th>File Name</th><th>Action</th></tr></thead>
                  <tbody id="fileTableBody"></tbody>
                </table>
              </div>
            </div>
          </div>
          <p style="color: #888; margin-top: 20px;">Streaming via FastAPI + MJPEG | Port: 8000</p>
        </div>
        <script>
          function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
            if (tabId === 'realtime') { document.getElementById('streamImg').src = '/api/video_feed'; }
            else { document.getElementById('streamImg').src = ''; refreshFileList(); }
          }
          const confSlider = document.getElementById('confSlider');
          const iouSlider = document.getElementById('iouSlider');
          const confValue = document.getElementById('confValue');
          const iouValue = document.getElementById('iouValue');
          function updateConfig() {
            const obj_thresh = parseFloat(confSlider.value);
            const nms_thresh = parseFloat(iouSlider.value);
            confValue.innerText = obj_thresh.toFixed(2);
            iouValue.innerText = nms_thresh.toFixed(2);
            fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ obj_thresh, nms_thresh }) });
          }
          confSlider.oninput = updateConfig;
          iouSlider.oninput = updateConfig;
          fetch('/api/config').then(res => res.json()).then(data => {
            confSlider.value = data.obj_thresh; iouSlider.value = data.nms_thresh;
            confValue.innerText = data.obj_thresh.toFixed(2); iouValue.innerText = data.nms_thresh.toFixed(2);
          });
          async function uploadVideo() {
            const fileInput = document.getElementById('videoUpload');
            if (!fileInput.files[0]) return alert('Please select a file');
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            const btn = event.currentTarget;
            btn.disabled = true; btn.innerText = 'Uploading...';
            try { await fetch('/api/video/upload', { method: 'POST', body: formData }); alert('Upload successful'); refreshFileList(); }
            catch (e) { alert('Upload failed'); }
            finally { btn.disabled = false; btn.innerText = 'Upload'; }
          }
          async function refreshFileList() {
            const res = await fetch('/api/video/list');
            const data = await res.json();
            const tbody = document.getElementById('fileTableBody');
            tbody.innerHTML = '';
            data.uploads.forEach(f => { const tr = document.createElement('tr'); tr.innerHTML = `<td>${f} (Original)</td><td><button class="btn" onclick="analyzeVideo('${f}')">Analyze</button></td>`; tbody.appendChild(tr); });
            data.outputs.forEach(f => { const tr = document.createElement('tr'); tr.innerHTML = `<td>${f} (Analyzed)</td><td><button class="btn" onclick="window.open('/api/video/download/${f}')">Download</button></td>`; tbody.appendChild(tr); });
          }
          async function analyzeVideo(filename) {
            const formData = new FormData(); formData.append('filename', filename);
            const res = await fetch('/api/video/analyze', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.status === 'started') { startStatusPolling(); } else { alert(data.message || 'Error starting analysis'); }
          }
          let pollInterval;
          function startStatusPolling() {
            document.getElementById('processingArea').style.display = 'block';
            if (pollInterval) clearInterval(pollInterval);
            pollInterval = setInterval(async () => {
              const res = await fetch('/api/video/status'); const data = await res.json();
              document.getElementById('currentFileName').innerText = data.current_video;
              document.getElementById('progressBar').style.width = data.progress + '%';
              document.getElementById('progressText').innerText = data.progress + '%';
              document.getElementById('errorText').innerText = data.error || '';
              if (!data.is_processing && data.progress === 100) { clearInterval(pollInterval); alert('Analysis completed!'); refreshFileList(); }
              else if (!data.is_processing && data.error) { clearInterval(pollInterval); }
            }, 1000);
          }
          fetch('/api/video/status').then(res => res.json()).then(data => { if (data.is_processing) startStatusPolling(); });
        </script>
      </body>
    </html>
    """, media_type="text/html")

def run_fastapi(host, port):
    print("\n" + "="*50, flush=True)
    print("Registered Routes:", flush=True)
    for route in app.routes:
        if hasattr(route, "methods"):
            print(f"Path: {route.path:35} | Methods: {route.methods}", flush=True)
    print("="*50 + "\n", flush=True)
    sys.stdout.flush()
    uvicorn.run(app, host=host, port=port, log_level="info", log_config=None)


# ---------------------------------------------------------------------------
# YOLOv11-OBB post-processing (host decode + rotated NMS)
#
# Model Zoo base/yolo_obb.yaml (Hailo-10H v5.4.0 build) overrides
# base/yolov8.yaml for oriented detection:
#   input 1024x1024x3, normalize_in_net mean 0 / std 255, padding_color 114
#   postprocessing: device_pre_post_layers.sigmoid false, meta_arch yobb,
#                   score_threshold 0.01 — no hpp, no on-chip NMS
#   anchors: scale_factors [0.5, 0.5] (grid offsets), regression_length 15
#            (16 DFL bins), strides [8, 16, 32]
#   evaluation: DOTA v1, 15 classes, labels_offset 0
#
# The compiled HEF exposes nine raw heads — per stride one 64-channel DFL box
# tensor, one 15-channel class tensor and one 1-channel angle tensor — so the
# host decodes and runs rotated NMS. The decode mirrors the Model Zoo's own
# YoloOBBPostProc (core/postprocessing/detection/yolo_obb.py):
#   angle  theta = (sigmoid(a) - 0.25) * pi                    [-pi/4, 3pi/4)
#   boxes  DFL softmax expectation over 16 bins -> (l, t, r, b) in stride units
#          w = l + r, h = t + b, centre = anchor + R(theta) @ ((r-l)/2, (t-b)/2)
#   NMS    probabilistic IoU + Fast-NMS, class offset max_wh, max_nms 30000,
#          keep post_nms_topk
# so the published hardware mAP/FPS stay reproducible. The first-inference log
# prints the head layout and value ranges for the on-hardware check (SOP §10/§14).
# ---------------------------------------------------------------------------

OBB_STRIDES = (8, 16, 32)            # ascending stride == descending spatial size
OBB_REG_BINS = 16                    # DFL bins (regression_length 15 -> 0..15)
OBB_HEADS_PER_STRIDE = 3             # box / class / angle
NUM_CLASSES = len(DEFAULT_CLASSES)   # 15 DOTA classes
OBB_IOU_THRESH = 0.7                 # Model Zoo nms_iou_thresh (base/yolov8.yaml)
OBB_SCORE_THRESH = 0.25              # live preview cut (zoo eval uses 0.01)
OBB_MAX_NMS = 30000                  # pre-NMS candidate cap
OBB_TOP_K = 300                      # post_nms_topk
OBB_MAX_WH = 7680                    # class-offset band for the NMS


def _sigmoid(x):
    """Numerically stable logistic (raw logits can exceed float32's exp range)."""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softmax_last(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _endnodes(hailo_output):
    if hailo_output is None:
        return []
    if isinstance(hailo_output, dict):
        return list(hailo_output.values())
    if isinstance(hailo_output, (list, tuple)):
        return list(hailo_output)
    return [hailo_output]


def _shape_of(tensor):
    try:
        return tuple(np.asarray(tensor).shape)
    except ValueError:
        return ("ragged",)


def _classify_obb_heads(endnodes):
    """Group the nine raw heads into per-stride (box, class, angle) triples.

    Heads are identified by channel count and spatial size, not by name, because
    the vstream names differ between compiles: 64 channels = DFL box
    (4 * 16 bins), NUM_CLASSES = class scores, 1 channel = angle. The returned
    lists are ordered by descending spatial size, i.e. ascending stride
    (8 -> 16 -> 32), matching the Model Zoo anchor order. Returns None when the
    tensor set is not the split-head OBB layout.
    """
    boxes, scores, angles = [], [], []
    for tensor in endnodes:
        try:
            arr = np.asarray(tensor)
        except ValueError:
            return None
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        _h, _w, c = arr.shape
        if c == 4 * OBB_REG_BINS:
            boxes.append(arr)
        elif c == NUM_CLASSES:
            scores.append(arr)
        elif c == 1:
            angles.append(arr)
        else:
            return None
    if not (len(boxes) == len(scores) == len(angles) == OBB_HEADS_PER_STRIDE):
        return None

    def spatial(a):
        return a.shape[0] * a.shape[1]
    boxes.sort(key=spatial, reverse=True)
    scores.sort(key=spatial, reverse=True)
    angles.sort(key=spatial, reverse=True)
    return boxes, scores, angles


def _decode_obb_stride(box_head, score_head, angle_head, stride):
    """Decode one stride into (rboxes (F*F, 5) [cx, cy, w, h, theta_rad] in input
    pixels, class scores (F*F, NUM_CLASSES)).

    Mirrors `YoloOBBPostProc._box_decoding`: the DFL softmax expectation over the
    16 bins gives the l/t/r/b distances in stride units; the centre is the anchor
    plus the rotated half-axis offset; w/h are the summed distances. The angle is
    `(sigmoid(a) - 0.25) * pi`.
    """
    fh, fw, _ = box_head.shape
    d = box_head.reshape(fh, fw, 4, OBB_REG_BINS).astype(np.float32)
    d = _softmax_last(d)
    bins = np.arange(OBB_REG_BINS, dtype=np.float32)
    dist = (d @ bins) * stride                       # (F, F, 4) in pixels

    theta = (_sigmoid(np.asarray(angle_head, dtype=np.float32).reshape(fh, fw))
             - 0.25) * np.pi

    lt = dist[..., :2]                               # l, t distances (pixels)
    rb = dist[..., 2:]                               # r, b distances (pixels)
    xf = (rb[..., 0] - lt[..., 0]) / 2.0
    yf = (rb[..., 1] - lt[..., 1]) / 2.0
    cos, sin = np.cos(theta), np.sin(theta)
    gx, gy = np.meshgrid(np.arange(fw), np.arange(fh))
    ax = (gx.astype(np.float32) + 0.5) * stride      # anchor centre (pixels)
    ay = (gy.astype(np.float32) + 0.5) * stride
    cx = ax + xf * cos - yf * sin
    cy = ay + xf * sin + yf * cos
    w = lt[..., 0] + rb[..., 0]
    h = lt[..., 1] + rb[..., 1]

    rboxes = np.stack([cx, cy, w, h, theta], axis=-1).reshape(-1, 5)
    scores = _sigmoid(np.asarray(score_head, dtype=np.float32).reshape(-1, NUM_CLASSES))
    return rboxes.astype(np.float32), scores


def _covariance_matrix(rboxes):
    """Gaussian covariance terms of rotated boxes (xywhr); zoo/ultralytics port."""
    gbbs = np.concatenate((rboxes[:, 2:4] ** 2 / 12.0, rboxes[:, 4:5]), axis=-1)
    ax1, ax2, angle = np.split(gbbs, 3, axis=-1)
    cos, sin = np.cos(angle), np.sin(angle)
    cos2, sin2 = cos ** 2, sin ** 2
    return ax1 * cos2 + ax2 * sin2, ax1 * sin2 + ax2 * cos2, (ax1 - ax2) * cos * sin


def _batch_probiou(obb1, obb2):
    """Probabilistic IoU between rotated boxes (arXiv 2106.06072)."""
    eps = 1e-7
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (np.squeeze(x, axis=-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = _covariance_matrix(obb1)
    a2, b2, c2 = (np.squeeze(x, axis=-1)[None] for x in _covariance_matrix(obb2))

    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = (((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    t3 = (((a1 + a2) * (b1 + b2) - (c1 + c2) ** 2)
          / (4 * np.sqrt(np.maximum(0, a1 * b1 - c1 ** 2)
                         * np.maximum(0, a2 * b2 - c2 ** 2)) + eps) + eps)
    bd = np.clip(t1 + t2 + 0.5 * np.log(t3), eps, 100.0)
    return 1.0 - np.sqrt(1.0 - np.exp(-bd) + eps)


def _fast_nms(rboxes, scores, iou_thresh):
    """Fast-NMS (arXiv 1904.02689) over probabilistic IoU; returns kept indices."""
    if rboxes.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(scores)[::-1]
    rboxes = rboxes[order]
    ious = np.triu(_batch_probiou(rboxes, rboxes), k=1)
    pick = np.where((ious >= iou_thresh).sum(0) <= 0)[0]
    return order[pick]


def _obb_detections(endnodes, obj_thresh, iou_thresh, input_h, input_w):
    """Full decode + rotated NMS. Returns (rboxes, classes, scores) or None.

    rboxes are [cx, cy, w, h, theta_rad] in network-input pixels; classes are the
    best-class ids (the Model Zoo keeps one class per anchor too).
    """
    heads = _classify_obb_heads(endnodes)
    if heads is None:
        return None
    box_heads, score_heads, angle_heads = heads

    all_rboxes, all_scores = [], []
    for i, stride in enumerate(OBB_STRIDES):
        rboxes, scores = _decode_obb_stride(box_heads[i], score_heads[i],
                                            angle_heads[i], stride)
        all_rboxes.append(rboxes)
        all_scores.append(scores)
    rboxes = np.concatenate(all_rboxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    # best class per anchor, then the confidence cut
    classes = scores.argmax(axis=1).astype(np.int32)
    best = scores.max(axis=1)
    keep = best > obj_thresh
    if not keep.any():
        return None
    rboxes, best, classes = rboxes[keep], best[keep], classes[keep]

    if len(rboxes) > OBB_MAX_NMS:
        top = np.argsort(best)[::-1][:OBB_MAX_NMS]
        rboxes, best, classes = rboxes[top], best[top], classes[top]

    # class-separated rotated NMS: offset the centres by class * max_wh
    nms_boxes = rboxes.copy()
    nms_boxes[:, 0] += classes * OBB_MAX_WH
    nms_boxes[:, 1] += classes * OBB_MAX_WH
    keep_idx = _fast_nms(nms_boxes, best, iou_thresh)[:OBB_TOP_K]
    if len(keep_idx) == 0:
        return None

    rboxes, classes, best = rboxes[keep_idx], classes[keep_idx], best[keep_idx]
    rboxes = rboxes.copy()
    rboxes[:, 0] = np.clip(rboxes[:, 0], 0.0, float(input_w))
    rboxes[:, 1] = np.clip(rboxes[:, 1], 0.0, float(input_h))
    return rboxes, classes, best


def post_process_hailo(hailo_output, obj_thresh, nms_thresh, input_h, input_w):
    """YOLOv11-OBB host decode. Returns (rboxes, classes, scores):

      rboxes  (N, 5) [cx, cy, w, h, theta_rad] in network-input pixels
              (the letterboxed 1024x1024 input; the caller un-letterboxes)
      classes (N,)   DOTA class ids 0..14
      scores  (N,)   sigmoided best-class scores

    Both thresholds are live: the HEF has no on-chip NMS.
    """
    global _DET_OUTPUT_LOGGED
    if hailo_output is None:
        return None, None, None
    if obj_thresh is None or obj_thresh <= 0:
        obj_thresh = OBB_SCORE_THRESH
    iou_thresh = nms_thresh if nms_thresh and nms_thresh > 0 else OBB_IOU_THRESH

    endnodes = _endnodes(hailo_output)
    heads = _classify_obb_heads(endnodes)
    detections = (_obb_detections(endnodes, obj_thresh, iou_thresh, input_h, input_w)
                  if heads is not None else None)

    if not _DET_OUTPUT_LOGGED:
        if heads is not None:
            print(f"[YOLOv11m_obb] layout=raw OBB heads, "
                  f"outputs={[_shape_of(e) for e in endnodes]}", flush=True)

            def _rng(a):
                a = np.asarray(a, dtype=np.float32)
                return f"{float(a.min()):.3g}..{float(a.max()):.3g}"
            box_h, score_h, angle_h = heads
            print(f"[YOLOv11m_obb] box_range={[_rng(b) for b in box_h]} "
                  f"score_range={[_rng(s) for s in score_h]} "
                  f"angle_range={[_rng(a) for a in angle_h]}", flush=True)
        else:
            print(f"[YOLOv11m_obb] unexpected output layout, "
                  f"outputs={[_shape_of(e) for e in endnodes]}", flush=True)
        if detections is None:
            print("[YOLOv11m_obb] no detections parsed from this layout", flush=True)
        else:
            rboxes, classes, scores = detections
            print(f"[YOLOv11m_obb] decoded {len(rboxes)} detections, scores "
                  f"{float(scores.min()):.3f}..{float(scores.max()):.3f}, "
                  f"angles(deg) {float(np.degrees(rboxes[:, 4]).min()):.1f}.."
                  f"{float(np.degrees(rboxes[:, 4]).max()):.1f}", flush=True)
        _DET_OUTPUT_LOGGED = True

    if detections is None:
        return None, None, None
    return detections

def unletterbox_obb(rboxes, lb_info):
    """Map [cx, cy, w, h, theta_rad] boxes from the letterboxed network input back
    to the original frame: the centre goes through (ratio, dw, dh), w/h divide by
    the ratio and theta is scale invariant. `lb_info` is the (ratio, dw, dh)
    captured for this exact frame by preprocess_frame — independent of the shared
    co_helper list, which races across threads (live preview + VideoAnalyzer)."""
    if rboxes is None or len(rboxes) == 0:
        return rboxes
    ratio, dw, dh = lb_info
    out = rboxes.copy().astype(np.float32)
    out[:, 0] = (out[:, 0] - dw) / ratio
    out[:, 1] = (out[:, 1] - dh) / ratio
    out[:, 2] = out[:, 2] / ratio
    out[:, 3] = out[:, 3] / ratio
    return out


def rbox_corners(box):
    """Four corners of an [cx, cy, w, h, theta] box (ultralytics corner order)."""
    cx, cy, w, h, theta = [float(v) for v in box]
    cos, sin = np.cos(theta), np.sin(theta)
    v1 = np.array([w / 2.0 * cos, w / 2.0 * sin])
    v2 = np.array([-h / 2.0 * sin, h / 2.0 * cos])
    c = np.array([cx, cy])
    return np.stack([c + v1 + v2, c + v1 - v2, c - v1 - v2, c - v1 + v2]).astype(np.int32)


def draw_obb(image, rboxes, scores, classes):
    for box, score, cl in zip(rboxes, scores, classes):
        cl = int(cl)
        if cl < 0 or cl >= len(CLASSES) or CLASSES[cl] == "N/A":
            continue
        pts = rbox_corners(box)
        cv2.polylines(image, [pts], True, (255, 0, 0), 2)
        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        cv2.putText(image, f'{CLASSES[cl]} {float(score):.2f}',
                    (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def preprocess_frame(frame, co_helper):
    """Letterbox + BGR to RGB. Returns (img, lb_info) where lb_info captures the
    exact ratio + padding for un-letterboxing independent of shared co_helper
    state.

    The HEF uses normalize_in_net with ImageNet RGB mean/std and
    padding_color=114, so letterbox pads with gray (114) and the app feeds raw
    uint8 RGB pixels — no manual normalization.
    """
    if getattr(co_helper, "letter_box_info_list", None) is not None:
        co_helper.letter_box_info_list.clear()
    img, ratio, (dw, dh) = co_helper.letter_box(
        im=frame.copy(),
        new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
        pad_color=(114, 114, 114),
        info_need=True,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, (ratio, dw, dh)

def inference_loop(cap, model, co_helper, is_video_file, target_fps):
    fps_counter = 0
    target_period = 1.0 / target_fps if target_fps > 0 else 0
    next_time = time.time()
    try:
        while not stop_event.is_set():
            if video_analyzer.is_processing:
                time.sleep(1.0)
                next_time = time.time()
                continue
            ret, frame = cap.read()
            if not ret:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            processed_img, lb_info = preprocess_frame(frame, co_helper)
            start_time = time.time()
            outputs = model.run(processed_img)
            inference_time = time.time() - start_time
            if outputs is not None:
                obj, nms = det_config.get()
                boxes, classes, scores = post_process_hailo(outputs, obj, nms, IMG_SIZE[1], IMG_SIZE[0])
                if boxes is not None and len(boxes) > 0:
                    real_boxes = unletterbox_obb(boxes, lb_info)
                    h, w = frame.shape[:2]
                    real_boxes[:, 0] = np.clip(real_boxes[:, 0], 0, w - 1)
                    real_boxes[:, 1] = np.clip(real_boxes[:, 1], 0, h - 1)
                    draw_obb(frame, real_boxes, scores, classes)
            inf_fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_counter = 0.9 * fps_counter + 0.1 * inf_fps if fps_counter > 0 else inf_fps
            cv2.putText(frame, f'Hailo FPS: {fps_counter:.1f}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_buffer.push_annotated(frame)
            if target_period > 0:
                now = time.time()
                next_time += target_period
                sleep_for = next_time - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                elif sleep_for < -target_period:
                    next_time = now + target_period
    finally:
        stop_event.set()


def encode_loop(preview_w, preview_h, jpeg_quality):
    last_v = -1
    while not stop_event.is_set():
        frame, last_v = frame_buffer.wait_annotated(last_v, timeout=1.0)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if preview_w > 0 and preview_h > 0 and (w, h) != (preview_w, preview_h):
            preview = cv2.resize(frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
        else:
            preview = frame
        ok, buf = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            frame_buffer.push_jpeg(buf.tobytes())


def main():
    parser = argparse.ArgumentParser(description='YOLOv11m_obb on CM5 + Hailo-10H (Web Preview Mode)')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .hef model (Hailo Executable Format)')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera device ID (default: 0). Use -1 to disable camera and run web-only mode.')
    parser.add_argument('--video_path', type=str, help='Path to video file (overrides camera_id)')
    parser.add_argument('--class_path', type=str, help='Path to class_config.txt file for dynamic category loading')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Web server host')
    parser.add_argument('--port', type=int, default=8000, help='Web server port')
    parser.add_argument('--preview_width', type=int, default=1280, help='MJPEG preview width (0 to disable resize). Default 1280.')
    parser.add_argument('--preview_height', type=int, default=720, help='MJPEG preview height (0 to disable resize). Default 720.')
    parser.add_argument('--jpeg_quality', type=int, default=80, help='MJPEG preview JPEG quality 1-100. Default 80.')
    parser.add_argument('--cam_width', type=int, default=1280, help='Requested USB camera width. Default 1280.')
    parser.add_argument('--cam_height', type=int, default=720, help='Requested USB camera height. Default 720.')
    parser.add_argument('--target_fps', type=float, default=30.0, help='Cap live preview inference rate (fps). 0 = uncapped. Default 30.')
    args = parser.parse_args()

    if not HAILO_AVAILABLE:
        print("Error: HailoRT is not available. Install the hailort wheel matching your driver version.")
        return

    if args.class_path:
        load_classes(args.class_path)

    global _global_model, _global_co_helper, IMG_SIZE
    model = HailoInfer(args.model_path)
    IMG_SIZE = (model.input_w, model.input_h)
    print(f"Model input size: {model.input_w}x{model.input_h}", flush=True)
    co_helper = COCO_test_helper(enable_letter_box=True)

    _global_model = model
    _global_co_helper = co_helper
    video_analyzer.set_engine(model, co_helper)

    web_thread = threading.Thread(target=run_fastapi, args=(args.host, args.port), daemon=True)
    web_thread.start()
    print(f"Web Preview started at http://{args.host}:{args.port}", flush=True)
    print(f"Preview: {args.preview_width}x{args.preview_height} JPEG q={args.jpeg_quality} | target_fps={args.target_fps}", flush=True)
    sys.stdout.flush()

    if args.camera_id == -1 and not args.video_path:
        print("Running in Video Analysis Mode. Access Web UI to process local videos.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            model.release()
        return

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
        capture_source = cap
        is_video_file = True
    else:
        cap = cv2.VideoCapture(args.camera_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
        capture_source = None
        is_video_file = False

    if not cap.isOpened():
        print(f"Error: Cannot open video source (ID: {args.camera_id if not args.video_path else args.video_path})")
        return

    if not is_video_file:
        capture_source = LatestFrameReader(cap).start()

    inf_thread = threading.Thread(target=inference_loop,
                                  args=(capture_source, model, co_helper, is_video_file, args.target_fps),
                                  daemon=True)
    enc_thread = threading.Thread(target=encode_loop,
                                  args=(args.preview_width, args.preview_height, args.jpeg_quality),
                                  daemon=True)
    inf_thread.start()
    enc_thread.start()

    try:
        while inf_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_event.set()
        if not is_video_file:
            capture_source.stop()
        inf_thread.join(timeout=2)
        enc_thread.join(timeout=2)
        cap.release()
        model.release()

if __name__ == '__main__':
    main()
