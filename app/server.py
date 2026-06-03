import base64
import json
import os
import sys
import time

from flask import Flask, Response, jsonify, request, send_from_directory

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.state_manager import AppState
from app.tracker import MindLensTracker

WEBSITE_DIR = os.path.join(BASE_DIR, 'website')

flask_app = Flask(__name__, static_folder=WEBSITE_DIR, static_url_path='')
state = AppState()
tracker = MindLensTracker(state)

PLACEHOLDER_IMAGE = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


@flask_app.route('/')
def index():
    return send_from_directory(WEBSITE_DIR, 'index.html')


@flask_app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with state.lock:
                frame = state.latest_frame
            if frame is None:
                frame = PLACEHOLDER_IMAGE
                content_type = b'image/png'
            else:
                content_type = b'image/jpeg'
            yield (b'--frame\r\nContent-Type: ' + content_type + b'\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.05)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/api/stream')
def stream():
    def event_stream():
        while True:
            data = json.dumps(state.get_status_dict())
            yield f'data: {data}\n\n'
            time.sleep(0.5)
    return Response(event_stream(),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@flask_app.route('/api/start', methods=['POST'])
def api_start():
    state.reset_timers()
    with state.lock:
        state.is_running = True
        state.camera_state = "Initializing"
        state.screen_state = "Initializing"
        state.unified_state = "Neutral"
        state.latest_frame = None
    tracker.start()
    return jsonify({"ok": True})


@flask_app.route('/api/stop', methods=['POST'])
def api_stop():
    session = state.stop_session()
    return jsonify({"ok": True, "session": session})


@flask_app.route('/api/toggle/camera', methods=['POST'])
def api_toggle_camera():
    with state.lock:
        state.camera_enabled = not state.camera_enabled
        enabled = state.camera_enabled
    return jsonify({"camera_enabled": enabled})


@flask_app.route('/api/toggle/screen', methods=['POST'])
def api_toggle_screen():
    with state.lock:
        state.screen_enabled = not state.screen_enabled
        enabled = state.screen_enabled
    return jsonify({"screen_enabled": enabled})


@flask_app.route('/api/labels', methods=['POST'])
def api_labels():
    data = request.get_json()
    with state.lock:
        if 'productive' in data:
            state.custom_productive = data['productive']
        if 'distracted' in data:
            state.custom_distracted = data['distracted']
    return jsonify({"ok": True})


@flask_app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify(state.get_status_dict())


@flask_app.route('/api/history', methods=['GET'])
def api_history():
    return jsonify({"sessions": state.get_history()})


@flask_app.route('/health', methods=['GET'])
def health():
    return jsonify({"ok": True})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    flask_app.run(host='0.0.0.0', port=port, threaded=True)
