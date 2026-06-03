import threading
import time
import webbrowser
import os
from app.server import flask_app

def open_browser(port):
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    print(f"\n[MindLens] Opening dashboard in your web browser: {url}")
    webbrowser.open(url)

def main():
    port = int(os.environ.get('PORT', 5050))
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    print("[MindLens] Starting local web server...")
    flask_app.run(host='127.0.0.1', port=port, debug=False)

if __name__ == '__main__':
    main()