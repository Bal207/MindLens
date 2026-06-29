import os
import threading
import time
import webbrowser

from app.server import flask_app


def _run_server(port):
    # Flask runs on a background thread so the GUI can own the main thread
    # (required by the native window toolkits on macOS/Windows/Linux).
    flask_app.run(host='127.0.0.1', port=port, debug=False,
                  use_reloader=False, threaded=True)


def _wait_until_up(port, timeout=20.0):
    import urllib.request
    url = f"http://127.0.0.1:{port}/health"
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=0.5)
            return True
        except Exception:
            time.sleep(0.15)
    return False


def main():
    port = int(os.environ.get('PORT', 5050))
    url = f"http://127.0.0.1:{port}"

    print("[MindLens] Starting local server...")
    threading.Thread(target=_run_server, args=(port,), daemon=True).start()
    _wait_until_up(port)

    # Launch as a real desktop app in its own native window. Falls back to the
    # default browser if the webview toolkit isn't available.
    try:
        import webview
        print("[MindLens] Opening app window...")
        webview.create_window(
            "MindLens",
            url,
            width=1440,
            height=920,
            min_size=(1024, 720),
        )
        webview.start()  # blocks until the window is closed
    except Exception as e:
        print(f"[MindLens] Native window unavailable ({e}); opening in browser: {url}")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
