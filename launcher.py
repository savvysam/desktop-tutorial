"""
Cross-platform launcher — opens the app in the default browser and starts the
Flask server.  Used by PyInstaller for both the Windows .exe and the macOS .app.
"""
import sys
import threading
import webbrowser
import time

def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    t = threading.Thread(target=open_browser, daemon=True)
    t.start()

    from app import app
    app.run(host="localhost", port=5000, debug=False, use_reloader=False)
