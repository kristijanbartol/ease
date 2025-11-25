# main.py
import threading, time, webbrowser
import uvicorn

def run_server():
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(0.7)  # give server a moment
    webbrowser.open("http://127.0.0.1:8000/")
    t.join()         # keep process alive
