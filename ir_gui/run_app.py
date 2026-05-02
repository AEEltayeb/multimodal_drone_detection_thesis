"""
IR Drone Detection System - Desktop Launcher
Double-click to start the app and open in browser.
"""
import subprocess
import time
import sys
import os
import webbrowser
import signal
import urllib.request
import urllib.error

# Prevent recursive spawning
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
else:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

processes = []

def cleanup(signum=None, frame=None):
    """Clean up all subprocesses"""
    print("\nShutting down...")
    for p in processes:
        try:
            p.terminate()
        except:
            pass
    sys.exit(0)

def wait_for_backend(max_attempts=30, interval=1.0):
    """Wait for backend to be ready by polling health endpoint."""
    print("  Waiting for backend to be ready...")
    for i in range(max_attempts):
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/api/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"  Backend ready after {i+1} seconds")
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            pass
        time.sleep(interval)
    return False

def kill_port(port):
    """Kill any process using the specified port (Windows)."""
    try:
        result = subprocess.run(
            f'netstat -ano | findstr :{port} | findstr LISTENING',
            shell=True, capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    if pid.isdigit():
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True,
                                       capture_output=True)
    except:
        pass

def main():
    print("=" * 50)
    print("   IR Drone Detection System")
    print("=" * 50)
    print()

    # Kill any existing processes on our ports
    print("[0/2] Cleaning up existing processes...")
    kill_port(8000)
    kill_port(5173)
    time.sleep(1)

    # Register cleanup handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Start backend (logs visible in terminal)
    print("[1/2] Starting backend server...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    processes.append(backend)

    # Wait for backend to be ready
    if not wait_for_backend():
        print("  Backend failed to start. Check dependencies.")
        print("  Try: pip install fastapi uvicorn ultralytics")
        cleanup()
        return

    # Start frontend
    print("[2/2] Starting frontend server...")
    frontend = subprocess.Popen(
        "npm run dev",
        cwd="ui",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    processes.append(frontend)
    time.sleep(3)

    # Open in default browser
    print()
    print("Opening app in browser...")
    webbrowser.open("http://localhost:5173")

    print()
    print("=" * 50)
    print("  App is running at: http://localhost:5173")
    print("  Backend API at:    http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    # Keep running until Ctrl+C
    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            if backend.poll() is not None:
                print("Backend stopped unexpectedly.")
                break
            if frontend.poll() is not None:
                print("Frontend stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()

if __name__ == "__main__":
    main()
