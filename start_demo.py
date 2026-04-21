import os
import sys
import time
import argparse
import subprocess
import threading
import webbrowser
import urllib.request


REQUIREMENTS = [
    "flask",
    "numpy",
    "gymnasium",
    "torch",
    "pandas",
]

BANNER = r"""
  ╔══════════════════════════════════════════════════════╗
  ║   ⚡ Smart Energy Grid — RL Optimization Dashboard   ║
  ║      Deep Q-Network · Microgrid Simulation           ║
  ╚══════════════════════════════════════════════════════╝
"""


def install_dependencies():
    print("[setup] Checking dependencies…")
    for pkg in REQUIREMENTS:
        try:
            __import__(pkg.replace("-", "_").split(">=")[0])
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ↓ Installing {pkg}…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def wait_for_server(url, timeout=20):
    """Poll server until it responds or times out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def start_server(port):
    """Start the Flask app in a subprocess."""
    script = os.path.join(os.path.dirname(__file__), "app.py")
    env = os.environ.copy()
    env["FLASK_ENV"] = "production"
    proc = subprocess.Popen(
        [sys.executable, script],
        env=env,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return proc


def main():
    parser = argparse.ArgumentParser(description="Start Smart Grid RL Demo")
    parser.add_argument("--port",       type=int,  default=5000)
    parser.add_argument("--no-install", action="store_true", help="Skip dependency install")
    parser.add_argument("--no-browser", action="store_true", help="Skip auto browser open")
    args = parser.parse_args()

    print(BANNER)

    if not args.no_install:
        install_dependencies()

    url = f"http://localhost:{args.port}"

    print(f"\n[server] Starting Flask on {url} …")
    proc = start_server(args.port)

    # Wait for server to be ready
    if wait_for_server(url):
        print(f"[server] ✓ Ready at {url}")
    else:
        print(f"[server] ⚠  Server may not be ready yet — check logs")

    if not args.no_browser:
        print("[browser] Opening dashboard…")
        webbrowser.open(url)

    print("\n[info] Press Ctrl+C to stop the server.\n")
    print("[tip]  To train the DQN agent first, run:  python train.py --episodes 500\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[server] Shutting down…")
        proc.terminate()
        proc.wait()
        print("[server] Done.")


if __name__ == "__main__":
    main()
