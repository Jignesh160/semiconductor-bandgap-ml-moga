"""
run_dashboard.py
================
One-click launcher for the QuantumGap AI Web Dashboard.
Starts a local web server and automatically opens the dashboard in your default browser.
"""

import os
import sys
import webbrowser
import http.server
import socketserver

PORT = 8000

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root_dir)

    url = f"http://localhost:{PORT}/dashboard/"
    print("=" * 60)
    print("  ⚛️  QuantumGap AI — Interactive Web Dashboard")
    print(f"  🌐  Opening browser at: {url}")
    print("=" * 60)

    # Open browser
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

    # Start HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print(f"\nServing dashboard at {url}")
            print("Press Ctrl+C in this terminal to stop the server.\n")
            httpd.serve_forever()
    except OSError as e:
        if "address already in use" in str(e).lower() or "10048" in str(e):
            print(f"\nServer is already running at {url}! (Port {PORT} in use)")
            print("You can navigate directly to the URL in your browser.")
        else:
            raise

if __name__ == "__main__":
    main()
