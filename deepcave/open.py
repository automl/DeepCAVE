import sys
import webbrowser
from threading import Timer

DELAY = 4


if __name__ == "__main__":
    if "--address" in sys.argv:
        address = sys.argv[sys.argv.index("--address") + 1]

    if "--port" in sys.argv:
        port = sys.argv[sys.argv.index("--port") + 1]

    # Open the link in browser
    def open_browser() -> None:
        webbrowser.open_new(f"http://{address}:{port}")

    Timer(DELAY, open_browser).start()
