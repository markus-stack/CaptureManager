from CaptureManager import CaptureManager
import os

# Main entry point to initialize and launch the CaptureManager using the provided configuration and optional ROS node.
# Calling manager.run() internally starts all input stream threads and orchestrates processing, output, and shutdown.
# The function itself only sets up and triggers the manager — all runtime logic and event handling is encapsulated within the class.
def main():
    node = None # Provide your ROS node here if needed
    json_path = os.path.join(os.path.dirname(__file__), 'capture_manager.json')
    manager = CaptureManager(json_path, node)
    manager.run()

if __name__ == '__main__':
    main()