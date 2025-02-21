import time
import win32gui
import win32con

def close_mcafee_popup(hwnd, _):
    """Callback function to check window titles and close windows containing 'McAfee'."""
    # Check if the window is visible
    if win32gui.IsWindowVisible(hwnd):
        # Retrieve the window title
        title = win32gui.GetWindowText(hwnd)
        if "McAfee" in title:
            print(f"Found window: {title}")
            try:
                # Send a message to close the window
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                print("Popup closed.")
            except Exception as e:
                print(f"Error closing window: {e}")

def main():
    print("Starting McAfee popup blocker...")
    while True:
        # Enumerate all top-level windows and call our function for each
        win32gui.EnumWindows(close_mcafee_popup, None)
        # Sleep for a short period before scanning again
        time.sleep(1)

if __name__ == '__main__':
    main()
