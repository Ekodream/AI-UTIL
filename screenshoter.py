# Class for handling screenshot events

from PIL import ImageGrab

class screenshoter:
    def __init__(self):
        pass

    def take_screenshot(self, filename):
        # Capture the screen
        screenshot = ImageGrab.grab()
        # Save the screenshot to the specified file
        screenshot.save(filename)