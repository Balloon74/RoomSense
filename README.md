# RoomSense

RoomSense uses a webcam to track one person, recognize basic postures and gestures, and estimate their position on a calibrated, top-down room map. The map uses a flat-floor perspective transform, so positions are approximate and are not true 3D coordinates or measurements in meters.

## Run

Requires Python 3.11 or 3.12. From the project directory, create and activate a virtual environment, install the project, then start RoomSense:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
roomsense
```

On first run, RoomSense downloads and caches its tracking models, so an internet connection is needed. Allow camera access if prompted.

## Calibrate the room

Press `C` to freeze the camera view, then click the four corners of the visible floor in this order: back-left, back-right, front-right, front-left. Press `Enter` to save the calibration to `calibration.json`. Press `R` to reset the selected corners or `Esc` to cancel. Keep the camera framing the same on later runs.

Press `Q` or `Esc` to quit.
