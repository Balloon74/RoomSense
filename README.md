# RoomSense

RoomSense uses one ordinary RGB webcam to track one person, estimate body-relative position, recognize postures and temporal gestures, and map estimated foot contact onto a user-calibrated top-down room view. V3 adds approximate arm pointing toward manually registered image regions, interaction modes, a bounded event feed, and optional structured session recording.

RoomSense V2 maps an image point onto an assumed flat floor with a four-corner homography. It does not reconstruct true 3D or provide physical coordinates. V1 relative X/Y/Z and gesture information remain available alongside the calibrated room coordinates.

## Setup

Use Python 3.11 or 3.12. Install Python, then create a virtual environment from this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs the `roomsense` command and the compatible camera and tracking dependencies.

MediaPipe downloads the lightweight Pose Landmarker model on the first run and caches it under `~/.cache/roomsense/`. Hand commands use a separate Hand Landmarker model, downloaded on first use and cached as `~/.cache/roomsense/hand_landmarker.task`. These downloads need an internet connection once. To use a model file you already have, pass its path as `RoomSenseConfig(pose_model_path="/path/to/model.task", hand_model_path="/path/to/hand_landmarker.task")` when launching from Python. If the hand model is unavailable, RoomSense reports that hand commands are unavailable and continues pose and spatial tracking.

On first launch, allow camera access for the terminal or app in **System Settings → Privacy & Security → Camera**. On macOS, RoomSense selects the built-in Mac camera by device type and matches OpenCV's device ordering, so a nearby iPhone Continuity Camera is not selected accidentally. On other systems, the default camera index is `0`. Camera size, pose confidence, smoothing, and movement thresholds are configurable in `roomsense/config.py`. Set `camera_index` in `RoomSenseConfig` only when you intentionally want to select a different camera.

## Run

From the project directory with the virtual environment active, run:

```bash
roomsense
```

## Calibrate the floor

1. Start RoomSense and point the camera at the visible floor area. Press `C` to freeze the current mirrored frame.
2. Click the four corners of one roughly rectangular floor area in this order: **back-left**, **back-right**, **front-right**, **front-left**. “Back” is farther from the camera; “front” is closer. The preview labels the next point and draws the selected polygon.
3. Press `R` to clear the points and start over. Press `Enter` after all four corners are selected to confirm. If the corners are invalid, reset and select them again. Press `Esc` to cancel.
4. A successful calibration is saved to `calibration.json` in the current project directory and loaded on later runs. Keep the camera framing and aspect ratio the same. If the saved calibration is invalid or the aspect ratio changes, RoomSense reports the issue and continues in V1 mode; press `C` to calibrate again.

RoomSense estimates floor contact from visible ankle landmarks, averaging both feet by visibility when possible and falling back to one reliable foot. The calibrated map reports `room_x` and `room_y` from `0.0` to `1.0`: X runs left to right, and Y runs from the back of the room to the front. V1 relative X/Y/Z remains shown separately.

## Controls

- `C`: calibrate or recalibrate the floor.
- `D`: toggle debug landmarks and raw/smoothed floor-position details.
- `R`: reset clicked calibration points in calibration mode; reset spatial trail, direction, speed, zone state, and session distance in live mode.
- `V`: start or stop optional JSONL session recording. Recording is off at startup and closes on quit.
- `Q` or `Esc`: quit in live mode. `Esc` cancels calibration; `Q` quits from calibration mode.

The HUD shows tracking state, interaction mode, pointing and target confidence, demo-action status, recent events, and recording state alongside existing position and zone information. Debug mode includes arm vectors, gesture history, and cooldowns. The room map shows the calibrated floor rectangle, configured zones and object markers, the estimated person location, direction, and a fading trail of about four seconds. Speed and distance use normalized room units, not meters.

## Gesture-based Mac controls

Mac controls are disabled by default and run in **dry-run mode** by default. In dry-run mode, gestures are recognized and the HUD/event feed displays messages such as `ACTION: NEXT TRACK` and `ACTION: VOLUME UP`; macOS receives no media or volume commands.

To explicitly enable real controls from the command line, run:

```bash
roomsense --enable-mac-controls
```

Or opt in from Python with `RoomSenseConfig(mac_controls_enabled=True)`. The command-line `--dry-run` option forces simulation even when the config enables real controls. The options `--enable-mac-controls` and `--dry-run` cannot be used together. Real control is available only on macOS. RoomSense uses built-in `osascript`/System Events commands for media keys and output volume; macOS may ask you to allow the launching app (such as Terminal) under **System Settings → Privacy & Security → Accessibility**.

### Activate and leave COMMAND MODE

The HUD always shows `COMMAND MODE: OFF` or `COMMAND MODE: ON`. Hold **both hands above the shoulders** for the configured `gesture_both_hands_hold_seconds` (0.55 seconds by default) to enter command mode. Repeat the same two-hand gesture to turn it off. If no command is accepted for `command_mode_timeout_seconds` (12 seconds by default), command mode exits automatically. Ordinary movement and pointing do not extend that timeout. The HUD also shows gesture confidence, cooldown, whether controls are in dry-run or real mode, and the latest action results.

Once command mode is on, the initial hand controls are:

| Hand gesture | Result |
| --- | --- |
| Swipe right | Next track |
| Swipe left | Previous track |
| Open palm held briefly | Play or pause |
| Thumb and index finger pinched, then move hand up | Volume up |
| Thumb and index finger pinched, then move hand down | Volume down |
| Closed fist held briefly | Cancel the pending gesture/action |

The recognizer requires multiple timestamped observations, a minimum confidence of 0.75, and gesture-specific hold times. A swipe must span at least three samples and the configured minimum duration. Open palm, pinch, and fist holds default to 0.5, 0.15, and 0.35 seconds. A held pose fires once until released. Discrete actions have a 0.8-second cooldown. Pinch volume changes are quantized to 5 percentage-point steps after each 0.08 normalized vertical movement and are rate-limited to one update every 0.25 seconds. These defaults and related thresholds can be tuned in `RoomSenseConfig` in `roomsense/config.py`.

A fist clears gestures that are still being recognized; it cannot reverse a media or volume action that has already been dispatched. Hand recognition depends on visible fingers, camera framing, lighting, hand size in the image, and MediaPipe confidence. Occlusion or unusual hand orientation can prevent recognition or classify a pose incorrectly. Tune the configurable thresholds for your camera and range of motion. Tests use fake hands and mocked action controllers, so they never send real media or volume commands.

## Zones

Example `DESK`, `BED`, `DOOR`, and `CENTER` polygons are in `DEFAULT_ZONE_POLYGONS` in `roomsense/config.py`. Edit their vertices in normalized room coordinates from `0.0` to `1.0`. Polygons may overlap. RoomSense emits one `ZONE_ENTERED` and one `ZONE_LEFT` event per membership transition.

## Pointing and room objects

Room objects are manually configured. Each enabled item has an image-space polygon used for pointing matches and may have a separate normalized room-map coordinate used only to draw its map marker. The default MONITOR, DESK, BED, DOOR, LAMP, and TV entries are disabled examples. A Python launch can replace them like this:

```python
from roomsense.config import RoomObjectConfig, RoomSenseConfig
from roomsense.main import run

monitor = RoomObjectConfig(
    id="monitor",
    name="MONITOR",
    enabled=True,
    image_region=((0.62, 0.18), (0.92, 0.18), (0.92, 0.48), (0.62, 0.48)),
    room_position=(0.82, 0.18),
    interaction_radius=0.025,
)
run(RoomSenseConfig(room_objects=(monitor,)))
```

Image-region coordinates are normalized to the mirrored camera preview, with `(0, 0)` at the top-left and `(1, 1)` at the bottom-right. That region is used to match the arm's approximate 2D image ray. `room_position` is a separate `(x, y)` on the calibrated map and does not affect target matching. It is a manually placed map marker, not a measured object location. The map labels its selected-object arrow as a manual map cue.

The V3 gesture detector uses time-stamped samples: horizontal wrist movement can trigger `SWIPE_LEFT` or `SWIPE_RIGHT`; both wrists held above the shoulders trigger `BOTH_HANDS_UP`; an extended arm triggers `POINT`, then `HOLD_POINT` after its configured hold. Cooldowns prevent repeated events. Both hands up toggles `NORMAL` and `COMMAND`; command mode exits on another held gesture or after its inactivity timeout. In command mode, a configured MONITOR target plus a swipe can show harmless `NEXT` or `PREVIOUS` demo status messages. Zone, movement, gesture, pointing, and mode events are typed separately from their action handlers. Built-in actions only update in-app text; they do not run shell commands or control other applications.

Visibility, extension, target stability and hold, gesture timing, mode timeout, event-feed size, and recording settings live in `RoomSenseConfig` in `roomsense/config.py`.

## Recording and replay

Press `V` to begin writing to `session_recording_path` (default `roomsense-session.jsonl`) and press it again to flush and stop. The file contains versioned JSONL event, state-change, and rate-limited position records. It stores normalized state only; it never stores image pixels or webcam video. To inspect a session without opening a camera, run:

```bash
python -m roomsense.replay roomsense-session.jsonl
python -m roomsense.replay roomsense-session.jsonl --realtime
```

Replay prints records in file order. Invalid input stops at the first malformed line and reports its line number.

## Homography and limitations

A homography is a perspective transform between four image points and a rectangular map. It is useful for mapping a point that lies on the assumed floor plane into normalized room coordinates. It does not recover height or true 3D position, and it does not turn normalized units into meters.

Results depend on correct corner selection, camera placement and angle, lens distortion, visible feet, and reliable pose landmarks. Body occlusion or ankle misdetection can move the estimated contact point. Points outside the selected floor boundary are excluded from zones and clipped to the map edge for display. V1 relative depth continues to depend on apparent shoulder width and can be affected by body rotation, clothing, and pose.

## Architecture

```text
roomsense/
  main.py                         camera lifecycle and application loop
  config.py                       camera/tracking settings and example zone polygons
  calibration/
    camera_calibration.py         corner validation and homography
    calibration_store.py          versioned atomic JSON persistence
  tracking/
    pose_tracker.py               MediaPipe inference and landmark smoothing
    position_tracker.py           V1 normalized X/Y and relative-depth estimate
    person_state.py               V1 movement classifier
  spatial/
    floor_position.py             ankle-based floor-contact estimate
    room_transform.py             calibrated normalized room coordinates
    zones.py                      polygon containment and transition events
    position_history.py           trail, direction, normalized speed and distance
    observation_state.py          missing-foot timeout for transient spatial state
    pointing.py                   image-space arm direction and confidence
    room_objects.py               validated manual room-object configuration
    target_selection.py           stable image-region target selection
  gestures/
    gesture_detector.py           posture and hand-raise rules
    temporal_gestures.py          timestamped swipe, point, and hold gestures
  interactions/
    engine.py                     camera-free interaction state composition
    events.py                     typed event values and bounded feed
    modes.py                      NORMAL/COMMAND mode state
    actions.py                    safe in-process demo handlers
  recording/
    session_recorder.py           opt-in versioned JSONL records
    recording_control.py          V-key recording lifecycle
  replay.py                       camera-free session reader and CLI
  visualization/
    calibration_view.py           frozen-frame point selection
    overlay.py                    skeleton, HUD, and optional debug view
    room_map.py                   V1 map and calibrated room map
  utils/
    smoothing.py                  scalar and vector exponential smoothing
```

Tracking, spatial calculations, zone state, configuration, and drawing remain separate. Calibration and spatial modules can be used without opening a webcam.

## Camera-free tests

Run the camera-free test suite:

```bash
python -m unittest discover -s tests -v
```

## Limitations and next step

A single RGB camera does not provide a reliable 3D arm ray, true object localization, or physical room dimensions. Pointing depends on pose confidence, camera angle, occlusion, framing, and manually drawn image regions. Gesture thresholds may need adjustment for range of motion and camera placement. The floor homography applies only to points on the assumed floor plane; it must not project an elevated arm onto a room object. V4 should evaluate validated depth input or synchronized cameras, with calibration and accuracy measurements, before offering true 3D pointing or device integrations.
