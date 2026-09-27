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

On first run, MediaPipe downloads the Pose Landmarker and Hand Landmarker models and caches them under `~/.cache/roomsense/`. This needs an internet connection once. To use a pose model file you already have, pass its path as `RoomSenseConfig(pose_model_path="/path/to/model.task")` when launching from Python.

Hand commands use a separate Hand Landmarker model, downloaded on first use and cached at `~/.cache/roomsense/hand_landmarker.task`. Set `hand_model_path` in `RoomSenseConfig` to use a local copy. If the hand model is unavailable, RoomSense reports that hand tracking is unavailable and continues pose and spatial tracking.

On first launch, allow camera access for the terminal or app in **System Settings → Privacy & Security → Camera**. On macOS, RoomSense selects the built-in Mac camera by device type and matches OpenCV's device ordering, so a nearby iPhone Continuity Camera is not selected accidentally. On other systems, the default camera index is `0`. Camera size, pose confidence, smoothing, and movement thresholds are configurable in `roomsense/config.py`. Set `camera_index` in `RoomSenseConfig` only when you intentionally want to select a different camera.

## Run

From the project directory with the virtual environment active, run:

```bash
source .venv/bin/activate && roomsense
```

## Calibrate the floor

1. Start RoomSense and point the camera at the visible floor area. Press `C` to freeze the current mirrored frame.
2. Click the four corners of one roughly rectangular floor area in this order: **back-left**, **back-right**, **front-right**, **front-left**. “Back” is farther from the camera; “front” is closer. The preview labels the next point and draws the selected polygon.
3. Press `R` to clear the points and start over. Press `Enter` after all four corners are selected to confirm. If the corners are invalid, reset and select them again. Press `Esc` to cancel.
4. A successful calibration is saved to `calibration.json` in the current project directory and loaded on later runs. Keep the camera framing and aspect ratio the same. If the saved calibration is invalid or the aspect ratio changes, RoomSense reports the issue and continues in V1 mode; press `C` to calibrate again.

RoomSense estimates floor contact from visible ankle landmarks, averaging both feet by visibility when possible and falling back to one reliable foot. The calibrated map reports `room_x` and `room_y` from `0.0` to `1.0`: X runs left to right, and Y runs from the back of the room to the front. V1 relative X/Y/Z remains shown separately.

## Controls

- `C`: calibrate or recalibrate the floor.
- `D`: toggle body, hand, and anonymous track continuity diagnostics, including raw/smoothed floor position, hand geometry, and re-identification candidate scores.
- `R`: reset clicked calibration points in calibration mode; reset spatial trail, direction, speed, zone state, and session distance in live mode.
- `V`: start or stop optional JSONL session recording. Recording is off at startup and closes on quit.
- `E`: start or stop a separate landmark-only evaluation capture. Evaluation capture is off at startup and closes on quit.
- `Q` or `Esc`: quit in live mode. `Esc` cancels calibration; `Q` quits from calibration mode.

The HUD shows temporary person ID and tracking state, interaction mode, pointing availability and target confidence, demo-action status, recent events, and both recording states alongside existing position and zone information. Debug mode includes hand diagnostics, arm vectors, gesture history, cooldowns, and re-identification candidate scores. The room map shows the calibrated floor rectangle, configured zones and object markers, the estimated person location, direction, and a fading trail of about four seconds. Speed and distance use normalized room units, not meters.

## Hand and finger tracking

RoomSense runs MediaPipe Hand Landmarker on the same mirrored Mac camera frame as body tracking and detects up to two hands. It retains each hand's 21 landmarks and tracks the thumb, index, middle, ring, and pinky as **extended**, **curled**, or **uncertain**. Normal mode draws the hand bones and connects an associated hand wrist to the corresponding body wrist. The association uses handedness and wrist proximity; a hand remains visible without a body link when the pose wrist is missing or too far away.

Supported stable hand states are **OPEN PALM**, **CLOSED FIST**, **POINTING**, **PEACE SIGN**, **THUMBS UP**, and **PINCHING**. A state must be observed for three consecutive frames before it changes, and three uncertain frames release the current state. The displayed pinch distance is relative to palm size, and openness is the fraction of fingers classified as extended. Pinching enters below a normalized distance of `0.16` and remains active until the distance exceeds `0.21`; `hand_gesture_enter_margin`, `hand_gesture_exit_margin`, and `hand_gesture_confirm_frames` in `RoomSenseConfig` tune those margins and confirmation time.

Press `D` to show each hand's five finger states, pinch distance, tracking-confidence estimate, palm angle and approximate normal, and body-wrist association. Confidence combines MediaPipe's handedness score with landmark validity; it is an estimate rather than a calibrated hand-detection probability. Palm orientation and distances are approximate image-relative measurements, not physical 3D coordinates.

The hand model is cached at `~/.cache/roomsense/hand_landmarker.task`. One hand-model inference runs per camera frame in addition to pose inference, so hand tracking can lower achievable frame rate depending on the Mac and camera resolution. RoomSense targets the configured `max_fps`; the preview's FPS is the runtime measurement. Reducing `camera_width` and `camera_height` in `RoomSenseConfig` reduces inference work.

## Gesture-based Mac controls

Mac controls are disabled by default and run in dry-run mode by default. Dry-run actions appear in the HUD and event feed without sending commands to macOS.

Enable real controls from the command line with `roomsense --enable-mac-controls`, or in Python with `RoomSenseConfig(mac_controls_enabled=True)`. Use `--dry-run` to force simulation. The command-line options are mutually exclusive. Real control is available only on macOS and may require Accessibility permission for the launching app.

Hold both hands above the shoulders for the configured duration to enter command mode, then repeat to leave it. Command mode exits after `command_mode_timeout_seconds` without an accepted action. Once active, swipe right or left to change tracks, hold an open palm to play or pause, move a pinched hand up or down to adjust volume, and hold a fist to cancel a gesture that is still pending. The HUD shows command mode, dry-run or real control state, recognized gesture, cooldown, confidence, and recent action results.

Hand recognition depends on framing, lighting, visibility, and MediaPipe confidence. Tune gesture timing and movement thresholds in `RoomSenseConfig`. Tests use fake observations and mocked controllers, so they do not send media or volume commands.

## Anonymous track continuity

RoomSense assigns temporary IDs such as `PERSON_001` and `PERSON_002` during one application run. A new track is `NEW`, an observed track is `TRACKED`, a missing track is `LOST`, and a confident return is `REACQUIRED`. The short-term lost-track registry expires entries after five seconds by default. Configure `reidentification_timeout_seconds`, `reidentification_confidence_threshold`, and `reidentification_ambiguity_margin` in `RoomSenseConfig`; set `reidentification_enabled=False` to disable it.

Matching combines predicted position and trajectory (45%), likely re-entry direction (25%), time since loss (15%), and coarse torso geometry (15%). A candidate must score at least `0.72` by default and lead the next candidate by at least `0.12`. If confidence is low or candidates are too close, RoomSense assigns a new anonymous ID. Press `D` to see candidate IDs, factor scores, thresholds, and the match or rejection reason.

This feature does not use face matching, clothing descriptors, frame crops, or appearance embeddings. IDs and continuity data stay in memory and are discarded on application exit or timeout; they are not written into session recordings or evaluation captures. Single-camera track continuity can miss or confuse returning people during occlusion, abrupt movement, similar body geometry, or pose errors. The current pose model tracks one person at a time, so RoomSense does not maintain simultaneous tracks for multiple visible people. These temporary labels do not identify a person in the real world.

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

## Gesture and pointing evaluation

Press `E` to record a separate `roomsense-evaluation.jsonl` capture. It stores timestamped normalized shoulder, elbow, and wrist coordinates with visibility values, along with the detector settings and enabled target definitions used for that run. It does not store camera frames or video. The capture is separate from `V` session recordings and is closed when stopped or when RoomSense quits.

After recording, add intent labels with:

```bash
python -m roomsense.evaluation annotate roomsense-evaluation.jsonl
```

Enter labels as `gesture START END NAME`, `target START END ID`, or `none START END`. Times are seconds from the first sample. Supported gestures are `SWIPE_LEFT`, `SWIPE_RIGHT`, `BOTH_HANDS_UP`, `POINT`, and `HOLD_POINT`. The command saves a `.labels.json` sidecar file. Evaluate the capture with:

```bash
python -m roomsense.evaluation evaluate roomsense-evaluation.jsonl
```

The report includes per-gesture precision and recall, false gesture and target activations per minute, target accuracy over labeled target intervals, and gesture detection latency. Precision or recall is `null` when that gesture has no examples to measure. Try candidate settings against the same labeled capture with repeated `--set NAME=VALUE` options, for example:

```bash
python -m roomsense.evaluation evaluate roomsense-evaluation.jsonl \
  --set pointing_min_extension=0.84 \
  --set target_stability_seconds=0.45
```

The report includes the settings used for that replay. Captured points have already passed the configured pose visibility filter and pose smoothing, so replay can raise the visibility cutoff but cannot lower it or change pose smoothing after capture. Compare candidate reports on separate labeled sessions before adopting thresholds; a single capture can overfit one person's movement and camera position.

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
    hand_tracker.py               MediaPipe two-hand video inference
    hand_geometry.py              camera-free landmark geometry and finger states
    hand_classifier.py            stable hand states and body-wrist association
    reidentification.py            in-memory anonymous lost-track matching
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
    evaluation_control.py         E-key landmark-capture lifecycle
  evaluation.py                   camera-free landmark replay, labels, and metrics
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
