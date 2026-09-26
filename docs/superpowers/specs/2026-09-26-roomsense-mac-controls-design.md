# RoomSense Gesture-Based Mac Controls Design

## Goal and constraints

Add hand-shape and hand-motion commands to RoomSense while preserving its existing pose, floor, zone, pointing, interaction, recording, and replay behavior. Command mode starts OFF. The default action path is DRY RUN and displays action messages without controlling macOS. Real controls require an explicit `--enable-mac-controls` CLI flag or a `RoomSenseConfig` setting. Gesture recognition must remain independent from operating-system calls, and tests must never perform real media or volume actions.

## Current project context

RoomSense runs MediaPipe Pose Landmarker once per mirrored webcam frame. `PoseTracker` returns normalized shoulder, elbow, wrist, hip, knee, and ankle landmarks. `TemporalGestureDetector` recognizes timestamped wrist swipes and held poses. `InteractionModeController` starts in `NORMAL`, toggles to `COMMAND` after both pose wrists are held above the shoulders, and supports an inactivity timeout. `InteractionEngine` combines these values with typed events and a safe in-process demo action registry. The overlay already renders interaction mode, action text, and recent events.

There is no hand or finger landmark tracker. Open-palm, thumb-index pinch, and fist recognition therefore require a MediaPipe Hand Landmarker stage alongside Pose Landmarker. It will consume the same RGB frame and will not replace or alter the pose-tracking path.

## Architecture

1. `roomsense/tracking/hand_tracker.py` wraps MediaPipe Hand Landmarker and returns immutable normalized hand observations: handedness, 21 hand landmarks, and detection confidence. It uses the installed MediaPipe package and a cached model asset, following the explicit download/cache pattern used by `PoseTracker`. Model-loading errors are reported without changing pose tracker behavior.
2. `roomsense/gestures/mac_controls.py` is camera- and OS-independent. It consumes timestamped hand observations plus command-mode state and produces gesture status and action intents. It owns hand classification, multi-frame swipe history, pinch lifecycle, volume-step quantization, gesture confidence/durations, per-gesture cooldowns/debounce, and cancellation/reset behavior. Missing or low-confidence observations break continuity so stale trajectories cannot complete.
3. The existing held `BOTH_HANDS_UP` gesture remains the intentional command-mode toggle. The existing mode controller remains the source of mode state and timeout. Its activity input will be restricted to accepted command actions (including volume steps), so ordinary tracking or an uncompleted gesture cannot keep command mode alive. Turning mode OFF clears pending gesture state.
4. `roomsense/actions/action_registry.py` defines the action-intent dispatch boundary and default dry-run handler. `roomsense/actions/macos_controller.py` implements explicitly enabled native macOS actions. The recognizer never imports or invokes the OS controller. Main wires one controller into the registry based on config/CLI options.
5. `main.py` runs hand tracking on the same frame, passes observations into the interaction/command layer, and sends accepted action intents to the registry. The overlay receives resulting mode, gesture, action, confidence, cooldown, and event-log state. Existing recording/replay schemas and pose evaluation remain unchanged unless an additive backwards-compatible field is needed.

## Gesture behavior and safeguards

All thresholds are configurable in `RoomSenseConfig`, validated for finite and sensible ranges, and use conservative defaults. A gesture event requires consecutive time-stamped observations, sufficient confidence, and its configured minimum duration; no action can be caused by one frame. Detector continuity resets on lost hands, invalid timestamps, fist cancellation, or command-mode exit. Per-action cooldown and release-to-rearm debounce prevent repeats from a held pose.

- **Command mode:** Starts OFF. Holding both wrists above their shoulders for the existing configured duration toggles it ON; repeating that gesture toggles it OFF. The HUD shows `COMMAND MODE: OFF` or `COMMAND MODE: ON`. Inactivity returns to OFF after the configured timeout.
- **Swipe right / swipe left:** A tracked wrist must travel beyond a configurable horizontal distance within a bounded time window, using multiple valid samples and adequate confidence. In command mode it maps to next track / previous track.
- **Open palm:** A hand must be confidently classified as open and remain stable for the minimum hold duration. It fires play/pause once, then must leave the open-palm class before firing again.
- **Pinch and vertical movement:** Thumb-tip to index-tip distance is measured relative to palm size. A stable pinch arms volume tracking and captures a vertical reference. Upward/downward movement crossing a configurable normalized distance emits one volume-up/down step at a time. Movement is quantized into a configurable volume percentage, rate-limited, and advances its reference after each step; it does not emit per frame. Releasing the pinch ends the gesture and clears its reference.
- **Closed fist:** A stable fist held for its minimum duration cancels the current pending gesture/action, releases pinch state, and clears temporal candidates. Already completed media or volume actions cannot be undone. A canceled hand must open/release before it can arm another command.

When command mode is OFF, recognized shapes may appear as status but cannot create action intents. Dry-run is the default even while command mode is ON.

## Action boundary and macOS implementation

Action intents are limited to `NEXT_TRACK`, `PREVIOUS_TRACK`, `PLAY_PAUSE`, `VOLUME_UP`, and `VOLUME_DOWN`. The dry-run controller is the default and returns readable statuses such as `ACTION: NEXT TRACK`, `ACTION: PLAY/PAUSE`, and `ACTION: VOLUME UP`; these enter the visible action field and bounded event log without invoking OS APIs.

`--enable-mac-controls` or `RoomSenseConfig.mac_controls_enabled=True` selects the real controller. It is accepted only on macOS. The controller uses argument-array `osascript` calls (no shell string execution): System Events media key codes for previous, play/pause, and next, and AppleScript `set volume output volume` for a bounded absolute output-volume value. The README will state that media-key delivery may require granting Accessibility permission to the launching terminal/app. Volume is clamped to the macOS output range. Subprocess errors are caught at the action boundary, reported in status/log output, and do not crash tracking. The action controller is injected so tests can replace it with a recording fake.

## UI and event log

The live overlay displays command mode explicitly, including the required OFF/ON text, current gesture (for example `PINCH`), most recent action (for example `ACTION: VOLUME UP`), remaining cooldown, and confidence. A bounded action event feed shows timestamped recognized actions/results. It distinguishes DRY RUN from enabled Mac control so the display never implies an action executed when it was only simulated.

## Configuration and CLI

Add validated settings for hand visibility/confidence, pinch ratio, open-palm/fist confidence, minimum hold durations, swipe distance/window/minimum samples, minimum gesture confidence, action debounce/cooldowns, command timeout, volume movement threshold, volume step percentage, and volume update interval. `mac_controls_enabled` defaults to `False`; dry-run is selected by default. The CLI exposes `--enable-mac-controls` as the explicit real-control opt-in and `--dry-run` to force simulation when config enables real controls. With neither flag, dry-run is the default. Existing `roomsense` invocation, Python `run(config)` use, and current configuration defaults remain compatible.

## Tests and acceptance

Camera-free `unittest` tests cover hand-shape classification; multi-frame swipe direction; pinch start, movement, and end; quantized and rate-limited volume steps; minimum duration/confidence; cooldown and debounce; mode-off gating and timeout; fist cancellation; dry-run messages; injected real-controller calls; and overlay/event status. Any macOS subprocess is patched or replaced with a fake, so tests cannot change system volume or media playback. Existing RoomSense tests are run after implementation. Tracking, calibration, pointing, zone, session recording, and replay behavior must continue to pass unchanged.

## README updates and limitations

Document each gesture, command-mode activation/timeout, dry-run defaults, the explicit real-control option, native macOS permissions, configurable thresholds, and the fact that fist cancels only pending work. Note that hand-pose quality depends on lighting, camera distance, occlusion, and hand orientation; thresholds may need adjustment, and camera-free tests do not establish live-camera accuracy.
