# RoomSense Advanced Hand and Finger Tracking

## Goal

Add camera-only tracking for two hands to RoomSense, with per-finger states, stable basic hand gestures, body-wrist association, and an overlay that joins hand landmarks to the existing body skeleton. Existing full-body tracking, calibration, spatial mapping, body gestures, and their tests remain supported.

## Design

Use MediaPipe Tasks `HandLandmarker` in `VIDEO` mode on the same mirrored RGB frames and monotonically increasing timestamps already used by `PoseTracker`. Configure a maximum of two hands and use the handedness and 21 normalized image landmarks returned by the model. Keep model creation, inference, timestamp handling, and model caching in a new `roomsense/tracking/hand_tracker.py` module. MediaPipe is already a project dependency; cache a hand model beside the existing pose model and download it on first use with the same error handling conventions.

Represent each observation as immutable plain data. Preserve all 21 landmarks, handedness, a tracking-confidence estimate, palm center, image-plane palm angle, a palm normal / approximate 3D orientation derived from the palm basis, named fingertip positions, per-finger extension state (`extended`, `curled`, `uncertain`), normalized pinch distance, openness, body-wrist association, and a stable recognized hand state. MediaPipe Tasks exposes handedness classification confidence rather than a per-hand detector score, so calculate tracking confidence from that score and landmark validity and document it as an estimate. Derive values in camera-free geometry/classification helpers so they are testable without MediaPipe or a camera. Use relative landmark distances and joint geometry normalized by palm size; return `uncertain` when visibility/geometry is insufficient or values fall between extension thresholds. Smooth landmarks and require temporal confirmation before changing recognized hand state.

Recognize `OPEN PALM`, `CLOSED FIST`, `POINTING`, `PEACE SIGN`, `THUMBS UP`, and `PINCHING` from finger states, palm direction, and pinch distance. Use a small per-hand state smoother with configurable enter/exit margins and consecutive-observation confirmation so transient model noise does not flicker. Keep handedness from MediaPipe as the anatomical side label, accounting for the existing horizontally mirrored preview where association is evaluated.

Associate hands one-to-one to `left_wrist` / `right_wrist` pose landmarks where available. Score candidates by wrist-to-hand-wrist distance relative to shoulder width and handedness consistency; reject distant/ambiguous matches and retain an unassociated hand when pose data is missing. Do not let association failure suppress a valid hand observation.

Pass hand observations alongside body landmarks to `TrackingOverlay`. Draw the MediaPipe hand connections and connect an associated hand wrist directly to its body wrist, making the two outputs read as a continuous skeleton. The existing `D` debug mode will show each hand's five finger states, pinch distance, confidence, palm angle/orientation, and body association. Hand overlays remain visible in normal mode; diagnostics appear only in debug mode.

The camera-free tests will cover landmark geometry, finger states, gesture classification, uncertain inputs, hysteresis, and wrist association. Preserve and run the entire existing suite. Update README architecture, first-run model behavior, debug controls, and exact install/run commands.

## Constraints and acceptance criteria

- Use only the Mac camera already opened by RoomSense; add no external sensor or second camera path.
- Detect at most two hands and support left and right handedness.
- Keep body pose processing, calibration, room mapping, body gesture detection, interaction logic, and existing tests intact.
- The normal camera display shows hand landmarks and associated wrist connections with the body skeleton.
- Debug mode exposes all hand diagnostics listed above.
- Geometry and gesture classification tests run without camera input or MediaPipe model downloads.
- Run the complete existing test suite plus the new camera-free tests and fix regressions.
- Give README an exact run command that uses the built-in Mac camera by default.

## Performance

Run one two-hand inference per existing camera frame, reuse the RGB frame, and avoid rendering or diagnostic string work when debug mode is off. Hand smoothing/classification must remain CPU-light. Report measured suite results and any runtime limitation; camera hardware throughput is not claimed without a live-camera measurement.
