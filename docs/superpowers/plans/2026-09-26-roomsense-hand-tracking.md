# RoomSense Hand Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add stable, camera-based two-hand and five-finger tracking connected to the existing RoomSense body skeleton.

**Architecture:** Keep MediaPipe Tasks inference in `HandTracker`, geometry in pure functions and immutable hand data, and temporal classification and body-wrist association in a camera-free module. Feed the resulting hand observations into the existing camera loop and overlay while preserving body, calibration, spatial, gesture, recording, evaluation, and room interaction behavior.

**Tech Stack:** Python 3.11+, existing MediaPipe Tasks, NumPy/OpenCV already in project dependencies, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-26-roomsense-hand-tracking-design.md`

## Global Constraints

- Use the existing Mac camera frame and mirrored RGB image; do not add another sensor or camera path.
- Configure MediaPipe HandLandmarker for at most two hands in `VIDEO` mode with increasing millisecond timestamps and separate detection, presence, and tracking confidence settings.
- Keep geometry, gesture classification, hysteresis, and body association testable without camera access or model downloads.
- Preserve all existing body pose, calibration, spatial, posture/temporal gesture, interaction, session recording, evaluation, and test behavior.
- Show hand landmarks in normal mode and hand diagnostics in the existing `D` debug mode.

## Review Focus

- Invalid, non-finite, or near-zero palm geometry must produce uncertain measurements instead of division errors; test degenerate landmarks in Task 1.
- Thumb orientation varies with view angle; test uncertain thumb and pinch boundaries in Task 1.
- Handedness can disagree with proximity to a pose wrist; test one-to-one distance association and no-pose fallback in Task 2.
- Brief frame noise must not change a displayed state; test candidate confirmation and state release in Task 2.
- Missing pose or missing hand output must not stop either tracker or stale the UI; test empty result handling in Task 3 and integration in Task 4.

---

### Task 0: Preserve the supplied RoomSense snapshot on the requested branch

**Files:**
- Modify: `README.md`
- Modify: `roomsense/config.py`
- Modify: `roomsense/interactions/engine.py`
- Modify: `roomsense/main.py`
- Modify: `roomsense/visualization/overlay.py`
- Create: `roomsense/evaluation.py`
- Create: `roomsense/recording/evaluation_control.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_interaction_engine.py`
- Modify: `tests/test_visualization.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Review the snapshot changes** carried into the branch worktree and confirm they include the evaluation flow and tests from the supplied project.
- [ ] **Step 2: Commit the preserved snapshot** as `chore: preserve current RoomSense evaluation features` so hand tracking builds on the complete project state.

### Task 1: Hand geometry and immutable observation model

**Files:**
- Create: `roomsense/tracking/hand_geometry.py`
- Test: `tests/test_hand_geometry.py`

**Interfaces:**
- Produces `HandPoint(x: float, y: float, z: float, visibility: float = 1.0)`, `FingerState`, `HandState`, and frozen `HandObservation` with all 21 points, handedness, tracking-confidence estimate, palm center/orientation, fingertip positions, finger states, pinch distance, openness, recognized state, and optional body association.
- Produces `analyze_hand(points: Sequence[HandPoint], handedness: str, tracking_confidence: float) -> HandObservation` and `HAND_CONNECTIONS`.
- Image coordinates and fingertip positions are normalized to the mirrored frame; pinch distance and openness are normalized by palm size. Palm orientation includes image-plane angle and an approximate normal/orientation from the wrist/index-MCP/pinky-MCP basis.

- [ ] **Step 1: Write failing camera-free tests** for palm center/orientation, all five fingertip positions, extended and curled fingers, normalized pinch distance/openness, and degenerate or borderline geometry returning `UNCERTAIN`.
- [ ] **Step 2: Run `python -m unittest discover -s tests -p 'test_hand_geometry.py' -v`** and confirm failures are missing hand geometry API or expected calculations.
- [ ] **Step 3: Implement geometry** using MediaPipe's 21-point index order and palm-size-relative distances; reject non-finite and near-zero geometry safely.
- [ ] **Step 4: Run the geometry tests** and confirm all pass without importing MediaPipe or opening a camera.
- [ ] **Step 5: Commit** as `feat: add camera-free hand geometry`.

### Task 2: Gesture classification, temporal stability, and wrist association

**Files:**
- Create: `roomsense/tracking/hand_classifier.py`
- Test: `tests/test_hand_classifier.py`

**Interfaces:**
- Consumes `HandObservation`, `HandState`, and `FingerState` from Task 1, and body `Landmark` from `pose_tracker.py`.
- Produces `classify_hand(observation: HandObservation) -> HandState | None`, `TemporalHandClassifier.update(hand_key: str, candidate: HandState | None) -> HandState | None`, and `associate_hands(hands: Sequence[HandObservation], body_landmarks: Mapping[str, Landmark]) -> tuple[HandObservation, ...]`.
- Uses three consecutive candidate observations before a new state is adopted; ambiguous or unsupported combinations remain unclassified. Association is one-to-one, requires visible pose wrists, uses handedness plus wrist distance normalized by shoulder width, and leaves unmatched hands valid and unassociated.

- [ ] **Step 1: Write failing tests** for OPEN PALM, CLOSED FIST, POINTING, PEACE SIGN, THUMBS UP, PINCHING; ambiguous finger states; three-frame state confirmation; nearest valid wrist matching; one-to-one assignment; missing pose; and distant/ambiguous wrists.
- [ ] **Step 2: Run `python -m unittest discover -s tests -p 'test_hand_classifier.py' -v`** and confirm the intended classifier and association APIs are absent.
- [ ] **Step 3: Implement pure classification and association** with stable tie-breaking and temporal confirmation.
- [ ] **Step 4: Run the classifier tests** and confirm transitions and fallback cases pass without camera or MediaPipe.
- [ ] **Step 5: Commit** as `feat: classify and associate tracked hands`.

### Task 3: MediaPipe Tasks video tracker

**Files:**
- Create: `roomsense/tracking/hand_tracker.py`
- Modify: `roomsense/config.py`
- Test: `tests/test_hand_tracker.py`

**Interfaces:**
- Produces `HandTracker(config: RoomSenseConfig)`, `process(rgb_frame: Any, timestamp_ms: int, body_landmarks: Mapping[str, Landmark] | None = None) -> tuple[HandObservation, ...]`, `close()`, and `reset_smoothing()`.
- Uses `mp.tasks.vision.HandLandmarkerOptions` with `RunningMode.VIDEO`, `num_hands=2`, and configured detection, presence, and tracking thresholds. Converts normalized model points into `HandPoint`, computes an estimated confidence from handedness score and point validity, smooths each anatomical hand, applies temporal classification, then associates to body wrists.
- Cache `hand_landmarker.task` beside the pose model under `~/.cache/roomsense/`; download only on first use. Preserve monotonic timestamps and return an empty tuple when no hand is visible. No new dependency is required.
- Add validated settings for hand detection/presence/tracking confidence, hand landmark smoothing, maximum wrist-association distance, and gesture confirmation frames with defaults `0.5`, `0.5`, `0.5`, `0.45`, `0.75` shoulder widths, and `3` frames.

- [ ] **Step 1: Write failing tests** for default/invalid hand settings, empty model result, left/right result conversion with 21 landmarks, two-hand limit, monotonic timestamp handling, and close/reset behavior using fake MediaPipe result objects.
- [ ] **Step 2: Run `python -m unittest discover -s tests -p 'test_hand_tracker.py' -v`** and confirm the tracker/settings APIs are absent.
- [ ] **Step 3: Implement the MediaPipe adapter** with lazy import, cached model download, two-hand VIDEO options, smoothed per-hand points, and the pure functions from Tasks 1–2.
- [ ] **Step 4: Run tracker tests** and confirm they need neither a live camera nor downloaded model data.
- [ ] **Step 5: Commit** as `feat: add MediaPipe two-hand video tracking`.

### Task 4: Body-loop and overlay integration

**Files:**
- Modify: `roomsense/main.py`
- Modify: `roomsense/visualization/overlay.py`
- Test: `tests/test_visualization.py`

**Interfaces:**
- `main.run` creates and closes `HandTracker` beside `PoseTracker`, passes the current shared RGB frame, timestamp, and pose landmarks, and supplies observations to `TrackingOverlay.draw`.
- `TrackingOverlay.draw(..., hands: Sequence[HandObservation] = ())` draws the 21-point hand connections in side-specific colors, connects an associated hand wrist to its corresponding body wrist, and shows hand state labels. With debug on, include finger states, pinch distance, tracking-confidence estimate, palm angle/normal, and body association.

- [ ] **Step 1: Write failing in-memory overlay tests** for hand connections, association wrist bridge, state label, and debug text input while preserving the existing body skeleton path.
- [ ] **Step 2: Run the targeted visualization tests** and confirm hand arguments/drawing are not present.
- [ ] **Step 3: Integrate the tracker into the existing loop and overlay** without changing pose, calibration, spatial, gesture, recording, evaluation, or interaction call semantics.
- [ ] **Step 4: Run visualization and core tracking tests** and confirm existing behavior remains intact.
- [ ] **Step 5: Commit** as `feat: connect hand landmarks to RoomSense skeleton`.

### Task 5: README and complete regression verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README** with hand model first-run download/cache behavior, supported hand states, finger-state and confidence caveats, body-wrist association, `D` diagnostics, architecture entries, and exact installation and built-in-camera run commands.
- [ ] **Step 2: Run the complete existing and new suite** with `python -m unittest discover -s tests -v` and fix regressions.
- [ ] **Step 3: Review the branch diff** to confirm full-body, calibration, spatial tracking, existing gestures, evaluation, recording, and all pre-existing tests remain present.
- [ ] **Step 4: Commit** as `docs: document RoomSense hand tracking`.
