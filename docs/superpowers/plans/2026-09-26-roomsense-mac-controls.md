# RoomSense Gesture-Based Mac Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Add safe, temporally recognized hand gestures that can control media and volume on macOS only after intentional command-mode activation.

**Architecture:** MediaPipe Hand Landmarker runs beside the existing Pose Landmarker and returns normalized observations. A camera/OS-independent gesture controller converts timestamped observations into action intents; an injected registry defaults to dry-run and uses native macOS actions only after explicit opt-in. The existing interaction mode owns command-mode state, and the HUD shows gesture/action status and a bounded action log.

**Tech Stack:** Python 3.11+, existing MediaPipe dependency, standard-library dataclasses/enums/deques/subprocess, `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-26-roomsense-mac-controls-design.md`

## Global Constraints

- Command mode starts OFF; dry-run is the default.
- Real controls require `--enable-mac-controls` or `RoomSenseConfig.mac_controls_enabled=True`; `--dry-run` forces simulation.
- No action may be caused by one frame; recognition uses timestamps, confidence, minimum duration, cooldown, and release-to-rearm debounce.
- Gesture recognition does not invoke OS APIs; action execution uses injected handlers.
- macOS controls use argument-array `osascript` calls and are unavailable on other platforms.
- Tests mock the macOS action boundary and never change volume or media playback.
- Existing pose, floor, zone, pointing, recording, replay, and camera behavior remains intact.

## Review Focus

- Missing/low-confidence hands during a swipe or pinch clear continuity; test in Task 3.
- Non-finite/decreasing timestamps and invalid settings are rejected; test in Tasks 1 and 3.
- Two visible hands do not duplicate an action or switch owner mid-gesture; keep an active hand as owner until release/loss, then choose the higher-confidence hand (left wins ties); test in Task 3.
- At the exact command timeout boundary, timeout wins before same-frame gesture dispatch and clears pending state; test in Task 4.
- Controller failure and non-macOS opt-in do not crash tracking or execute OS actions in tests; test in Tasks 2 and 4.

---

## File map

- Create `roomsense/tracking/hand_tracker.py` for hand values and MediaPipe inference/model lifecycle.
- Create `roomsense/actions/action_registry.py` for typed intents/results, dry-run dispatch, and bounded action history.
- Create `roomsense/actions/macos_controller.py` for native media-key and output-volume calls.
- Create `roomsense/gestures/mac_controls.py` for hand classification, temporal state, cancellation/debounce/cooldowns, and volume quantization.
- Modify `roomsense/config.py`, `roomsense/interactions/engine.py`, `roomsense/interactions/modes.py`, `roomsense/main.py`, `roomsense/__main__.py`, and `roomsense/visualization/overlay.py` for validated settings and wiring.
- Add focused tests for hand tracking, action safety, gestures, and CLI; extend engine/config/mode/overlay tests. Update `README.md`.

## Tasks

### Task 1: Hand landmark tracker and configuration

**Files:** create `roomsense/tracking/hand_tracker.py`, `tests/test_hand_tracker.py`; modify `roomsense/config.py`, `tests/test_config.py`.

**Interfaces:** `HandPoint(x: float, y: float, z: float)`, `TrackedHand(handedness: str, landmarks: tuple[HandPoint, ...], confidence: float)`, and `HandTracker(config).process(rgb_frame, timestamp_ms=None) -> tuple[TrackedHand, ...]`; `close()` releases the task. Each hand has exactly 21 finite normalized landmarks; use the MediaPipe handedness category score as observation confidence. Add `hand_model_path: str | None = None`, `hand_detection_confidence: float = 0.55`, `hand_tracking_confidence: float = 0.55`, and `mac_controls_enabled: bool = False`.

- [x] Write `test_tracker_maps_21_normalized_landmarks_and_handedness`, `test_tracker_returns_empty_tuple_when_no_hand_is_found`, and `test_tracker_close_releases_task` using a fake MediaPipe task.
- [x] Run `python -m unittest discover -s tests -p 'test_hand_tracker.py' -v`; confirm the new API is absent.
- [x] Implement lazy MediaPipe imports, cached `hand_landmarker.task`, monotonic VIDEO timestamps, and value validation; do not modify `PoseTracker`.
- [x] Run focused tracker and `test_config.py` suites; both must pass without downloading a model or opening a camera.
- [x] Commit as `feat: add MediaPipe hand landmark tracking`.

### Task 2: Action registry and native macOS controller

**Files:** create `roomsense/actions/__init__.py`, `roomsense/actions/action_registry.py`, `roomsense/actions/macos_controller.py`, `tests/test_mac_actions.py`.

**Interfaces:** `MacAction(str, Enum)` values `NEXT_TRACK`, `PREVIOUS_TRACK`, `PLAY_PAUSE`, `VOLUME_UP`, `VOLUME_DOWN`; immutable `ActionIntent(action: MacAction, gesture: str, timestamp: float, confidence: float)` and `ActionResult(action: MacAction, timestamp: float, message: str, dry_run: bool, succeeded: bool)`; `ActionRegistry(mac_controls_enabled: bool = False, controller: ActionController | None = None, history_size: int = 8).dispatch(intent: ActionIntent) -> ActionResult`, with a chronological `history` tuple bounded to `history_size`; `ActionController.execute(intent: ActionIntent) -> None`; and `MacOSController(volume_step_percent: int = 5).execute(intent: ActionIntent) -> None`. When controls are enabled with no injected controller, construct `MacOSController` only on Darwin and reject unsupported platforms.

- [x] Write tests `test_default_registry_returns_dry_run_message_without_calling_controller`, `test_enabled_registry_dispatches_to_injected_controller`, `test_registry_keeps_bounded_action_history`, `test_macos_controller_maps_media_actions_to_osascript_keys`, `test_macos_controller_quantizes_and_clamps_volume`, `test_macos_controller_reports_subprocess_failure`, and `test_non_macos_real_control_activation_is_rejected`; patch `subprocess.run` in every real-controller test. Use key codes 98/100/101 for previous/play-pause/next; volume reads the current output level, applies ±5, clamps to 0–100, then sets it.
- [x] Run `python -m unittest discover -s tests -p 'test_mac_actions.py' -v`; confirm expected missing APIs.
- [x] Implement dry-run messages (`ACTION: NEXT TRACK`, `ACTION: PREVIOUS TRACK`, `ACTION: PLAY/PAUSE`, `ACTION: VOLUME UP`, `ACTION: VOLUME DOWN`) and real argument-array `osascript` execution. Reject real dispatch off macOS, clamp output volume to 0–100, and return failures as results.
- [x] Run the focused suite; the dry-run test must observe zero controller calls and every OS test must use its patched subprocess.
- [x] Commit as `feat: add safe macOS action dispatch`.

### Task 3: Temporal hand gesture recognition

**Files:** create `roomsense/gestures/mac_controls.py`, `tests/test_mac_controls.py`; modify `roomsense/config.py`, `tests/test_config.py`.

**Interfaces:** `HandShape` values `OPEN_PALM`, `PINCH`, `FIST`, `OTHER`; `HandClassification(shape: HandShape, confidence: float)`; immutable `GestureStatus(gesture: str | None, confidence: float, cooldown_seconds: float, cancelled: bool, actions: tuple[ActionIntent, ...])`; `classify_hand(hand: TrackedHand, config: RoomSenseConfig) -> HandClassification`; `HandGestureController(config: RoomSenseConfig).update(hands: Sequence[TrackedHand], timestamp: float, command_mode: bool) -> GestureStatus`; `reset() -> None` clears state. Actions are Task 2 `ActionIntent` values and are returned only when `command_mode=True`.

- [x] Add `test_config_rejects_invalid_mac_gesture_thresholds` in `tests/test_config.py`; write tests `test_shape_classifier_distinguishes_open_palm_pinch_and_fist`, `test_open_palm_requires_configured_hold`, `test_swipe_requires_multiple_samples_and_reports_direction`, `test_pinch_start_movement_and_release`, `test_volume_quantizes_each_threshold_crossing_and_rate_limits`, `test_fist_cancels_pinch_and_requires_release_to_rearm`, `test_low_confidence_or_missing_hand_breaks_continuity`, `test_mode_off_never_emits_action`, `test_two_hands_keep_one_stable_gesture_owner`, `test_cooldown_and_debounce_suppress_repeated_actions`, and `test_non_finite_or_decreasing_timestamps_are_rejected`.
- [x] Run `python -m unittest discover -s tests -p 'test_mac_controls.py' -v`; confirm the new API is missing.
- [x] Implement palm-relative thumb/index distance, finger extension/curl classification, and wrist trajectories. Use `gesture_min_confidence=0.75`, `gesture_open_palm_hold_seconds=0.5`, `gesture_pinch_hold_seconds=0.15`, `gesture_pinch_ratio=0.30`, `gesture_fist_hold_seconds=0.35`, `gesture_min_samples=3`, `gesture_swipe_min_duration_seconds=0.08`, `gesture_volume_movement_threshold=0.08`, `gesture_volume_step_percent=5`, and `gesture_volume_update_interval_seconds=0.25`. Require swipe sample duration at least `gesture_swipe_min_duration_seconds`. Classify fist before pinch, then open palm. Keep an active hand as owner until release/loss; when idle choose the higher-confidence hand, with left winning ties. Require consecutive valid samples and confidence at least `gesture_min_confidence`. Emit one held-pose action until release. Quantize a volume step only after crossing its movement threshold and rate-limit interval; advance the movement anchor only after an emitted step. Fist clears pending gesture state but cannot undo a dispatched action.
- [x] Run `test_mac_controls.py` and `test_config.py`; prove actions require multiple valid observations and mode OFF emits none.
- [x] Commit as `feat: recognize temporal hand commands`.

### Task 4: Command-mode and camera-loop integration

**Files:** modify `roomsense/interactions/engine.py`, `roomsense/interactions/modes.py`, `roomsense/main.py`, `roomsense/__main__.py`; create/extend `tests/test_interaction_engine.py`, `tests/test_interaction_modes.py`, `tests/test_main_controls.py`.

**Interfaces:** add optional `hands: tuple[TrackedHand, ...] = ()` to `InteractionObservation`; add `command_gesture: GestureStatus` and `action_results: tuple[ActionResult, ...]` to `InteractionUpdate`; allow `InteractionEngine(..., mac_action_registry: ActionRegistry | None = None)`. Preserve existing fields and `run(config)`.

- [x] Add tests `test_both_hands_up_is_required_to_enter_command_mode`, `test_mode_off_cannot_dispatch_completed_hand_gesture`, `test_only_accepted_command_actions_refresh_timeout`, `test_timeout_clears_pending_gesture_and_exits_once`, `test_fist_cancel_is_visible_without_dispatching_action`, `test_run_with_injected_hand_tracker_preserves_pose_pipeline`, `test_hand_tracker_unavailable_leaves_pose_tracking_running`, and CLI tests for mutually exclusive `--enable-mac-controls`/`--dry-run` flags, CLI-over-config precedence, and default dry-run behavior.
- [x] Run focused engine, mode, and CLI tests; confirm new integration assertions fail before implementation.
- [x] Feed hand observations to the engine and action registry, gate on the existing both-hands-up transition, refresh timeout only for accepted commands, clear pending state on OFF/timeout, and keep dry-run default. Validate unsupported real-control platforms before opening the camera. Run Hand Landmarker on the same RGB frame and close it on every shutdown path. If hand-model setup fails, report that hand commands are unavailable and continue existing pose/spatial tracking. Evaluate mode timeout before same-frame hand dispatch, then refresh activity only after an accepted action. Do not infer real-control opt-in from the OS.
- [x] Run focused tests plus `test_interaction_engine.py`, `test_interaction_modes.py`, `test_temporal_gestures.py`, and `test_tracking_core.py`; existing pose and spatial behavior must remain intact.
- [x] Commit as `feat: gate hand controls behind command mode`.

### Task 5: Gesture HUD and action event feed

**Files:** modify `roomsense/visualization/overlay.py`, `tests/test_visualization.py`.

**Interfaces:** extend `TrackingOverlay.draw` with optional `command_gesture`, `action_results`, and `mac_controls_enabled`; show command mode, gesture, action, cooldown, confidence, execution mode, and recent action results.

- [x] Write `test_hud_shows_command_mode_off_and_dry_run`, `test_hud_shows_gesture_action_confidence_and_cooldown`, and `test_action_log_is_visible_and_bounded` using mocked `cv2.putText`.
- [x] Run `python -m unittest discover -s tests -p 'test_visualization.py' -v`; confirm new labels are absent.
- [x] Render exact `COMMAND MODE: OFF`/`COMMAND MODE: ON`, gesture/action/confidence/cooldown values, dry-run/real status, and a bounded timestamped action feed without obscuring existing HUD fields.
- [x] Run focused visualization tests and all existing `test_visualization.py` cases.
- [x] Commit as `feat: display hand command status`.

### Task 6: CLI documentation and full verification

**Files:** modify `README.md`; update `tests/test_main_controls.py` only if needed to match documented CLI behavior.

- [x] Document gesture mappings, both-hands-up activation and timeout, dry-run default, `roomsense --enable-mac-controls`, config opt-in, `--dry-run` override, Accessibility permission, fist cancellation limits, model cache/download, and hand-tracking limitations.
- [x] Run `python -m unittest discover -s tests -v`; every existing and new test must pass with only injected/patched action handlers.
- [x] Run `python -m compileall -q roomsense tests` and `git diff --check`; both must succeed.
- [x] Review the final diff for tracking-path changes and accidental OS execution in tests; commit as `docs: explain RoomSense gesture controls`.
