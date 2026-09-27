"""RoomSense webcam application with calibrated tracking and V3 interactions."""

from __future__ import annotations

import sys
from pathlib import Path
from time import monotonic, monotonic_ns, sleep

from roomsense.calibration.calibration_store import load_calibration, save_calibration
from roomsense.config import RoomSenseConfig
from roomsense.gestures.gesture_detector import GestureDetector
from roomsense.interactions.engine import InteractionEngine, InteractionObservation
from roomsense.recording.recording_control import RecordingController
from roomsense.recording.session_recorder import SessionRecorder
from roomsense.spatial.floor_position import FloorPosition, FloorPositionTracker
from roomsense.spatial.observation_state import SpatialObservationMonitor
from roomsense.spatial.position_history import MovementMetrics, PositionHistory
from roomsense.spatial.room_transform import RoomTransform
from roomsense.spatial.zones import Zone, ZoneTracker
from roomsense.tracking.person_state import MovementTracker
from roomsense.tracking.position_tracker import PositionTracker
from roomsense.tracking.pose_tracker import PoseTracker
from roomsense.tracking.reidentification import (
    PersonReidentifier,
    ReidentificationState,
    TrackObservation,
)
from roomsense.visualization.calibration_view import CalibrationView
from roomsense.visualization.overlay import TrackingOverlay


def run(config: RoomSenseConfig | None = None) -> int:
    settings = config or RoomSenseConfig()
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV or NumPy is missing. Activate the project environment and install requirements.txt.",
              file=sys.stderr)
        return 1

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    camera_index = settings.camera_index if settings.camera_index is not None else 0
    camera_name: str | None = None
    if sys.platform == "darwin" and settings.camera_index is None:
        try:
            from AVFoundation import (  # type: ignore[import-not-found]
                AVCaptureDevice,
                AVCaptureDeviceTypeBuiltInWideAngleCamera,
                AVMediaTypeMuxed,
                AVMediaTypeVideo,
            )
        except ImportError:
            print(
                "Built-in camera selection needs pyobjc-framework-AVFoundation. "
                "Install requirements.txt again in the active virtual environment.",
                file=sys.stderr,
            )
            return 1

        # OpenCV's AVFoundation backend combines these lists and sorts them by
        # uniqueID before interpreting a camera index. Mirror that ordering so
        # the selected device index refers to the same camera in both APIs.
        devices = list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo))
        devices.extend(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed))
        devices.sort(key=lambda device: str(device.uniqueID()))
        for index, device in enumerate(devices):
            if device.deviceType() == AVCaptureDeviceTypeBuiltInWideAngleCamera:
                camera_index = index
                camera_name = str(device.localizedName())
                break
        else:
            print(
                "Could not find a built-in Mac camera. Connect or enable the Mac camera, "
                "or set camera_index explicitly in RoomSenseConfig.",
                file=sys.stderr,
            )
            return 1

        print(f"Using built-in Mac camera: {camera_name}")

    camera = cv2.VideoCapture(camera_index, backend)
    if not camera.isOpened():
        camera.release()
        print(
            f"Could not open camera {camera_name or camera_index}. Check macOS Camera permissions "
            "or camera_index in RoomSenseConfig.",
            file=sys.stderr,
        )
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
    camera.set(cv2.CAP_PROP_FPS, settings.max_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    tracker: PoseTracker | None = None
    recording_controller: RecordingController | None = None
    try:
        tracker = PoseTracker(settings)
        position_tracker = PositionTracker(settings)
        person_reidentifier = (
            PersonReidentifier(
                settings.reidentification_timeout_seconds,
                settings.reidentification_confidence_threshold,
                settings.reidentification_ambiguity_margin,
            )
            if settings.reidentification_enabled else None
        )
        movement_tracker = MovementTracker(
            settings.movement_window_seconds,
            settings.movement_horizontal_threshold,
            settings.movement_depth_threshold,
        )
        gesture_detector = GestureDetector(
            settings.raised_hand_margin,
            settings.standing_leg_ratio,
            settings.sitting_leg_ratio,
        )
        overlay = TrackingOverlay()
        interaction_engine = InteractionEngine(settings, settings.room_objects)
        recording_controller = RecordingController(
            lambda: SessionRecorder(settings.session_recording_path, settings.record_position_interval_seconds)
        )
        calibration_view = CalibrationView()
        calibration_path = Path(settings.calibration_path).expanduser()
        configured_zones = tuple(Zone(name, tuple(polygon)) for name, polygon in settings.zone_polygons)
        zone_tracker = ZoneTracker(configured_zones)
        floor_tracker = FloorPositionTracker(settings.floor_landmark_visibility, settings.smoothing_alpha)
        history = PositionHistory(settings.history_retention_seconds)
        spatial_observations = SpatialObservationMonitor(settings.tracking_lost_seconds)
        movement_metrics: MovementMetrics = history.clear_tracking()
        calibration = None
        room_transform: RoomTransform | None = None
        calibration_loaded = False
        current_floor_position: FloorPosition | None = None
        zone_transition: str | None = None
        debug = False
        identity_transition: str | None = None
        identity_transition_until = 0.0

        window_name = "RoomSense — Room Tracking"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        frame_times: list[float] = []
        lost_since: float | None = None
        read_failures = 0

        while True:
            ok, camera_frame = camera.read()
            now = monotonic()
            if not ok or camera_frame is None:
                read_failures += 1
                if read_failures >= 30:
                    print("Camera stopped returning frames. RoomSense is closing.", file=sys.stderr)
                    break
                sleep(0.02)
                continue
            read_failures = 0
            zone_update = None
            camera_frame = cv2.flip(camera_frame, 1)  # Match the mirrored preview used for calibration clicks.
            if not calibration_loaded:
                frame_height, frame_width = camera_frame.shape[:2]
                try:
                    calibration = load_calibration(calibration_path, (frame_width, frame_height))
                except ValueError as exc:
                    print(f"Saved calibration is invalid ({exc}); press C to recalibrate.", file=sys.stderr)
                    calibration = None
                if calibration is not None:
                    room_transform = RoomTransform(calibration)
                    print(f"Loaded floor calibration from {calibration_path}.")
                calibration_loaded = True

            frame = camera_frame.copy()
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = tracker.process(rgb, monotonic_ns() // 1_000_000)

            if landmarks:
                lost_since = None
                position = position_tracker.update(landmarks)
                movement = movement_tracker.update(position.x, position.y, position.z, now)
                states = gesture_detector.detect(landmarks)
                movement_text = movement.value
                pose_text = [state.value for state in states]
                lost = False
            else:
                if lost_since is None:
                    lost_since = now
                lost = now - lost_since >= settings.tracking_lost_seconds
                if lost:
                    position_tracker.reset()
                    movement_tracker.reset()
                    tracker.reset_smoothing()
                position = None
                movement_text = movement_tracker.current.value
                pose_text = []

            identity_update = None
            person_state_for_hud = None
            if person_reidentifier is not None:
                if landmarks and position is not None:
                    left_shoulder = landmarks.get("left_shoulder")
                    right_shoulder = landmarks.get("right_shoulder")
                    left_hip = landmarks.get("left_hip")
                    right_hip = landmarks.get("right_hip")
                    shoulder_width = None
                    torso_ratio = None
                    if left_shoulder is not None and right_shoulder is not None:
                        shoulder_width = ((right_shoulder.x - left_shoulder.x) ** 2
                                          + (right_shoulder.y - left_shoulder.y) ** 2) ** 0.5
                        if shoulder_width > 1e-4 and left_hip is not None and right_hip is not None:
                            shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2.0,
                                               (left_shoulder.y + right_shoulder.y) / 2.0)
                            hip_center = ((left_hip.x + right_hip.x) / 2.0,
                                          (left_hip.y + right_hip.y) / 2.0)
                            torso_length = ((hip_center[0] - shoulder_center[0]) ** 2
                                            + (hip_center[1] - shoulder_center[1]) ** 2) ** 0.5
                            torso_ratio = torso_length / shoulder_width
                        else:
                            shoulder_width = None
                    observation = TrackObservation(
                        timestamp=now,
                        x=(position.x + 1.0) / 2.0,
                        y=(1.0 - position.y) / 2.0,
                        z=position.z,
                        shoulder_width=shoulder_width,
                        torso_ratio=torso_ratio,
                    )
                    identity_update = person_reidentifier.update(observation)
                else:
                    identity_update = person_reidentifier.update(None, timestamp=now)

                if identity_update.state in (ReidentificationState.NEW, ReidentificationState.REACQUIRED):
                    identity_transition = identity_update.state.value
                    identity_transition_until = now + 1.0
                    person_state_for_hud = identity_transition
                elif identity_update.state is ReidentificationState.LOST:
                    identity_transition = None
                    person_state_for_hud = ReidentificationState.LOST.value
                elif now < identity_transition_until and identity_transition is not None:
                    person_state_for_hud = identity_transition
                else:
                    person_state_for_hud = ReidentificationState.TRACKED.value

            current_floor_position = (
                floor_tracker.update(landmarks, room_transform)
                if landmarks and room_transform is not None else None
            )
            spatial_position_valid = bool(
                current_floor_position and current_floor_position.projectable and current_floor_position.in_bounds
            )
            if room_transform is not None:
                if spatial_position_valid:
                    spatial_observations.mark_valid()
                    room_point = (current_floor_position.room_x, current_floor_position.room_y)
                    zone_update = zone_tracker.update(room_point)
                    if zone_update.entered:
                        zone_transition = f"ENTERED {zone_update.entered[0]}"
                    elif zone_update.left:
                        zone_transition = f"LEFT {zone_update.left[0]}"
                    movement_metrics = history.update(now, room_point)
                else:
                    # A mapped point beyond the calibrated floor is a definite zone
                    # exit. Missing or unprojectable feet expire after a brief grace period.
                    if current_floor_position is not None and (
                        not current_floor_position.projectable or not current_floor_position.in_bounds
                    ):
                        zone_update = zone_tracker.update(None)
                        if zone_update.left:
                            zone_transition = f"LEFT {zone_update.left[0]}"
                    if spatial_observations.mark_missing(now):
                        floor_tracker.reset()
                        movement_metrics = history.clear_tracking()
                        zone_update = zone_tracker.update(None)
                        if zone_update.left:
                            zone_transition = f"LEFT {zone_update.left[0]}"

            entered_zones = zone_update.entered if zone_update is not None else ()
            left_zones = zone_update.left if zone_update is not None else ()
            room_position = (
                (current_floor_position.room_x, current_floor_position.room_y)
                if spatial_position_valid and current_floor_position is not None else None
            )
            interaction = interaction_engine.update(InteractionObservation(
                timestamp=now,
                landmarks=landmarks,
                room_position=room_position,
                zone_entered=entered_zones,
                zone_left=left_zones,
                movement_state=movement_text,
            ))
            recording_state = {
                "timestamp": now,
                "room_position": room_position,
                "movement": movement_text,
                "mode": interaction.mode.value,
                "pointing": [
                    {"side": arm.side, "confidence": arm.confidence, "direction": arm.direction}
                    for arm in interaction.pointing
                ],
                "candidate_target": interaction.target.candidate.object.id
                if interaction.target.candidate else None,
                "confirmed_target": interaction.target.confirmed.object.id
                if interaction.target.confirmed else None,
                "gestures": [gesture.name for gesture in interaction.gestures],
            }
            recording_controller.record(interaction.events, recording_state)

            frame_times.append(now)
            frame_times = [stamp for stamp in frame_times if now - stamp <= 1.0]
            fps = len(frame_times) / max(0.05, now - frame_times[0]) if len(frame_times) > 1 else 0.0
            display = overlay.draw(
                frame, landmarks, position, movement_text, pose_text, fps, lost,
                calibration=calibration, floor_position=current_floor_position,
                zones=configured_zones, current_zones=zone_tracker.current,
                history=history.samples, direction=movement_metrics.direction,
                speed=movement_metrics.speed if spatial_position_valid else 0.0,
                distance=movement_metrics.distance,
                debug=debug, zone_transition=zone_transition,
                pointing=interaction.pointing, target=interaction.target, mode=interaction.mode,
                recent_events=interaction.recent_events, demo_status=interaction.demo_status,
                recorder_active=recording_controller.active, objects=interaction_engine.room_objects.objects,
                debug_state=interaction.debug_state,
                person_id=identity_update.person_id if identity_update is not None else None,
                person_state=person_state_for_hud,
                reidentification_debug=(
                    {
                        "reason": identity_update.reason,
                        "confidence": identity_update.confidence,
                        "threshold": settings.reidentification_confidence_threshold,
                        "ambiguity_margin": settings.reidentification_ambiguity_margin,
                        "candidates": identity_update.candidates,
                    }
                    if identity_update is not None else None
                ),
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key == 27:
                break
            if key in (ord("d"), ord("D")):
                debug = not debug
            elif key in (ord("r"), ord("R")):
                movement_metrics = history.reset_session()
                floor_tracker.reset()
                zone_tracker.reset()
                current_floor_position = None
                zone_transition = "SPATIAL SESSION RESET"
            elif key in (ord("v"), ord("V")):
                try:
                    active = recording_controller.toggle()
                    zone_transition = "SESSION RECORDING STARTED" if active else "SESSION RECORDING SAVED"
                except OSError as exc:
                    zone_transition = "SESSION RECORDING FAILED"
                    print(f"Could not start session recording: {exc}", file=sys.stderr)
            elif key in (ord("c"), ord("C")):
                new_calibration, quit_requested = calibration_view.run(camera_frame, window_name)
                if quit_requested:
                    break
                if new_calibration is not None:
                    calibration = new_calibration
                    room_transform = RoomTransform(calibration)
                    floor_tracker.reset()
                    movement_metrics = history.reset_session()
                    zone_tracker.reset()
                    current_floor_position = None
                    zone_transition = "CALIBRATION SAVED"
                    try:
                        save_calibration(calibration_path, calibration)
                        print(f"Saved floor calibration to {calibration_path}.")
                    except OSError as exc:
                        zone_transition = "CALIBRATION ACTIVE; SAVE FAILED"
                        print(f"Could not save calibration to {calibration_path}: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"RoomSense could not start tracking: {exc}", file=sys.stderr)
        return 1
    finally:
        if recording_controller is not None:
            try:
                recording_controller.close()
            except OSError as exc:
                print(f"Could not fully flush session recording: {exc}", file=sys.stderr)
        try:
            if tracker is not None:
                tracker.close()
        finally:
            try:
                camera.release()
            finally:
                cv2.destroyAllWindows()
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
