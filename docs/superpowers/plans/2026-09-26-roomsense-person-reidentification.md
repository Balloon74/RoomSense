# RoomSense Person Re-identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Reconnect a returning single-camera track to a likely recent anonymous session ID, while declining low-confidence or ambiguous matches.

**Architecture:** Keep matching in a camera-independent `roomsense/tracking/reidentification.py` module. `main.py` will provide normalized position and coarse torso geometry, while `TrackingOverlay` displays the lifecycle state and optional candidate diagnostics. All registry state remains process-local.

**Tech Stack:** Python 3.11+, dataclasses/enums, existing MediaPipe landmark and OpenCV HUD flow, `unittest`.

**Spec:** [2026-09-26-roomsense-person-reidentification-design.md](../specs/2026-09-26-roomsense-person-reidentification-design.md)

## Global Constraints

- Use anonymous IDs `PERSON_001`, `PERSON_002`, and so on; never model a named real-world identity.
- Do not collect face, clothing, frame crops, or appearance embeddings.
- Keep re-identification state and geometry in memory only; do not add it to recordings or save it across launches.
- Keep matching independent of pose inference and camera hardware.
- A match requires both the configured confidence threshold and a clear margin over the next candidate; otherwise allocate a new ID.
- Preserve the existing one-pose MediaPipe runtime.

## Review Focus

- An immediate return near the predicted path should recover the same ID, while a return with substantially different position should get a new ID. (Task 1 tests both.)
- A return after the retention timeout must not see an expired candidate. (Task 1 timeout test.)
- Two similarly plausible lost candidates must not be resolved arbitrarily. (Task 1 ambiguity test.)
- When multiple lost candidates exist, a distinct best candidate should win and debug output should retain each factor. (Task 1 multiple-candidate test.)
- A missing camera observation should show `LOST`, while recovered/new IDs and debug details remain visible without changing pose extraction. (Task 2 overlay tests.)
- If torso geometry is unavailable, that missing evidence must not inflate the match score. (Task 1 missing-geometry test.)

---

### Task 1: Camera-free re-identification engine

**Files:**
- Create: `roomsense/tracking/reidentification.py`
- Create: `tests/test_reidentification.py`

**Interfaces:**
- Produces `TrackObservation(timestamp: float, x: float, y: float, z: float, shoulder_width: float | None = None, torso_ratio: float | None = None)`.
- Produces `ReidentificationState` values `NEW`, `TRACKED`, `LOST`, and `REACQUIRED`.
- Produces `CandidateScore(person_id: str, factors: Mapping[str, float], score: float, reason: str)` and `ReidentificationUpdate(person_id: str | None, state: ReidentificationState, confidence: float | None, candidates: tuple[CandidateScore, ...], reason: str)`.
- Produces `PersonReidentifier(timeout_seconds: float = 5.0, confidence_threshold: float = 0.72, ambiguity_margin: float = 0.12)` with `update(observation: TrackObservation | None, timestamp: float | None = None) -> ReidentificationUpdate`. A present observation supplies its timestamp; a missing observation requires the timestamp argument so lost records can expire.
- The manager estimates per-axis velocity from the last two active observations. Predict the lost position with `last_position + velocity * min(elapsed, 1.0)`, clamp normalized X/Y to `[0, 1]`, and compute trajectory score as `exp(-distance / 0.35)`, with Z difference scaled by `0.5` inside the Euclidean distance.
- Compute expected re-entry side by projecting that velocity up to `0.75` seconds and checking whether the projection reaches an image boundary. The direction factor is `max(0, 1 - distance_to_expected_boundary / 0.20)`; when no boundary is predicted, use neutral `0.5`. Time factor is `exp(-elapsed / timeout_seconds)`. Geometry factor is the mean `exp(-abs(log(returned / stored)) / 0.35)` over shoulder width and torso ratio values present on both records; if none are comparable, use `0.0` without changing weights.
- Final score is `0.45 * trajectory + 0.25 * direction + 0.15 * time + 0.15 * geometry`. Reject below threshold; reject as ambiguous if the best score is less than `ambiguity_margin` above the second-best; otherwise select the best candidate. `reason` values distinguish selected, low confidence, ambiguous, expired/no candidate, and new identity.

- [x] **Step 1: Write the failing lifecycle tests** for initial `PERSON_001` allocation, continued `TRACKED`, immediate return as `REACQUIRED`, and movement along the expected return side.
- [x] **Step 2: Run those tests and confirm expected failures** with `python -m unittest tests.test_reidentification -v`.
- [x] **Step 3: Implement in-memory active/lost track lifecycle** and the four normalized factors: predicted trajectory/position (weight 0.45), expected re-entry direction (0.25), elapsed-time decay (0.15), and torso geometry (0.15). Use a 5-second default retention, 0.72 minimum score, and 0.12 minimum lead over the next candidate. Store only bounded recent observations needed for velocity and a compact lost-track record.
- [x] **Step 4: Run the lifecycle tests and confirm they pass.**
- [x] **Step 5: Add failing tests for timeout expiry, clearly different tracks, ambiguous candidates, selecting the best of multiple lost tracks, and unavailable geometry.** Assert resulting IDs/states and diagnostic factor/reason values; confirm missing geometry contributes zero rather than increasing the score.
- [x] **Step 6: Run the new tests and confirm the expected rejection/match failures.**
- [x] **Step 7: Implement expiry, confidence gating, ambiguity rejection, and per-candidate diagnostics.** Missing geometry contributes a zero geometry factor without renormalizing away its weight, so absent evidence cannot inflate confidence.
- [x] **Step 8: Run `python -m unittest tests.test_reidentification -v` and confirm all engine tests pass.**
- [x] **Step 9: Commit the tested engine** with `feat: add anonymous track re-identification engine`.

### Task 2: Configuration, runtime connection, and HUD

**Files:**
- Modify: `roomsense/config.py`
- Modify: `roomsense/main.py`
- Modify: `roomsense/visualization/overlay.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_visualization.py`

**Interfaces:**
- Consumes `PersonReidentifier`, `TrackObservation`, and `ReidentificationUpdate` from Task 1.
- Adds `reidentification_enabled: bool = True`, `reidentification_timeout_seconds: float = 5.0`, `reidentification_confidence_threshold: float = 0.72`, and `reidentification_ambiguity_margin: float = 0.12` to `RoomSenseConfig`.
- Adds keyword-only `person_id`, `person_state`, and `reidentification_debug` inputs to `TrackingOverlay.draw`.

- [x] **Step 1: Write failing configuration tests** for defaults, invalid timeout, out-of-range confidence/margin, and non-boolean enable values.
- [x] **Step 2: Run `python -m unittest tests.test_config -v` and confirm those assertions fail.**
- [x] **Step 3: Add and validate the four config fields**, then run the config tests.
- [x] **Step 4: Write failing overlay tests** that capture `cv2.putText` and assert the HUD renders anonymous ID plus `NEW`, `LOST`, and `REACQUIRED`, and that debug mode shows candidate IDs, factor scores, total score, configured threshold/margin, and match/reject reason. `reidentification_debug` is a mapping with `reason`, `threshold`, `ambiguity_margin`, and a `candidates` sequence of `CandidateScore` values.
- [x] **Step 5: Run `python -m unittest tests.test_visualization -v` and confirm the new assertions fail.**
- [x] **Step 6: Add the person status line and compact re-identification debug panel** to the overlay. Preserve existing drawing behavior when these optional values are omitted.
- [x] **Step 7: Instantiate the re-identifier from config in `main.py`**; pass the current normalized position and available shoulder/hip proportions on detections, and a missing-observation update when pose tracking returns no landmarks. Pass update state and candidate details to the overlay. Do not change `PoseTracker` or its one-pose configuration.
- [x] **Step 8: Run `python -m unittest tests.test_config tests.test_visualization -v` and confirm the integration-facing tests pass.**
- [x] **Step 9: Commit the tested runtime and HUD integration** with `feat: show anonymous person tracking state`.

### Task 3: Privacy limitations and full-suite verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Documents the behavior exposed by Tasks 1 and 2; no new code interface.

- [x] **Step 1: Update the README** with anonymous session ID lifecycle, local-only retention, factors and threshold/margin behavior, debug information, and limitations. State explicitly that single-camera continuity is imperfect, the one-pose detector does not resolve simultaneous people, and this does not establish identity.
- [x] **Step 2: Run the full existing suite** with `python -m unittest discover -s tests -v`.
- [x] **Step 3: Fix any regressions in the owning task and rerun the affected tests plus the full suite** until all tests pass.
- [x] **Step 4: Commit the README update and verified result** with `docs: explain anonymous track continuity limits`.
