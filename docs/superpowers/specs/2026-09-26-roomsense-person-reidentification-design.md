# RoomSense Anonymous Track Re-identification Design

## Goal

Reconnect a returning camera track to a recently lost anonymous session person only when continuity evidence is strong enough. This feature does not identify a real-world person.

## Current architecture

`PoseTracker` uses MediaPipe to produce one pose at a time. `main.py` derives normalized position and movement, tracks missing observations, and sends pose and position information to `TrackingOverlay`. Tracking currently has no anonymous person ID or lost-track registry. Camera-free unit tests use `unittest`.

## Design

Add `roomsense/tracking/reidentification.py`, independent of pose inference. It accepts an optional observation containing a monotonic timestamp, normalized torso center and relative depth, movement direction/velocity, and coarse torso proportions derived from visible landmarks. On a missing observation it marks the active ID `LOST` and retains a compact in-memory record until its configurable timeout expires. On a new observation it scores every unexpired lost record and reconnects the best candidate only when the score passes the configured threshold and exceeds the next-best candidate by the configured ambiguity margin. Otherwise it allocates the next session ID (`PERSON_001`, etc.).

The score exposes normalized factor details for predicted position/trajectory continuity, movement direction and expected re-entry side, elapsed-time decay, and coarse torso geometry. No face, clothing, image crop, or appearance embedding is collected. Candidate geometry and motion stay in memory and are discarded when the process ends or the record expires. The ID counter and registry are never written to session or evaluation recordings.

The existing one-pose runtime remains the observation source. `main.py` passes current observations or missing timestamps to the re-identification module and supplies the resulting anonymous ID, state, and latest diagnostic details to `TrackingOverlay`. The HUD displays the anonymous ID and lifecycle state. Debug mode shows candidate IDs, each score factor, the total score, the selected threshold and margin, and a concise match/no-match reason.

Re-identification timeout, minimum score, and ambiguity margin are configurable and validated in `RoomSenseConfig`. Matching does not depend on calibration or persist across application launches.

## State and confidence behavior

- First observation allocates an ID and reports `NEW`, then `TRACKED` on continued observations.
- A missing observation moves the active ID to `LOST` and starts its retention timeout.
- A qualifying match restores that ID and reports `REACQUIRED` before returning to `TRACKED`.
- A low-confidence, expired, or ambiguous match allocates a new ID. No uncertain candidate is forced.
- Debug diagnostics retain all eligible candidate scores and state why each candidate was rejected or selected.

Confidence is an interpretable weighted combination of the four continuity factors. The threshold and ambiguity margin are configurable; their initial defaults must favor avoiding false reconnections over maximizing reconnection rate. The implementation will document the weights and defaults alongside the config fields.

## Validation

Add camera-free tests that simulate immediate return, return along the expected trajectory, timeout expiration, a clearly different return, ambiguous candidates, and multiple retained lost IDs. Run the complete existing suite with `python -m unittest discover -s tests -v`.

## Limitations

One RGB camera and one detected pose provide weak continuity evidence. Similar positions and body proportions can produce false or missed matches, occlusion can disrupt motion estimates, and the one-pose detector does not track multiple visible people simultaneously. IDs are temporary labels for one process only; they do not establish identity. README documentation must state these limitations and the confidence-based fallback behavior.
