# Requirements for every NEW generation (from the Easy data review, 2026-10-13)

**Rule: published simulations are never changed.** Easy/City/Simulation_1..100 and Easy/Forest/Simulation_1..104
stay exactly as they are. Everything below applies only to batches generated from now on.

Source: the user's review of the 204 Easy sims (100 City + 104 Forest). Video, telemetry and actions were all
consistent; the issues are about what is missing or repetitive for VLN.

## 1. Instruction variety (most serious)
- Problem: 207/210 instructions start "Take off from the"; one fixed sentence per task type; only 211 distinct words;
  one instruction per flight.
- Do: 2-3 paraphrases per flight (LLM-generated, a human checks a sample) varying word order, verbs
  (circle / orbit / do a loop around) and how places are named (colour, size, nearby objects, direction);
  add a short version too. Store as `instructions: [...]` and keep the original `instruction` field.

## 2. Link instruction steps to frames
- Problem: `instruction_steps` is one entry (the whole sentence); nothing links text to `events`.
- Do: write `instruction_alignment`: `[{text, events: [i], frame_start, frame_end}, ...]`, built from `events`.

## 3. Per-frame target visibility
- Problem: no visibility / pixel position of landmarks; some hovers have the target in view < 80 % because the
  drone is still turning (old City 97: 49 %, 47: 55 %, 17: 61 %, 9: 68 %).
- Do: store per-frame visibility and pixel position of each landmark (telemetry + intrinsics); make the pilot
  finish turning to face the target BEFORE the hover clock starts; flag `look_at_quality` if still partial.

## 4. Depth and segmentation
- Problem: RGB video only.
- Do: enable Replicator depth + semantic/instance outputs in render_slice (16-bit PNG depth, indexed PNG masks,
  every frame or every 2nd); give every prop/landmark a semantic label in the scene builders
  (city_v2.py, city_props.py, props.py, easy_v2.py). Must be in place before the next batch is rendered.

## 5. Scene object list / map
- Problem: Forest v2 meta lists only the target; no top-down map or obstacle map.
- Do: write every placed object (name, type, position, size, description) to meta.json; optionally a top-down
  image and obstacle grid per sim.

## 6. Not only perfect flights (lower priority)
- Do: a small share with deliberate detours/corrections and some "wrong target" runs labelled as failures.

## 7. Metadata and documentation
- `coordinate_frame` block: ENU metres, heading 0 deg = +x, counter-clockwise positive, camera pitch positive down;
  full per-frame camera extrinsics (position + rotation).
- `splits.json` (e.g. 80/10/10) balanced by environment and task type, whole scene layout in one split.
- Rewrite ENVIRONMENT.txt for new folders (action.json = stick changes only, current code paths).
- `dataset_version` field in meta.json.
- Props not at rest stay flagged (`all_at_rest`).

## 8. More task types
- Add spatial/relational single tasks: "land next to X", "stop in front of X facing it",
  "fly to the Nth X along the street", stop-when-you-see-it, counting.

## Suggested order
1. Paraphrases (#1). 2. Alignment, visibility, coordinate notes, splits (#2, #3, #7).
3. Depth + segmentation (#4) before the next render. 4. Full object list (#5). 5. New task types and failure examples (#8, #6).

---
# Review of Easy Coastal v2.0/v2.1 (2026-10-20) — apply ONLY to new generations
Published sims (Forest, City, Coastal) stay untouched. These rules override earlier items where they conflict
(#4 depth/seg outputs and #7 splits.json are cancelled for publishing).

C1. task_visibility must be measured over the actual task segment only: circle = the orbit beat, inspect = the
    hover after the drone faces the target (aligned_at). Never include the approach.
C2. pass_over: evaluate only the frames where the drone is near/over the target (e.g. horizontal distance below
    a threshold around the closest point), not the whole goto/approach.
C3. instruction_alignment: each clause gets its own non-overlapping frame range; two clauses may not share one
    segment. Validate before publish.
C4. bbox_px: compute from the segmentation mask (visible pixels only, occlusion-aware), not from projected 3D box
    corners. Segmentation stays internal.
C5. No duplicate instructions across simulations (check all instruction variants of the batch + previous batches
    of the same environment; reject and re-word on overlap).
C6. Paraphrases must differ from the original: similarity limit < 0.8 (same metric for all), automatic check.
C7. Short instruction must keep every required quantity (distances, heights, counts, e.g. "66 m").
C8. scene_objects: every object gets a description (colour/material, size, role, location relative to landmarks),
    not just a class label, so similar objects can be told apart.
C9. Always write the `sky` field in meta.json.
C10. Events must cover the whole flight up to the last frame (add a final "landed / idle" event after touchdown).
C11. Publish only 4 files per sim: video (trajectory.mp4), telemetry.json, meta.json, action.json.
     No depth.mkv / segmentation.mkv / other extras in the output folders.
C12. No train/test splitting: do not write splits.json.
