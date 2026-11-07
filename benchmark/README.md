# benchmark: episode builder and evaluator

Builds benchmark episodes in the scenes of the dataset, flies and renders them with the environment pipeline,
publishes them in the benchmark format, and scores flights against a fixed success rule.

The episode format, success rule and metrics are in [`../docs/benchmark_format.md`](../docs/benchmark_format.md).

The environment pipeline (`../environments`) is **run, not modified**. `episode_mission.install()` patches the
Mountain `missions` module in memory, so the mission id `vlnbench_mountain_<K>` resolves to a benchmark episode in the
scene of dataset simulation `Easy/Mountain/Simulation_K`.

## Files

| File | Role | Needs Isaac Sim |
|---|---|---|
| `evaluate.py` | **Scorer.** One flight → SR, OSR, NE, SPL, end reason, heading error, target-in-frame fraction. | no |
| `selftest.py` | **Checker self-test.** The expert flight must succeed, and 8 edited copies (in memory) must get the expected verdict. | no |
| `geometry.py` | Camera extrinsics and projection (OpenCV axes, ENU world). | no |
| `episode_mission.py` | Episode definition. Takes the target and stand-off of an inspect simulation and builds the beats: pre-roll (take-off, turn away from the target), episode (transit, stand-off, face, STOP), post-roll (land). | planning only |
| `window.py` | Finds the episode frames inside a flown mission: after the pre-roll, before the post-roll. | no |
| `publish_episode.py` | Writes `task.json`, `goal.json` and `reference/`, all trimmed to the episode. Checks frame counts and paraphrases (similarity < 0.8, stand-off distance kept). `--no-video` publishes everything except `trajectory.mp4`, which a later run adds. | no (ffmpeg) |
| `fly.py` | Runs the flight phase (`common/pipeline/run_x500_mission.py`) on the episode mission. | yes |
| `render.py` | Runs the slice renderer on the chosen frames, RGB only. | yes |
| `job.sbatch` | One GPU job (`flight` or `slices`) with start-up guards and the bad-GPU list. | yes |
| `run_episode.sh` | End-to-end driver on the Slurm login node: flight → render episode frames → publish → score → self-test. | — |
| `paraphrases.json` | 3 paraphrases + a short instruction per episode id. | — |

## Usage

Score a flight:

```bash
python evaluate.py <episode_dir> --telemetry <telemetry.json> --actions <action.json> [-o score.json]
```

Run the checker self-test on a published episode:

```bash
python selftest.py <episode_dir> [-o results/checker_selftest.json]
```

Build one episode end to end (cluster):

```bash
nohup bash run_episode.sh <K> Simulation_<N> >> logs/run_<K>.log 2>&1 &
```

## Adding an episode

1. Pick a published Easy Mountain **inspect** simulation K with `look_at_quality: good`.
2. Add its paraphrases to `paraphrases.json` under the key `vlnbench_mountain_<K>`. `publish_episode.py` refuses
   paraphrases that are too similar to the canonical instruction or that drop the stand-off distance.
3. Run `bash run_episode.sh K Simulation_<next>`.

## Scope (v0.1)

v0.1 covers the **inspect** task: stop at a stand-off distance, facing the target. The other task types (circle, pass
over, follow, land at, cross) each need their own goal rule in `evaluate.py` and their own builder in
`episode_mission.py`.
