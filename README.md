# Mindful Arm

**An interactive vision–language–action (VLA) pipeline visualizer.**

A simulated 6-DOF robot arm executes typed manipulation instructions while a synchronized panel shows the VLA pipeline deciding its every move — vision in, language in, fusion/reasoning, action out — so the "thought" and the "motion" happen together, in real time, in one view.

> **Illustrative model — mimics real VLA architecture, not live inference.**
> The pipeline *structure* is faithful to how real VLA models (RT-2, OpenVLA, π0) work: an image observation and a tokenized instruction are fused, and the result is decoded into action tokens that map to end-effector motion. The arm's kinematics are real (analytic inverse kinematics). The "reasoning" itself, however, is a rule-based stand-in — no trained neural network runs here, and the UI says so on-screen.

<!-- TODO: demo GIF -->
<!-- ![Mindful Arm demo](docs/demo.gif) -->

## Why this exists

Physical AI is everywhere, but the pipeline is invisible: you either see a robot moving (with no idea why) or a research diagram (with no robot moving). Mindful Arm shows both at once. Type an instruction, watch the pipeline stages illuminate one after another at the exact pace the arm's motion unfolds, and click any stage for an explanation of what happens there in a real VLA.

## Features

- **Synchronized pipeline panel** — four stages (Vision In → Language In → Fusion/Reasoning → Action Out) activate in sequence; the Action stage streams decoded action tokens (Δx, Δy, Δz, gripper) with the active token highlighted in lockstep with the arm's motion
- **Real inverse kinematics** — analytic elbow-up IK for a yaw–shoulder–elbow arm with a vertical-wrist constraint; live joint telemetry rendered in-scene
- **Scene understanding overlay** — the Vision stage sweeps the scene and draws detection boxes with object IDs, coordinates, and confidences
- **Grounding made visible** — the Fusion stage shows language phrases resolving to scene objects ("red block" → obj_01) and the emitted manipulation plan
- **Constrained instruction set** — three templates (place on shelf, move beside, stack on), parameterized by block color, parsed from free text; unsupported input gets a directive error
- **Explainable stages** — click any stage card for what happens there in a real VLA vs. what the scripted stand-in is doing

## Supported instructions

```
Pick up the red block and place it on the shelf
Move the blue block next to the green block
Stack the red block on the blue block
```

Any of the three verbs works with any block color in the scene (red, blue, green).

## Architecture

Single-file React component (`src/MindfulArm.jsx`), no dependencies beyond React itself.

| Layer | Implementation |
|---|---|
| Scene | 2.5D isometric rendering on `<canvas>` (depth-sorted cubes, shadows, grid) |
| Kinematics | Analytic IK: base yaw + 2-link planar (shoulder/elbow) + fixed vertical wrist |
| Motion | Cartesian waypoint planner (approach → grasp → lift → transfer → place → retract) with eased joint-space execution at ~210 mm/s |
| "Brain" | Rule-based parser → explicit grounding → plan → action-token stream (the honest fake) |
| Sync | One `requestAnimationFrame` loop drives both the arm and the active-token highlight |

## Running it

The component is self-contained. Drop it into any React app:

```bash
npm create vite@latest mindful-arm-app -- --template react
cd mindful-arm-app && npm install
# copy src/MindfulArm.jsx in, then in src/App.jsx:
#   import MindfulArm from './MindfulArm'
#   export default MindfulArm
npm run dev
```

## Roadmap (not in v1)

- Gesture mode — canned non-manipulation motions ("wave") that bypass the pipeline
- Mimic mode — webcam pose tracking, arm copies your movements (teleoperation by imitation)
- Open-ended instructions beyond the supported set

## License

MIT
