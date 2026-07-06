import React, { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";

/* ============================================================
   MINDFUL ARM — VLA pipeline visualizer (v2, full 3D)
   A 6-DOF arm rendered in three.js executes constrained
   language instructions while a synchronized panel shows the
   vision–language–action pipeline stage by stage.
   Kinematics are real (analytic IK on a 6-joint chain).
   The "brain" is a rule-based stand-in for a real VLA.
   ============================================================ */

/* ---------- palette / type tokens ---------- */
const C = {
  bg: "#0C0F14",
  panel: "#12161E",
  panelUp: "#171C26",
  line: "#242C3A",
  lineSoft: "#1B2230",
  text: "#E7ECF3",
  muted: "#8B97A9",
  faint: "#5B6779",
  amber: "#F5A524",
  amberDim: "rgba(245,165,36,0.14)",
  cyan: "#5BC8E8",
  ok: "#57C77C",
  red: "#E5484D",
  blue: "#4A82F7",
  green: "#46A758",
};
const BLOCK_HEX = { red: 0xe5484d, blue: 0x4a82f7, green: 0x46a758 };
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace";

/* ---------- kinematic constants (mm) ----------
   Sim frame is robotics-style z-up; mapped to three.js y-up
   at render time via (x, y, z_up) -> (x, z_up, -y).          */
const S = 22;              // block edge
const BASE_H = 46;         // table -> shoulder joint (J2)
const L1 = 95;             // shoulder -> elbow
const L2 = 85;             // elbow -> wrist
const GRIPL = 34;          // wrist center -> fingertip
const HOME = { x: 112, y: 4, z: 96 };
const SHELF = { x: 60, y: -115, w: 56, d: 56, h: 40 };
const SPEED = 210;         // cartesian mm/s
const GRIP_MS = 480;

const INITIAL_BLOCKS = () => [
  { id: "obj_01", color: "red", x: 150, y: -55, z: 0 },
  { id: "obj_02", color: "blue", x: 150, y: 30, z: 0 },
  { id: "obj_03", color: "green", x: 115, y: 85, z: 0 },
];

const PRESETS = [
  "Pick up the red block and place it on the shelf",
  "Move the blue block next to the green block",
  "Stack the red block on the blue block",
];

/* ---------- math ---------- */
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const deg = (r) => (r * 180) / Math.PI;
const toV3 = (p) => new THREE.Vector3(p.x, p.z, -p.y); // sim z-up -> three y-up

/* Analytic IK for the position chain (J1 yaw, J2 shoulder, J3 elbow),
   with J5 wrist pitch holding the gripper vertical. J4/J6 are the
   redundant wrist rolls, held at 0 for this task family. */
function solveIK(tip) {
  const yaw = Math.atan2(tip.y, tip.x);
  let wr = Math.hypot(tip.x, tip.y);
  let wz = tip.z + GRIPL - BASE_H;
  let d = Math.hypot(wr, wz);
  const maxD = L1 + L2 - 0.75;
  if (d > maxD) { const s = maxD / d; wr *= s; wz *= s; d = maxD; }
  const a = Math.atan2(wz, wr);
  const interior = Math.acos(clamp((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1));
  const b = Math.acos(clamp((d * d + L1 * L1 - L2 * L2) / (2 * L1 * d), -1, 1));
  const shoulder = a + b;                 // elbow-up branch
  const elbowRel = -interior;             // relative link-2 angle
  const wristPitch = -Math.PI / 2 - shoulder - elbowRel; // point straight down
  return { yaw, shoulder, elbowRel, wristPitch, roll1: 0, roll2: 0 };
}

/* ---------- instruction parsing (rule-based stand-in) ---------- */
function parseInstruction(text) {
  const t = text.toLowerCase();
  const colors = [...t.matchAll(/\b(red|blue|green)\b/g)].map((m) => m[1]);
  if (t.includes("shelf") && colors.length >= 1) {
    return { type: "shelf", src: colors[0] };
  }
  if ((t.includes("stack") || /\bon (the )?(red|blue|green)\b/.test(t)) && colors.length >= 2 && colors[0] !== colors[1]) {
    return { type: "stack", src: colors[0], dst: colors[1] };
  }
  if ((t.includes("next to") || t.includes("beside") || t.includes("move")) && colors.length >= 2 && colors[0] !== colors[1]) {
    return { type: "move", src: colors[0], dst: colors[1] };
  }
  return null;
}

function describeGoal(cmd) {
  if (cmd.type === "shelf") return { name: "zone_shelf", verb: "PLACE" };
  if (cmd.type === "stack") return { name: `top of ${cmd.dst} block`, verb: "STACK" };
  return { name: `beside ${cmd.dst} block`, verb: "PLACE" };
}

function buildWaypoints(cmd, blocks) {
  const src = blocks.find((b) => b.color === cmd.src);
  let place, placeLabel;
  if (cmd.type === "shelf") {
    place = { x: SHELF.x, y: SHELF.y, tipZ: SHELF.h + S };
    placeLabel = "zone_shelf";
  } else {
    const dst = blocks.find((b) => b.color === cmd.dst);
    if (cmd.type === "stack") {
      place = { x: dst.x, y: dst.y, tipZ: dst.z + S + S };
      placeLabel = dst.id;
    } else {
      const cands = [[0, 38], [0, -38], [38, 0], [-38, 0]];
      let spot = null;
      for (const [ox, oy] of cands) {
        const x = dst.x + ox, y = dst.y + oy;
        const r = Math.hypot(x, y);
        const free = blocks.every((b) => b.id === src.id || Math.hypot(b.x - x, b.y - y) > 30);
        const onTable = x > 60 && x < 200 && y > -132 && y < 132;
        if (free && onTable && r > 70 && r < 172) { spot = { x, y }; break; }
      }
      spot = spot || { x: dst.x, y: dst.y + 38 };
      place = { ...spot, tipZ: S };
      placeLabel = `beside ${dst.id}`;
    }
  }
  const hover = 58;
  const sTop = src.z + S;
  const wps = [
    { tip: { x: src.x, y: src.y, z: sTop + hover }, grip: 1, label: "approach" },
    { tip: { x: src.x, y: src.y, z: sTop }, grip: 1, label: "descend" },
    { tip: { x: src.x, y: src.y, z: sTop }, grip: 0, label: "grasp", attach: true },
    { tip: { x: src.x, y: src.y, z: sTop + hover }, grip: 0, label: "lift" },
    { tip: { x: place.x, y: place.y, z: place.tipZ + hover }, grip: 0, label: "transfer" },
    { tip: { x: place.x, y: place.y, z: place.tipZ }, grip: 0, label: "lower" },
    { tip: { x: place.x, y: place.y, z: place.tipZ }, grip: 1, label: "release", detach: true },
    { tip: { x: place.x, y: place.y, z: place.tipZ + hover }, grip: 1, label: "retract" },
    { tip: { ...HOME }, grip: 1, label: "home" },
  ];
  return { wps, srcId: src.id, place, placeLabel };
}

function buildSegments(startTip, startGrip, wps) {
  const segs = [];
  let prevTip = { ...startTip }, prevGrip = startGrip;
  for (const w of wps) {
    const dx = w.tip.x - prevTip.x, dy = w.tip.y - prevTip.y, dz = w.tip.z - prevTip.z;
    const dist = Math.hypot(dx, dy, dz);
    const gripChange = w.grip !== prevGrip;
    const dur = gripChange && dist < 1 ? GRIP_MS : Math.max(340, (dist / SPEED) * 1000);
    segs.push({
      from: prevTip, to: w.tip, g0: prevGrip, g1: w.grip, dur,
      label: w.label, attach: !!w.attach, detach: !!w.detach,
      token: { dx: dx / 1000, dy: dy / 1000, dz: dz / 1000, g: w.grip, label: w.label },
    });
    prevTip = w.tip; prevGrip = w.grip;
  }
  return segs;
}

const fmt = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

/* ---------- stage metadata ---------- */
const STAGES = [
  {
    key: "vision", title: "Vision In", sub: "RGB observation → visual tokens",
    explain: "A camera frame is encoded — typically by a vision transformer — into patch embeddings the model can attend over. Here, a scripted detector reports each object's class, color, and table coordinates, standing in for those learned features.",
  },
  {
    key: "language", title: "Language In", sub: "Instruction → text tokens",
    explain: "The instruction is tokenized and embedded by a language backbone. Real VLAs (RT-2, OpenVLA, π0) inherit this from a pretrained vision-language model, which is why they generalize across phrasings. Here the tokens feed a rule-based parser instead.",
  },
  {
    key: "fusion", title: "Fusion / Reasoning", sub: "Grounding language in the scene",
    explain: "Visual and text tokens attend to each other in a shared transformer; \u201Cred block\u201D gets grounded to a specific object and a goal state is implicitly formed. Here an explicit planner does the grounding and emits a two-step manipulation plan.",
  },
  {
    key: "action", title: "Action Out", sub: "Decoded action tokens → motor commands",
    explain: "The fused representation is decoded into a sequence of action tokens: small end-effector deltas plus a gripper state. This part is real in this demo — each token below is consumed by analytic inverse kinematics driving all six joints.",
  },
];

/* ============================================================ */
export default function App() {
  const mountRef = useRef(null);
  const telemetryRef = useRef(null);
  const simRef = useRef({
    blocks: INITIAL_BLOCKS(),
    tip: { ...HOME },
    grip: 1,
    held: null,
    stage: -1,
    detShown: 0,
    scanT: -1,
    segments: null,
    segIdx: 0,
    segT: 0,
    placeMarker: null,
    running: false,
    time: 0,
    srcId: null,
    onDone: null,
  });
  const runIdRef = useRef(0);
  const timersRef = useRef([]);

  const [status, setStatus] = useState("idle");
  const [stage, setStage] = useState(-1);
  const [selStage, setSelStage] = useState(null);
  const [inputText, setInputText] = useState(PRESETS[0]);
  const [errMsg, setErrMsg] = useState("");
  const [detections, setDetections] = useState([]);
  const [langTokens, setLangTokens] = useState([]);
  const [grounding, setGrounding] = useState(null);
  const [actionTokens, setActionTokens] = useState([]);
  const [curToken, setCurToken] = useState(-1);

  const reduced = typeof window !== "undefined" &&
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const stageMs = reduced ? 700 : 1350;

  const later = useCallback((fn, ms) => {
    const id = setTimeout(fn, ms);
    timersRef.current.push(id);
    return id;
  }, []);
  const sleep = (ms) => new Promise((res) => later(res, ms));

  /* ---------- run pipeline (identical logic to v1) ---------- */
  const run = useCallback(async (text) => {
    const sim = simRef.current;
    if (sim.running) return;
    const cmd = parseInstruction(text);
    if (!cmd) {
      setErrMsg("Outside the supported instruction set. Try a preset, or name two block colors and an action (place on shelf / stack on / move next to).");
      return;
    }
    const srcExists = sim.blocks.some((b) => b.color === cmd.src);
    const dstOk = !cmd.dst || sim.blocks.some((b) => b.color === cmd.dst);
    if (!srcExists || !dstOk) { setErrMsg("That block isn't in the scene."); return; }

    setErrMsg("");
    const myRun = ++runIdRef.current;
    const alive = () => runIdRef.current === myRun;

    sim.running = true;
    setStatus("running");
    setSelStage(null);
    setDetections([]); setLangTokens([]); setGrounding(null);
    setActionTokens([]); setCurToken(-1);

    /* Stage 1 — Vision In */
    sim.stage = 0; setStage(0);
    sim.scanT = 0; sim.detShown = 0;
    const dets = sim.blocks.map((b) => ({
      id: b.id, cls: `cube_${b.color}`, x: b.x, y: b.y,
      conf: (0.93 + Math.random() * 0.06).toFixed(2),
    }));
    dets.push({ id: "zone_01", cls: "shelf", x: SHELF.x, y: SHELF.y, conf: "0.99" });
    for (let i = 0; i < dets.length; i++) {
      later(() => {
        if (!alive()) return;
        sim.detShown = i + 1;
        setDetections(dets.slice(0, i + 1));
      }, (stageMs / (dets.length + 1)) * (i + 1));
    }
    await sleep(stageMs);
    if (!alive()) return;
    sim.scanT = -1;

    /* Stage 2 — Language In */
    sim.stage = 1; setStage(1);
    const words = text.trim().split(/\s+/);
    for (let i = 0; i < words.length; i++) {
      later(() => { if (alive()) setLangTokens(words.slice(0, i + 1)); },
        Math.min(900, stageMs * 0.7) / words.length * (i + 1));
    }
    await sleep(stageMs);
    if (!alive()) return;

    /* Stage 3 — Fusion / Reasoning */
    sim.stage = 2; setStage(2);
    const built = buildWaypoints(cmd, sim.blocks);
    const goal = describeGoal(cmd);
    const srcBlock = sim.blocks.find((b) => b.color === cmd.src);
    later(() => {
      if (!alive()) return;
      setGrounding({
        links: [
          { phrase: `${cmd.src} block`, target: srcBlock.id },
          { phrase: cmd.type === "shelf" ? "the shelf" : `${cmd.dst} block`, target: cmd.type === "shelf" ? "zone_01" : sim.blocks.find((b) => b.color === cmd.dst).id },
        ],
        plan: [`GRASP(${srcBlock.id})`, `${goal.verb}(${built.placeLabel})`],
      });
    }, stageMs * 0.35);
    await sleep(stageMs * 1.15);
    if (!alive()) return;

    /* Stage 4 — Action Out (synchronized with motion) */
    sim.stage = 3; setStage(3);
    const segs = buildSegments(sim.tip, sim.grip, built.wps);
    setActionTokens(segs.map((s) => s.token));
    sim.placeMarker = { x: built.place.x, y: built.place.y, z: built.place.tipZ - S };
    sim.segments = segs;
    sim.segIdx = 0; sim.segT = 0;
    sim.srcId = built.srcId;
    setCurToken(0);

    await new Promise((res) => { sim.onDone = res; });
    if (!alive()) return;

    sim.placeMarker = null;
    sim.stage = -1; setStage(-1);
    sim.running = false;
    setStatus("done");
  }, [later, stageMs]);

  const reset = useCallback(() => {
    runIdRef.current++;
    timersRef.current.forEach(clearTimeout);
    timersRef.current = [];
    const sim = simRef.current;
    sim.blocks = INITIAL_BLOCKS();
    sim.tip = { ...HOME }; sim.grip = 1; sim.held = null;
    sim.stage = -1; sim.segments = null; sim.placeMarker = null;
    sim.scanT = -1; sim.running = false; sim.detShown = 0; sim.onDone = null;
    setStatus("idle"); setStage(-1); setSelStage(null);
    setDetections([]); setLangTokens([]); setGrounding(null);
    setActionTokens([]); setCurToken(-1); setErrMsg("");
  }, []);

  /* ---------- three.js scene ---------- */
  useEffect(() => {
    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0f14);
    scene.fog = new THREE.Fog(0x0c0f14, 950, 1900);

    const camera = new THREE.PerspectiveCamera(42, 1, 1, 5000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mount.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.borderRadius = "8px";
    renderer.domElement.style.touchAction = "none";

    /* lights */
    scene.add(new THREE.AmbientLight(0x9db2d0, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 0.95);
    key.position.set(180, 320, 160);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -260; key.shadow.camera.right = 260;
    key.shadow.camera.top = 260; key.shadow.camera.bottom = -260;
    key.shadow.camera.far = 900;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x5bc8e8, 0.18);
    fill.position.set(-200, 140, -180);
    scene.add(fill);

    /* table */
    const tableMat = new THREE.MeshStandardMaterial({ color: 0x131a26, roughness: 0.92, metalness: 0.08 });
    const table = new THREE.Mesh(new THREE.BoxGeometry(330, 10, 330), tableMat);
    table.position.set(90, -5, 5);
    table.receiveShadow = true;
    scene.add(table);
    const grid = new THREE.GridHelper(330, 11, 0x2a3444, 0x1c2432);
    grid.position.set(90, 0.15, 5);
    scene.add(grid);

    /* shelf */
    const shelfMat = new THREE.MeshStandardMaterial({ color: 0x283242, roughness: 0.8, metalness: 0.2 });
    const shelf = new THREE.Mesh(new THREE.BoxGeometry(SHELF.w, SHELF.h, SHELF.d), shelfMat);
    shelf.position.set(SHELF.x, SHELF.h / 2, -SHELF.y);
    shelf.castShadow = true; shelf.receiveShadow = true;
    scene.add(shelf);

    /* blocks */
    const blockMeshes = {};
    for (const b of simRef.current.blocks) {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(S, S, S),
        new THREE.MeshStandardMaterial({ color: BLOCK_HEX[b.color], roughness: 0.55, metalness: 0.1 })
      );
      m.castShadow = true; m.receiveShadow = true;
      scene.add(m);
      blockMeshes[b.id] = m;
    }

    /* ---------- 6-joint arm hierarchy ----------
       J1 base yaw · J2 shoulder · J3 elbow · J4 wrist roll ·
       J5 wrist pitch · J6 flange roll                        */
    const linkMat = new THREE.MeshStandardMaterial({ color: 0x4c5c7a, roughness: 0.42, metalness: 0.65 });
    const darkMat = new THREE.MeshStandardMaterial({ color: 0x222b3c, roughness: 0.6, metalness: 0.5 });
    const jointMat = new THREE.MeshStandardMaterial({ color: 0x141a26, roughness: 0.45, metalness: 0.6, emissive: 0x000000 });
    const jointMats = [];
    const mkJointMat = () => { const m = jointMat.clone(); jointMats.push(m); return m; };

    const plate = new THREE.Mesh(new THREE.CylinderGeometry(36, 40, 7, 32), darkMat);
    plate.position.y = 3.5; plate.castShadow = true; plate.receiveShadow = true;
    scene.add(plate);

    const j1 = new THREE.Group(); // base yaw
    scene.add(j1);
    const column = new THREE.Mesh(new THREE.CylinderGeometry(15, 19, BASE_H, 24), darkMat);
    column.position.y = BASE_H / 2; column.castShadow = true;
    j1.add(column);

    const j2 = new THREE.Group(); // shoulder pitch (about local z)
    j2.position.y = BASE_H;
    j1.add(j2);
    const shoulderBall = new THREE.Mesh(new THREE.SphereGeometry(11, 20, 16), mkJointMat());
    shoulderBall.castShadow = true;
    j2.add(shoulderBall);
    const link1 = new THREE.Mesh(new THREE.CylinderGeometry(6.5, 7.5, L1, 16), linkMat);
    link1.rotation.z = -Math.PI / 2; // cylinder y-axis -> local +x
    link1.position.x = L1 / 2;
    link1.castShadow = true;
    j2.add(link1);

    const j3 = new THREE.Group(); // elbow pitch
    j3.position.x = L1;
    j2.add(j3);
    const elbowBall = new THREE.Mesh(new THREE.SphereGeometry(9, 20, 16), mkJointMat());
    elbowBall.castShadow = true;
    j3.add(elbowBall);
    const link2 = new THREE.Mesh(new THREE.CylinderGeometry(5.2, 6.2, L2, 16), linkMat);
    link2.rotation.z = -Math.PI / 2;
    link2.position.x = L2 / 2;
    link2.castShadow = true;
    j3.add(link2);

    const j4 = new THREE.Group(); // wrist roll (about local x = link axis)
    j4.position.x = L2;
    j3.add(j4);
    const rollHousing = new THREE.Mesh(new THREE.CylinderGeometry(6, 6, 13, 16), darkMat);
    rollHousing.rotation.z = -Math.PI / 2;
    rollHousing.castShadow = true;
    j4.add(rollHousing);

    const j5 = new THREE.Group(); // wrist pitch
    j4.add(j5);
    const wristBall = new THREE.Mesh(new THREE.SphereGeometry(7.5, 18, 14), mkJointMat());
    wristBall.castShadow = true;
    j5.add(wristBall);

    const j6 = new THREE.Group(); // flange roll (about local x = tool axis)
    j5.add(j6);
    const palm = new THREE.Mesh(new THREE.BoxGeometry(6, 9, 32), linkMat);
    palm.position.x = 5; palm.castShadow = true;
    j6.add(palm);
    const fingerGeo = new THREE.BoxGeometry(GRIPL - 6, 3.6, 4.2);
    const fingerL = new THREE.Mesh(fingerGeo, linkMat);
    const fingerR = new THREE.Mesh(fingerGeo, linkMat);
    fingerL.position.x = 8 + (GRIPL - 6) / 2;
    fingerR.position.x = 8 + (GRIPL - 6) / 2;
    fingerL.castShadow = true; fingerR.castShadow = true;
    j6.add(fingerL); j6.add(fingerR);

    /* place marker */
    const marker = new THREE.Mesh(
      new THREE.RingGeometry(12, 15.5, 40),
      new THREE.MeshBasicMaterial({ color: 0xf5a524, transparent: true, opacity: 0.9, side: THREE.DoubleSide })
    );
    marker.rotation.x = -Math.PI / 2;
    marker.visible = false;
    scene.add(marker);

    /* vision scan plane */
    const scanPlane = new THREE.Mesh(
      new THREE.PlaneGeometry(330, 135),
      new THREE.MeshBasicMaterial({ color: 0x5bc8e8, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false })
    );
    scanPlane.rotation.y = Math.PI / 2;
    scanPlane.visible = false;
    scene.add(scanPlane);

    /* detection wireframes + labels */
    const detColor = new THREE.LineBasicMaterial({ color: 0x5bc8e8, transparent: true, opacity: 0.9 });
    const makeLabel = (text) => {
      const cv = document.createElement("canvas");
      cv.width = 256; cv.height = 48;
      const cx = cv.getContext("2d");
      cx.font = "500 26px 'IBM Plex Mono', monospace";
      cx.fillStyle = "#5BC8E8";
      cx.textBaseline = "middle";
      cx.fillText(text, 8, 24);
      const tex = new THREE.CanvasTexture(cv);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
      sp.scale.set(64, 12, 1);
      return sp;
    };
    const detVis = [];
    for (const b of simRef.current.blocks) {
      const wf = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(S + 9, S + 9, S + 9)), detColor);
      const label = makeLabel(b.id);
      wf.visible = false; label.visible = false;
      scene.add(wf); scene.add(label);
      detVis.push({ kind: "block", id: b.id, wf, label });
    }
    {
      const wf = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(SHELF.w + 10, SHELF.h + 10, SHELF.d + 10)), detColor);
      const label = makeLabel("zone_01");
      wf.visible = false; label.visible = false;
      wf.position.set(SHELF.x, SHELF.h / 2, -SHELF.y);
      label.position.set(SHELF.x, SHELF.h + 24, -SHELF.y);
      scene.add(wf); scene.add(label);
      detVis.push({ kind: "shelf", id: "zone_01", wf, label });
    }

    /* ---------- custom orbit controls ---------- */
    const orbit = { theta: 0.85, phi: 1.08, R: 460, target: new THREE.Vector3(80, 40, 0) };
    const applyCamera = () => {
      const { theta, phi, R, target } = orbit;
      camera.position.set(
        target.x + R * Math.sin(phi) * Math.cos(theta),
        target.y + R * Math.cos(phi),
        target.z + R * Math.sin(phi) * Math.sin(theta)
      );
      camera.lookAt(target);
    };
    applyCamera();

    const pointers = new Map();
    let pinchDist = 0;
    const el = renderer.domElement;
    const onDown = (e) => {
      el.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
      }
    };
    const onMove = (e) => {
      if (!pointers.has(e.pointerId)) return;
      const prev = pointers.get(e.pointerId);
      const dx = e.clientX - prev.x, dy = e.clientY - prev.y;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 1) {
        orbit.theta += dx * 0.0055;
        orbit.phi = clamp(orbit.phi - dy * 0.0045, 0.25, 1.45);
        applyCamera();
      } else if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDist > 0) {
          orbit.R = clamp(orbit.R * (pinchDist / d), 220, 1100);
          applyCamera();
        }
        pinchDist = d;
      }
    };
    const onUp = (e) => { pointers.delete(e.pointerId); pinchDist = 0; };
    const onWheel = (e) => {
      e.preventDefault();
      orbit.R = clamp(orbit.R * (1 + e.deltaY * 0.001), 220, 1100);
      applyCamera();
    };
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("wheel", onWheel, { passive: false });

    /* resize */
    const resize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    /* ---------- animation loop ---------- */
    let raf, last = performance.now();
    const frame = (now) => {
      const sim = simRef.current;
      const dt = Math.min(50, now - last); last = now;
      sim.time += dt;

      /* advance motion (same state machine as v1) */
      if (sim.segments) {
        const seg = sim.segments[sim.segIdx];
        sim.segT += dt;
        const t = ease(clamp(sim.segT / seg.dur, 0, 1));
        sim.tip = {
          x: lerp(seg.from.x, seg.to.x, t),
          y: lerp(seg.from.y, seg.to.y, t),
          z: lerp(seg.from.z, seg.to.z, t),
        };
        sim.grip = lerp(seg.g0, seg.g1, t);
        if (sim.held) {
          const b = sim.blocks.find((k) => k.id === sim.held);
          b.x = sim.tip.x; b.y = sim.tip.y; b.z = sim.tip.z - S;
        }
        if (sim.segT >= seg.dur) {
          if (seg.attach) sim.held = sim.srcId;
          if (seg.detach) {
            const b = sim.blocks.find((k) => k.id === sim.held);
            if (b) { b.x = seg.to.x; b.y = seg.to.y; b.z = seg.to.z - S; }
            sim.held = null;
          }
          sim.segIdx++;
          sim.segT = 0;
          if (sim.segIdx >= sim.segments.length) {
            sim.segments = null;
            setCurToken(-1);
            if (sim.onDone) { const f = sim.onDone; sim.onDone = null; f(); }
          } else {
            setCurToken(sim.segIdx);
          }
        }
      } else if (!sim.running) {
        const b = Math.sin(sim.time / 1400) * 3;
        sim.tip = { x: HOME.x, y: HOME.y, z: HOME.z + b };
      }
      if (sim.scanT >= 0) sim.scanT = Math.min(1, sim.scanT + dt / stageMs);

      /* solve IK, drive the 6-joint chain */
      const j = solveIK(sim.tip);
      j1.rotation.y = j.yaw;
      j2.rotation.z = j.shoulder;
      j3.rotation.z = j.elbowRel;
      j4.rotation.x = j.roll1;
      j5.rotation.z = j.wristPitch;
      j6.rotation.x = j.roll2;
      const sep = 5 + sim.grip * 9;
      fingerL.position.z = sep;
      fingerR.position.z = -sep;

      /* joint glow while executing */
      const glow = sim.stage === 3 ? 0xf5a524 : 0x000000;
      for (const m of jointMats) m.emissive.setHex(glow);
      for (const m of jointMats) m.emissiveIntensity = sim.stage === 3 ? 0.55 : 0;

      /* blocks */
      for (const b of sim.blocks) {
        const mesh = blockMeshes[b.id];
        mesh.position.set(b.x, b.z + S / 2, -b.y);
      }

      /* marker */
      if (sim.placeMarker) {
        marker.visible = true;
        marker.position.set(sim.placeMarker.x, sim.placeMarker.z + 0.8, -sim.placeMarker.y);
      } else marker.visible = false;

      /* vision overlays */
      const vision = sim.stage === 0;
      scanPlane.visible = vision && sim.scanT >= 0;
      if (scanPlane.visible) {
        scanPlane.position.set(-45 + sim.scanT * 270, 66, 5);
      }
      detVis.forEach((d, i) => {
        const show = vision && i < sim.detShown;
        d.wf.visible = show; d.label.visible = show;
        if (show && d.kind === "block") {
          const b = sim.blocks.find((k) => k.id === d.id);
          d.wf.position.set(b.x, b.z + S / 2, -b.y);
          d.label.position.set(b.x, b.z + S + 22, -b.y);
        }
      });

      /* telemetry */
      if (telemetryRef.current) {
        telemetryRef.current.textContent =
          `J1 ${deg(j.yaw).toFixed(1)}°  J2 ${deg(j.shoulder).toFixed(1)}°  J3 ${deg(j.elbowRel).toFixed(1)}°  ` +
          `J4 ${deg(j.roll1).toFixed(1)}°  J5 ${deg(j.wristPitch).toFixed(1)}°  J6 ${deg(j.roll2).toFixed(1)}°  ` +
          `grip ${(sim.grip * 100).toFixed(0)}%  tip [${(sim.tip.x / 1000).toFixed(3)}, ${(sim.tip.y / 1000).toFixed(3)}, ${(sim.tip.z / 1000).toFixed(3)}] m` +
          (sim.running ? "   ● EXECUTING" : "   ○ IDLE");
      }

      renderer.render(scene, camera);
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      el.removeEventListener("wheel", onWheel);
      renderer.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
    };
  }, [stageMs]);

  useEffect(() => () => timersRef.current.forEach(clearTimeout), []);

  /* ---------- stage card content ---------- */
  const stageBody = (i) => {
    if (i === 0) {
      return detections.length === 0 ? (
        <Hint>Waiting for observation…</Hint>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          {detections.map((d) => (
            <div key={d.id} style={{ display: "flex", justifyContent: "space-between", fontFamily: MONO, fontSize: 11 }}>
              <span style={{ color: C.cyan }}>{d.id}</span>
              <span style={{ color: C.text }}>{d.cls}</span>
              <span style={{ color: C.muted }}>({(d.x / 1000).toFixed(2)}, {(d.y / 1000).toFixed(2)})</span>
              <span style={{ color: C.faint }}>{d.conf}</span>
            </div>
          ))}
        </div>
      );
    }
    if (i === 1) {
      return langTokens.length === 0 ? (
        <Hint>Waiting for instruction…</Hint>
      ) : (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
          {langTokens.map((w, k) => (
            <span key={k} style={{
              fontFamily: MONO, fontSize: 11, padding: "2px 6px",
              background: "#1B2230", border: `1px solid ${C.line}`,
              borderRadius: 4, color: C.text,
            }}>{w}</span>
          ))}
          <span style={{ fontFamily: MONO, fontSize: 10, color: C.faint, alignSelf: "center" }}>
            → {langTokens.length} tokens
          </span>
        </div>
      );
    }
    if (i === 2) {
      return !grounding ? (
        <Hint>Waiting for fused context…</Hint>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {grounding.links.map((l, k) => (
            <div key={k} style={{ fontFamily: MONO, fontSize: 11 }}>
              <span style={{ color: C.text }}>“{l.phrase}”</span>
              <span style={{ color: C.faint }}> → </span>
              <span style={{ color: C.cyan }}>{l.target}</span>
              <span style={{ color: C.ok }}> ✓</span>
            </div>
          ))}
          <div style={{ fontFamily: MONO, fontSize: 11, color: C.amber, marginTop: 2 }}>
            plan: {grounding.plan.join(" → ")}
          </div>
        </div>
      );
    }
    return actionTokens.length === 0 ? (
      <Hint>No action chunk decoded yet.</Hint>
    ) : (
      <div style={{ display: "flex", flexDirection: "column", gap: 2, maxHeight: 168, overflowY: "auto" }}>
        {actionTokens.map((t, k) => {
          const active = k === curToken;
          const done = curToken === -1 ? status === "done" : k < curToken;
          return (
            <div key={k} style={{
              display: "flex", gap: 8, alignItems: "baseline",
              fontFamily: MONO, fontSize: 10.5, padding: "2px 6px",
              borderRadius: 4,
              background: active ? C.amberDim : "transparent",
              border: `1px solid ${active ? "rgba(245,165,36,0.5)" : "transparent"}`,
              color: active ? C.text : done ? C.faint : C.muted,
              transition: "background 200ms",
            }}>
              <span style={{ color: active ? C.amber : C.faint, minWidth: 26 }}>a{String(k).padStart(2, "0")}</span>
              <span>Δx {fmt(t.dx)}</span>
              <span>Δy {fmt(t.dy)}</span>
              <span>Δz {fmt(t.dz)}</span>
              <span>g {t.g.toFixed(1)}</span>
              <span style={{ marginLeft: "auto", color: active ? C.amber : C.faint }}>{t.label}</span>
            </div>
          );
        })}
      </div>
    );
  };

  /* ---------- layout ---------- */
  return (
    <div style={{
      minHeight: "100vh", background: C.bg, color: C.text,
      fontFamily: "'Space Grotesk', system-ui, sans-serif",
      display: "flex", flexDirection: "column",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-thumb { background: #2A3444; border-radius: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        button { cursor: pointer; }
        button:focus-visible, input:focus-visible { outline: 2px solid ${C.amber}; outline-offset: 2px; }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
        @media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
      `}</style>

      {/* header */}
      <header style={{
        display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap",
        padding: "14px 20px", borderBottom: `1px solid ${C.line}`,
      }}>
        <div>
          <div style={{ fontSize: 17, fontWeight: 600, letterSpacing: "0.04em" }}>
            MINDFUL ARM
          </div>
          <div style={{ fontSize: 11.5, color: C.muted, marginTop: 1 }}>
            Vision · Language · Action — watch the pipeline decide while the arm moves
          </div>
        </div>
        <div style={{
          marginLeft: "auto", fontFamily: MONO, fontSize: 10.5,
          color: C.amber, border: `1px solid rgba(245,165,36,0.45)`,
          background: C.amberDim, borderRadius: 5, padding: "5px 10px",
        }}>
          ILLUSTRATIVE MODEL — mimics real VLA architecture, not live inference
        </div>
      </header>

      {/* main */}
      <main style={{
        flex: 1, display: "grid", gap: 14, padding: 16,
        gridTemplateColumns: "minmax(300px, 370px) 1fr",
      }}>
        {/* pipeline column */}
        <section aria-label="VLA pipeline" style={{ display: "flex", flexDirection: "column", gap: 0, minWidth: 0 }}>
          {STAGES.map((s, i) => {
            const active = stage === i;
            const isDone = status === "done" || (stage > -1 && i < stage);
            const open = selStage === i;
            return (
              <div key={s.key} style={{ display: "flex", gap: 10, minWidth: 0 }}>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 16 }}>
                  <div style={{
                    width: 10, height: 10, borderRadius: "50%", marginTop: 16, flexShrink: 0,
                    background: active ? C.amber : isDone ? "#3E4C63" : "#1B2230",
                    border: `1.5px solid ${active ? C.amber : isDone ? "#576A88" : C.line}`,
                    boxShadow: active ? `0 0 10px ${C.amber}` : "none",
                    animation: active ? "pulse 1.2s ease-in-out infinite" : "none",
                  }} />
                  {i < STAGES.length - 1 && (
                    <div style={{
                      width: 2, flex: 1, background: isDone ? "#3E4C63" : C.lineSoft,
                      transition: "background 300ms",
                    }} />
                  )}
                </div>
                <button
                  onClick={() => setSelStage(open ? null : i)}
                  aria-expanded={open}
                  style={{
                    textAlign: "left", width: "100%", marginBottom: 10, minWidth: 0,
                    background: active ? C.panelUp : C.panel,
                    border: `1px solid ${active ? "rgba(245,165,36,0.55)" : C.line}`,
                    borderRadius: 8, padding: "10px 12px", color: C.text,
                    transition: "border-color 250ms, background 250ms",
                    font: "inherit",
                  }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
                    <span style={{ fontFamily: MONO, fontSize: 10, color: active ? C.amber : C.faint }}>
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span style={{ fontWeight: 600, fontSize: 13.5 }}>{s.title}</span>
                    <span style={{ marginLeft: "auto", fontFamily: MONO, fontSize: 9.5, color: active ? C.amber : C.faint }}>
                      {active ? "ACTIVE" : isDone ? "DONE" : "IDLE"}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: C.muted, margin: "2px 0 8px" }}>{s.sub}</div>
                  {stageBody(i)}
                  {open && (
                    <div style={{
                      marginTop: 9, paddingTop: 8, borderTop: `1px solid ${C.lineSoft}`,
                      fontSize: 11.5, lineHeight: 1.55, color: C.muted,
                    }}>
                      {s.explain}
                    </div>
                  )}
                  <div style={{ fontSize: 9.5, color: C.faint, marginTop: 6 }}>
                    {open ? "click to collapse" : "click for what happens here in a real VLA"}
                  </div>
                </button>
              </div>
            );
          })}
        </section>

        {/* 3D scene column */}
        <section aria-label="Simulated 3D scene" style={{
          background: C.panel, border: `1px solid ${C.line}`, borderRadius: 10,
          padding: 10, display: "flex", flexDirection: "column", minWidth: 0, gap: 8,
        }}>
          <div
            ref={mountRef}
            role="img"
            aria-label="3D tabletop scene with a six-jointed robot arm, colored blocks, and a shelf"
            style={{ flex: 1, minHeight: 380, position: "relative", borderRadius: 8, overflow: "hidden" }}
          />
          <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div ref={telemetryRef} style={{ fontFamily: MONO, fontSize: 10, color: C.faint, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} />
            <div style={{ fontFamily: MONO, fontSize: 10, color: C.faint, flexShrink: 0 }}>
              drag to orbit · scroll / pinch to zoom
            </div>
          </div>
        </section>
      </main>

      {/* command bar */}
      <footer style={{ borderTop: `1px solid ${C.line}`, padding: "12px 20px 16px" }}>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <input
            value={inputText}
            onChange={(e) => { setInputText(e.target.value); setErrMsg(""); }}
            onKeyDown={(e) => { if (e.key === "Enter" && status !== "running") run(inputText); }}
            placeholder="Type a manipulation instruction…"
            aria-label="Manipulation instruction"
            style={{
              flex: "1 1 320px", background: "#0F131B", color: C.text,
              border: `1px solid ${errMsg ? "rgba(229,72,77,0.6)" : C.line}`, borderRadius: 7,
              padding: "10px 12px", fontFamily: MONO, fontSize: 12.5,
            }}
          />
          <button
            onClick={() => run(inputText)}
            disabled={status === "running"}
            style={{
              background: status === "running" ? "#3A3122" : C.amber,
              color: status === "running" ? C.muted : "#141005",
              border: "none", borderRadius: 7, padding: "10px 20px",
              fontWeight: 600, fontSize: 13, letterSpacing: "0.03em",
              opacity: status === "running" ? 0.8 : 1, font: "inherit",
            }}>
            {status === "running" ? "RUNNING…" : "RUN"}
          </button>
          <button
            onClick={reset}
            style={{
              background: "transparent", color: C.muted,
              border: `1px solid ${C.line}`, borderRadius: 7, padding: "10px 14px",
              fontSize: 12.5, font: "inherit",
            }}>
            Reset scene
          </button>
        </div>
        {errMsg && (
          <div style={{ marginTop: 7, fontSize: 11.5, color: C.red, fontFamily: MONO }}>{errMsg}</div>
        )}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 9, alignItems: "center" }}>
          <span style={{ fontSize: 10.5, color: C.faint, fontFamily: MONO }}>SUPPORTED:</span>
          {PRESETS.map((p) => (
            <button key={p}
              onClick={() => { setInputText(p); setErrMsg(""); if (status !== "running") run(p); }}
              style={{
                background: "#151B26", border: `1px solid ${C.line}`, color: C.muted,
                borderRadius: 999, padding: "5px 11px", fontSize: 11, font: "inherit",
              }}>
              {p}
            </button>
          ))}
        </div>
      </footer>
    </div>
  );
}

function Hint({ children }) {
  return <div style={{ fontSize: 11, color: "#5B6779", fontFamily: MONO }}>{children}</div>;
}
