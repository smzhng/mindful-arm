import React, { useEffect, useRef, useState, useCallback } from "react";

/* ============================================================
   MINDFUL ARM — VLA pipeline visualizer (v1)
   A simulated 6-DOF arm executes constrained language
   instructions while a synchronized panel shows the
   vision–language–action pipeline stage by stage.
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
const BLOCK_HEX = { red: C.red, blue: C.blue, green: C.green };

/* ---------- kinematic constants (mm) ---------- */
const S = 22;              // block edge
const BASE_H = 46;         // table -> shoulder joint
const L1 = 95;             // shoulder -> elbow
const L2 = 85;             // elbow -> wrist
const GRIPL = 34;          // wrist -> gripper tip
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

/* Analytic IK: gripper points straight down.
   tip = desired gripper-tip position in world mm. */
function solveIK(tip) {
  const yaw = Math.atan2(tip.y, tip.x);
  const r = Math.hypot(tip.x, tip.y);
  // wrist sits GRIPL above the tip; shoulder is origin of the plane
  let wr = r;
  let wz = tip.z + GRIPL - BASE_H;
  let d = Math.hypot(wr, wz);
  const maxD = L1 + L2 - 0.75;
  if (d > maxD) {
    const s = maxD / d;
    wr *= s; wz *= s; d = maxD;
  }
  const a = Math.atan2(wz, wr);
  const cosInt = clamp((d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1);
  const interior = Math.acos(cosInt);              // 0..pi
  const b = Math.acos(clamp((d * d + L1 * L1 - L2 * L2) / (2 * L1 * d), -1, 1));
  const shoulder = a + b;                          // elbow-up branch
  const elbowRel = -interior;                      // relative link-2 angle
  return { yaw, shoulder, elbowRel };
}

/* Forward kinematics -> world positions of each joint */
function forwardK(j) {
  const cy = Math.cos(j.yaw), sy = Math.sin(j.yaw);
  const sh = { x: 0, y: 0, z: BASE_H };
  const a1 = j.shoulder;
  const el = {
    x: cy * L1 * Math.cos(a1),
    y: sy * L1 * Math.cos(a1),
    z: BASE_H + L1 * Math.sin(a1),
  };
  const a2 = j.shoulder + j.elbowRel;
  const wr = {
    x: el.x + cy * L2 * Math.cos(a2),
    y: el.y + sy * L2 * Math.cos(a2),
    z: el.z + L2 * Math.sin(a2),
  };
  const tip = { x: wr.x, y: wr.y, z: wr.z - GRIPL };
  return { sh, el, wr, tip };
}

/* ---------- isometric projection ---------- */
const ISO = { cx: 258, cy: 208, ax: 0.86, ay: 0.5 };
const P = (x, y, z) => [ISO.cx + (x - y) * ISO.ax, ISO.cy + (x + y) * ISO.ay - z];

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

/* Build the waypoint sequence for a parsed command */
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
      token: {
        dx: dx / 1000, dy: dy / 1000, dz: dz / 1000, g: w.grip, label: w.label,
      },
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
    explain: "Visual and text tokens attend to each other in a shared transformer; “red block” gets grounded to a specific object and a goal state is implicitly formed. Here an explicit planner does the grounding and emits a two-step manipulation plan.",
  },
  {
    key: "action", title: "Action Out", sub: "Decoded action tokens → motor commands",
    explain: "The fused representation is decoded into a sequence of action tokens: small end-effector deltas plus a gripper state. This part is real in this demo — each token below is consumed by analytic inverse kinematics that drives the arm's joints.",
  },
];

/* ============================================================ */
export default function App() {
  const canvasRef = useRef(null);
  const simRef = useRef({
    blocks: INITIAL_BLOCKS(),
    tip: { ...HOME },
    grip: 1,
    held: null,
    stage: -1,            // -1 idle, 0..3 pipeline
    detShown: 0,
    scanT: -1,            // 0..1 sweep during vision
    segments: null,
    segIdx: 0,
    segT: 0,
    placeMarker: null,
    running: false,
    time: 0,
  });
  const runIdRef = useRef(0);
  const timersRef = useRef([]);

  const [status, setStatus] = useState("idle"); // idle | running | done
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

  /* ---------- run pipeline ---------- */
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

    // The animation loop advances segments; wait for it to finish.
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

  /* ---------- animation + render loop ---------- */
  useEffect(() => {
    const cvs = canvasRef.current;
    const ctx = cvs.getContext("2d");
    const W = 620, H = 470;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cvs.width = W * dpr; cvs.height = H * dpr;
    ctx.scale(dpr, dpr);

    let raf, last = performance.now();

    const drawCube = (x, y, zb, s, hex) => {
      const h = s / 2;
      const T = [P(x - h, y - h, zb + s), P(x + h, y - h, zb + s), P(x + h, y + h, zb + s), P(x - h, y + h, zb + s)];
      const Fx = [P(x + h, y - h, zb + s), P(x + h, y + h, zb + s), P(x + h, y + h, zb), P(x + h, y - h, zb)];
      const Fy = [P(x - h, y + h, zb + s), P(x + h, y + h, zb + s), P(x + h, y + h, zb), P(x - h, y + h, zb)];
      const poly = (pts, fill) => {
        ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
        pts.slice(1).forEach((p) => ctx.lineTo(p[0], p[1]));
        ctx.closePath(); ctx.fillStyle = fill; ctx.fill();
        ctx.strokeStyle = "rgba(0,0,0,0.35)"; ctx.lineWidth = 1; ctx.stroke();
      };
      poly(Fy, shade(hex, -0.32));
      poly(Fx, shade(hex, -0.16));
      poly(T, shade(hex, 0.12));
    };

    const shade = (hex, amt) => {
      const n = parseInt(hex.slice(1), 16);
      let r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255;
      const f = (v) => clamp(Math.round(amt >= 0 ? v + (255 - v) * amt : v * (1 + amt)), 0, 255);
      return `rgb(${f(r)},${f(g)},${f(b)})`;
    };

    const drawShadow = (x, y, z, r) => {
      const [px, py] = P(x, y, 0);
      const a = clamp(0.28 - z / 700, 0.06, 0.28);
      ctx.beginPath();
      ctx.ellipse(px, py, r, r * 0.5, 0, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,0,0,${a})`; ctx.fill();
    };

    const frame = (now) => {
      const sim = simRef.current;
      const dt = Math.min(50, now - last); last = now;
      sim.time += dt;

      /* --- advance motion --- */
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
        // idle breathing
        const b = Math.sin(sim.time / 1400) * 3;
        sim.tip = { x: HOME.x, y: HOME.y, z: HOME.z + b };
      }
      if (sim.scanT >= 0) sim.scanT = Math.min(1, sim.scanT + dt / stageMs);

      /* --- render --- */
      ctx.clearRect(0, 0, W, H);

      // table
      const TB = [P(-30, -150, 0), P(210, -150, 0), P(210, 140, 0), P(-30, 140, 0)];
      ctx.beginPath(); ctx.moveTo(TB[0][0], TB[0][1]);
      TB.slice(1).forEach((p) => ctx.lineTo(p[0], p[1])); ctx.closePath();
      ctx.fillStyle = "#111722"; ctx.fill();
      ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.stroke();
      // grid
      ctx.strokeStyle = "rgba(90,110,140,0.12)"; ctx.lineWidth = 1;
      for (let gx = -30; gx <= 210; gx += 40) {
        const a = P(gx, -150, 0), b2 = P(gx, 140, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b2[0], b2[1]); ctx.stroke();
      }
      for (let gy = -150; gy <= 140; gy += 40) {
        const a = P(-30, gy, 0), b2 = P(210, gy, 0);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b2[0], b2[1]); ctx.stroke();
      }

      // shelf
      drawShadow(SHELF.x, SHELF.y, 0, 30);
      // shelf body as a box
      (function () {
        const { x, y, w, d, h } = SHELF;
        const hw = w / 2, hd = d / 2;
        const top = [P(x - hw, y - hd, h), P(x + hw, y - hd, h), P(x + hw, y + hd, h), P(x - hw, y + hd, h)];
        const fx = [P(x + hw, y - hd, h), P(x + hw, y + hd, h), P(x + hw, y + hd, 0), P(x + hw, y - hd, 0)];
        const fy = [P(x - hw, y + hd, h), P(x + hw, y + hd, h), P(x + hw, y + hd, 0), P(x - hw, y + hd, 0)];
        const poly = (pts, fill) => {
          ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
          pts.slice(1).forEach((p) => ctx.lineTo(p[0], p[1])); ctx.closePath();
          ctx.fillStyle = fill; ctx.fill();
          ctx.strokeStyle = "rgba(0,0,0,0.4)"; ctx.stroke();
        };
        poly(fy, "#1B222E"); poly(fx, "#232C3B"); poly(top, "#2E3A4D");
        const [lx, ly] = P(x, y, h);
        ctx.fillStyle = C.faint;
        ctx.font = "9px ui-monospace, monospace";
        ctx.textAlign = "center";
        ctx.fillText("SHELF", lx, ly + 4);
      })();

      // place marker
      if (sim.placeMarker) {
        const m = sim.placeMarker;
        const [mx, my] = P(m.x, m.y, m.z);
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.ellipse(mx, my, 16, 8, 0, 0, Math.PI * 2);
        ctx.strokeStyle = C.amber; ctx.lineWidth = 1.4; ctx.stroke();
        ctx.setLineDash([]);
      }

      // blocks (depth-sorted), skipping held (drawn with arm)
      const sorted = [...sim.blocks].sort((a, b) => a.x + a.y - (b.x + b.y));
      for (const b of sorted) {
        if (b.id !== sim.held) {
          if (b.z > 1) drawShadow(b.x, b.y, b.z, 14);
          drawCube(b.x, b.y, b.z, S, BLOCK_HEX[b.color]);
        }
      }

      /* --- arm --- */
      const j = solveIK(sim.tip);
      const fk = forwardK(j);
      drawShadow(fk.tip.x, fk.tip.y, fk.tip.z, 10);
      if (sim.held) {
        const b = sim.blocks.find((k) => k.id === sim.held);
        drawShadow(b.x, b.y, b.z, 14);
        drawCube(b.x, b.y, b.z, S, BLOCK_HEX[b.color]);
      }

      // base
      const [b0x, b0y] = P(0, 0, 0);
      ctx.beginPath(); ctx.ellipse(b0x, b0y, 26, 13, 0, 0, Math.PI * 2);
      ctx.fillStyle = "#1A2130"; ctx.fill(); ctx.strokeStyle = C.line; ctx.stroke();
      const [s0x, s0y] = P(0, 0, BASE_H);
      ctx.beginPath();
      ctx.moveTo(b0x - 11, b0y); ctx.lineTo(s0x - 11, s0y);
      ctx.lineTo(s0x + 11, s0y); ctx.lineTo(b0x + 11, b0y); ctx.closePath();
      ctx.fillStyle = "#222B3C"; ctx.fill(); ctx.strokeStyle = "#2E3A50"; ctx.stroke();

      const seg2 = (a, b, w, color) => {
        const pa = P(a.x, a.y, a.z), pb = P(b.x, b.y, b.z);
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]);
        ctx.lineCap = "round"; ctx.lineWidth = w; ctx.strokeStyle = color; ctx.stroke();
      };
      seg2(fk.sh, fk.el, 13, "#39465E");
      seg2(fk.sh, fk.el, 9, "#4C5C7A");
      seg2(fk.el, fk.wr, 11, "#39465E");
      seg2(fk.el, fk.wr, 7, "#4C5C7A");

      // joints
      const jointDot = (p, r) => {
        const [px, py] = P(p.x, p.y, p.z);
        ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2);
        ctx.fillStyle = "#141A26"; ctx.fill();
        ctx.strokeStyle = sim.stage === 3 ? C.amber : "#5A6A88";
        ctx.lineWidth = 1.6; ctx.stroke();
      };
      jointDot(fk.sh, 6); jointDot(fk.el, 5); jointDot(fk.wr, 4.5);

      // gripper: two fingers perpendicular to yaw
      const nx = -Math.sin(j.yaw), ny = Math.cos(j.yaw);
      const sep = 5 + sim.grip * 9;
      const f1t = { x: fk.wr.x + nx * sep, y: fk.wr.y + ny * sep, z: fk.wr.z - 6 };
      const f1b = { x: fk.tip.x + nx * sep, y: fk.tip.y + ny * sep, z: fk.tip.z };
      const f2t = { x: fk.wr.x - nx * sep, y: fk.wr.y - ny * sep, z: fk.wr.z - 6 };
      const f2b = { x: fk.tip.x - nx * sep, y: fk.tip.y - ny * sep, z: fk.tip.z };
      seg2(f1t, f2t, 5, "#39465E");
      seg2(f1t, f1b, 4, "#7487A8");
      seg2(f2t, f2b, 4, "#7487A8");

      /* --- vision overlay --- */
      if (sim.stage === 0) {
        // sweep line
        if (sim.scanT >= 0) {
          const sy2 = 90 + sim.scanT * 300;
          const grad = ctx.createLinearGradient(0, sy2 - 26, 0, sy2);
          grad.addColorStop(0, "rgba(91,200,232,0)");
          grad.addColorStop(1, "rgba(91,200,232,0.16)");
          ctx.fillStyle = grad; ctx.fillRect(40, sy2 - 26, W - 80, 26);
          ctx.strokeStyle = "rgba(91,200,232,0.6)"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(40, sy2); ctx.lineTo(W - 40, sy2); ctx.stroke();
        }
        // detection boxes
        const targets = [...sim.blocks.map((b) => ({ x: b.x, y: b.y, z: b.z, id: b.id, s: S })),
          { x: SHELF.x, y: SHELF.y, z: 0, id: "zone_01", s: SHELF.w, tall: SHELF.h }];
        targets.slice(0, sim.detShown).forEach((b) => {
          const h = b.s / 2;
          const zTop = b.z + (b.tall || S);
          const corners = [
            P(b.x - h, b.y - h, b.z), P(b.x + h, b.y - h, b.z), P(b.x + h, b.y + h, b.z), P(b.x - h, b.y + h, b.z),
            P(b.x - h, b.y - h, zTop), P(b.x + h, b.y - h, zTop), P(b.x + h, b.y + h, zTop), P(b.x - h, b.y + h, zTop),
          ];
          const xs = corners.map((c) => c[0]), ys = corners.map((c) => c[1]);
          const x0 = Math.min(...xs) - 4, x1 = Math.max(...xs) + 4;
          const y0 = Math.min(...ys) - 4, y1 = Math.max(...ys) + 4;
          ctx.strokeStyle = C.cyan; ctx.lineWidth = 1.2;
          ctx.setLineDash([3, 3]);
          ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
          ctx.setLineDash([]);
          ctx.fillStyle = C.cyan;
          ctx.font = "9px ui-monospace, monospace";
          ctx.textAlign = "left";
          ctx.fillText(b.id, x0, y0 - 4);
        });
      }

      /* --- telemetry --- */
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillStyle = C.faint;
      const tele = `J0 yaw ${deg(j.yaw).toFixed(1)}°   J1 ${deg(j.shoulder).toFixed(1)}°   J2 ${deg(j.elbowRel).toFixed(1)}°   grip ${(sim.grip * 100).toFixed(0)}%   tip [${(sim.tip.x / 1000).toFixed(3)}, ${(sim.tip.y / 1000).toFixed(3)}, ${(sim.tip.z / 1000).toFixed(3)}] m`;
      ctx.fillText(tele, 14, H - 12);
      ctx.textAlign = "right";
      ctx.fillStyle = sim.running ? C.amber : C.faint;
      ctx.fillText(sim.running ? "● EXECUTING" : "○ IDLE", W - 14, H - 12);

      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
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
    // action stream
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
                {/* rail */}
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
                {/* card */}
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

        {/* scene column */}
        <section aria-label="Simulated scene" style={{
          background: C.panel, border: `1px solid ${C.line}`, borderRadius: 10,
          padding: 10, display: "flex", flexDirection: "column", minWidth: 0,
        }}>
          <canvas
            ref={canvasRef}
            style={{ width: "100%", maxWidth: 720, margin: "0 auto", aspectRatio: "620 / 470", display: "block" }}
            role="img"
            aria-label="Isometric tabletop scene with a 6-DOF robot arm, colored blocks, and a shelf"
          />
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

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace";

function Hint({ children }) {
  return <div style={{ fontSize: 11, color: "#5B6779", fontFamily: MONO }}>{children}</div>;
}
