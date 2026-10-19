import { SHAPE, embed } from './geometry';
import { initialPiece, buildWorld, splitPiece, lineCrossesPiece, ensureCache, buildPieceSim, buildPieceRender } from './pieces';
import { SoftBody } from './physics';
import { Renderer, M4 } from './renderer';
import { WebGLRenderer } from './glRenderer';
import { buildKnifeMesh, KNIFE } from './knife';
import { PaletteDef, Piece, BladeState, CutPlan, WorldData } from './types';

export const srgbToLin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
export const linToSrgb = (c: number) => (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
export const hex = (rgb: [number, number, number] | number[]) =>
  '#' + rgb.map(c => Math.round(Math.min(1, Math.max(0, linToSrgb(c))) * 255).toString(16).padStart(2, '0')).join('');

export const PALETTES: Record<string, PaletteDef> = {
  crimson: {
    name: 'Crimson',
    flesh: [0.93, 0.07, 0.11], fleshDeep: [0.42, 0.018, 0.035],
    pale: [0.8, 0.86, 0.62], skin: [0.03, 0.13, 0.035], stripe: [0.005, 0.034, 0.01],
    seed: [0.01, 0.0065, 0.005], shadowTint: [0.6, 0.47, 0.46],
    ui: { flesh: [0.62, 0.03, 0.06] },
  },
  golden: {
    name: 'Golden',
    flesh: [0.97, 0.52, 0.05], fleshDeep: [0.6, 0.24, 0.012],
    pale: [0.84, 0.88, 0.64], skin: [0.04, 0.15, 0.03], stripe: [0.007, 0.04, 0.01],
    seed: [0.012, 0.008, 0.005], shadowTint: [0.64, 0.55, 0.44],
    ui: { flesh: [0.72, 0.3, 0.02] },
  },
  rose: {
    name: 'Rosé',
    flesh: [0.95, 0.2, 0.3], fleshDeep: [0.6, 0.08, 0.15],
    pale: [0.84, 0.9, 0.72], skin: [0.06, 0.2, 0.1], stripe: [0.012, 0.06, 0.03],
    seed: [0.014, 0.009, 0.007], shadowTint: [0.64, 0.5, 0.52],
    ui: { flesh: [0.72, 0.16, 0.24] },
  },
};

export const BG_SRGB = [0.886, 0.875, 0.855]; // #E2DFDA
export const BG_LIN = BG_SRGB.map(srgbToLin);
export const BG_RENDER = BG_LIN.map(c => c + 0.04);

export const SCALE_CM = 3.5;
export const DENSITY = 1.3;
export const MAX_PIECES = 14;

export interface SimCallbacks {
  onStatusChange?: (text: string, state: 'live' | 'paused' | 'off') => void;
  onFallback?: (title: string, detail: string) => void;
  onToast?: (message: string) => void;
  onStrokeChange?: (x1: number, y1: number, x2: number, y2: number, on: boolean, fading: boolean) => void;
  onCursorChange?: (cursor: 'grab' | 'grabbing' | 'orbit' | 'knife' | '') => void;
  onStatsChange?: (stats: {
    particles: number;
    tets: number;
    mass: string;
    volume: string;
    kinetic: string;
    pieces: number;
    fps: number;
  }) => void;
}

export class SimulationController {
  canvas: HTMLCanvasElement;
  callbacks: SimCallbacks;

  pieces: Piece[] = [];
  world!: WorldData;
  sim!: SoftBody;
  renderer!: Renderer | WebGLRenderer;
  rendererMode = 'WEBGPU · LIVE';
  dyn!: Float32Array;

  currentPalette = 'crimson';
  state = {
    paused: false,
    slow: false,
    showMesh: false,
    tool: 'hand' as 'hand' | 'knife',
    az: 0.62,
    el: 0.6,
    zoom: 1,
    time: 0,
  };

  cam: {
    viewProj: Float32Array;
    view: Float32Array;
    invViewProj: Float32Array;
    pos: number[];
    fwd: number[];
  } = {
    viewProj: new Float32Array(16),
    view: new Float32Array(16),
    invViewProj: new Float32Array(16),
    pos: [0, 0, 0],
    fwd: [0, 0, -1],
  };
  target = [0, 0.24, 1.0];
  home = [0, 0.24, 1.0];
  FOV = (27 * Math.PI) / 180;

  keyDir = (() => {
    const d = [-0.45, 0.86, -0.36];
    const l = Math.hypot(...d);
    return d.map(v => v / l);
  })();
  light: {
    dir: number[];
    intensity: number;
    lightVP: Float32Array;
    topVP: Float32Array;
  } = {
    dir: this.keyDir,
    intensity: 2.3,
    lightVP: new Float32Array(16),
    topVP: new Float32Array(16),
  };

  knifeMesh = buildKnifeMesh();
  HOVER_H = 0.42;
  kp = { P: [0, SHAPE.T + this.HOVER_H + 1.4, 0.9], yaw: -0.62, roll: 0.16, pitch: 0.05 };
  blade: BladeState = {
    holdY: 0,
    mode: null,
    p: [0, 0, 0],
    d: [1, 0, 0],
    n: [0, 0, 1],
    a0: -KNIFE.xr,
    a1: KNIFE.Lb - KNIFE.xr,
    top: KNIFE.H * 0.95,
    halfGap: 0.03,
    side: null,
  };
  knifeAnim: {
    t: number;
    plan: CutPlan;
    from: { P: number[]; yaw: number; roll: number; pitch: number };
    committed: boolean;
  } | null = null;
  hoverSeen = false;
  hoverX = 0;
  hoverY = 0;
  hoverDirty = false;

  drag = {
    mode: null as 'grab' | 'knife' | 'orbit' | null,
    id: -1,
    plane: null as { p: [number, number, number]; n: number[] } | null,
    twist: 0,
    twist0: 0,
    second: null as { id: number; x: number; y: number } | null,
    lastX: 0,
    lastY: 0,
  };
  knifeStroke = { ax: 0, ay: 0, bx: 0, by: 0 };
  lastPickTri = -1;

  running = false;
  animId: number | null = null;
  recoveries = 0;
  reducedMotion = false;

  resizeObserver?: ResizeObserver;
  wholePiece!: Piece;

  constructor(canvas: HTMLCanvasElement, callbacks: SimCallbacks = {}) {
    this.canvas = canvas;
    this.callbacks = callbacks;
    this.reducedMotion = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  async start() {
    this.callbacks.onStatusChange?.('3D · STARTING', 'live');

    this.pieces = [initialPiece()];
    this.wholePiece = this.pieces[0];
    this.world = buildWorld(this.pieces);
    this.sim = new SoftBody(this.world.sim, {
      firmness: 0.4,
      damping: this.reducedMotion ? 0.65 : 0.45,
    });

    let rendererInstance: Renderer | WebGLRenderer | null = null;
    let modeText = 'WEBGL2 · LIVE';

    if (typeof navigator !== 'undefined' && (navigator as any).gpu) {
      try {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter) {
          const device = await adapter.requestDevice();
          const r = new Renderer(this.canvas, device, this.world.render, this.sim.edges, this.sim.n);
          await r.checkShaders();
          this.wireRenderer(r);
          rendererInstance = r;
          modeText = 'WEBGPU · LIVE';
        }
      } catch (err: any) {
        console.warn('WebGPU not available or failed to initialize, switching to WebGL2 fallback:', err);
      }
    }

    if (!rendererInstance) {
      try {
        const glR = new WebGLRenderer(this.canvas, this.world.render, this.sim.edges, this.sim.n);
        rendererInstance = glR;
        modeText = 'WEBGL2 · LIVE';
      } catch (glErr: any) {
        console.error('Neither WebGPU nor WebGL2 could be initialized:', glErr);
        this.callbacks.onFallback?.(
          'Renderer offline',
          'Neither WebGPU nor WebGL2 could be initialized on this browser. Please verify graphics acceleration is enabled in your browser.'
        );
        this.callbacks.onStatusChange?.('OFFLINE', 'off');
        return;
      }
    }

    this.renderer = rendererInstance;
    this.rendererMode = modeText;
    this.renderer.setKnife(this.knifeMesh);
    this.dyn = this.renderer.dyn;
    this.sim.blade = this.blade;

    this.resize();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);

    document.body.classList.add('ready');
    document.getElementById('root')?.classList.add('ready');

    this.bindEvents();
    this.running = true;

    // Expose window.__melon
    (window as any).__melon = {
      sim: this.sim,
      state: this.state,
      cam: this.cam,
      pick: (ray: any) => this.pick(ray),
      rayFrom: (x: number, y: number) => this.rayFrom(x, y),
      renderer: this.renderer,
      performCut: (ax: number, ay: number, bx: number, by: number) => this.performCut(ax, ay, bx, by),
      planCut: (ax: number, ay: number, bx: number, by: number) => this.planCut(ax, ay, bx, by),
      startCut: (plan: CutPlan) => this.startCut(plan),
      kp: this.kp,
      blade: this.blade,
      get knifeAnim() {
        return (this as any).knifeAnim;
      },
      setHover: (x: number, y: number) => {
        this.hoverX = x;
        this.hoverY = y;
        this.hoverSeen = true;
      },
      advance: (sec: number) => {
        for (let t = 0; t < sec - 1e-6; t += 1 / 60) {
          this.updateKnife(1 / 60, 1 / 60);
          this.sim.step();
        }
      },
      compStats: () => {
        const c = this.sim.nComp, out: number[][] = [];
        for (let k = 0; k < c; k++) {
          let n = 0, x = 0, y = 0, z = 0, v = 0;
          for (let i = 0; i < this.sim.n; i++) {
            if (this.sim.comp[i] === k) {
              n++;
              x += this.sim.x[3 * i];
              y += this.sim.x[3 * i + 1];
              z += this.sim.x[3 * i + 2];
              v = Math.max(v, Math.hypot(this.sim.v[3 * i], this.sim.v[3 * i + 1], this.sim.v[3 * i + 2]));
            }
          }
          out.push([x / n, y / n, z / n, v].map(q => +q.toFixed(3)));
        }
        return out;
      },
      get pieces() {
        return this.pieces.length;
      },
      get fps() {
        return (this as any)._fps || 60;
      },
    };

    this.sim.reset(this.reducedMotion ? 0 : 0.35);
    this.runLoop();
  }

  wireRenderer(r: Renderer) {
    r.onError = e => console.error('[webgpu]', (e as any)?.message);
    r.onLost = async info => {
      if (info.reason === 'destroyed') return;
      if (this.recoveries++ < 2) {
        try {
          const fresh = await Renderer.create(this.canvas, this.world.render, this.sim.edges, this.sim.n);
          fresh.dyn = this.dyn;
          fresh.setKnife(this.knifeMesh);
          this.renderer = fresh;
          this.wireRenderer(fresh);
          this.resize();
          return;
        } catch (e) {
          console.warn(e);
        }
      }
      try {
        const glR = new WebGLRenderer(this.canvas, this.world.render, this.sim.edges, this.sim.n);
        glR.dyn = this.dyn;
        glR.setKnife(this.knifeMesh);
        this.renderer = glR;
        this.rendererMode = 'WEBGL2 · LIVE';
        this.resize();
        return;
      } catch (e) {
        console.error(e);
      }
      this.callbacks.onFallback?.(
        'The GPU device was lost.',
        ((info as any)?.message || '') + ' Reload the page to restart the study.'
      );
      this.callbacks.onStatusChange?.('OFFLINE', 'off');
    };
  }

  resize() {
    if (!this.renderer) return;
    const r = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.resize(r.width * dpr, r.height * dpr);
  }

  setPalette(key: string) {
    if (PALETTES[key]) {
      this.currentPalette = key;
      document.documentElement.style.setProperty('--accent', hex(PALETTES[key].ui.flesh));
    }
  }

  setTool(t: 'hand' | 'knife') {
    if (t === 'knife' && this.state.tool !== 'knife' && !this.knifeAnim) {
      const h = this.hoverPose();
      this.kp.P = [h.P[0], h.P[1] + 1.6, h.P[2]];
      this.kp.yaw = h.yaw;
      this.kp.roll = h.roll;
      this.kp.pitch = h.pitch;
    }
    this.state.tool = t;
    this.callbacks.onCursorChange?.(t === 'knife' ? 'knife' : '');
  }

  setPaused(paused: boolean) {
    this.state.paused = paused;
    if (paused && this.drag.mode === 'grab') {
      this.sim.endGrab();
      this.drag.mode = null;
    }
    const liveText = this.rendererMode;
    const pausedText = this.rendererMode.replace('LIVE', 'PAUSED');
    this.callbacks.onStatusChange?.(paused ? pausedText : liveText, paused ? 'paused' : 'live');
  }

  setSlow(slow: boolean) {
    this.state.slow = slow;
  }

  setShowMesh(show: boolean) {
    this.state.showMesh = show;
  }

  setFirmness(val: number) {
    this.sim.firmness = val;
  }

  setDamping(val: number) {
    this.sim.damping = val;
  }

  nudge() {
    if (!this.state.paused) {
      this.sim.nudge(this.reducedMotion ? 0.45 : 1);
    }
  }

  reset() {
    this.knifeAnim = null;
    this.blade.mode = null;
    if (this.pieces.length > 1) {
      this.installWorld([this.wholePiece]);
    }
    this.sim.reset(this.reducedMotion ? 0 : 0.35);
  }

  // Camera & framing
  safeRect(W: number, H: number) {
    const cr = this.canvas.getBoundingClientRect();
    if (document.body.classList.contains('stacked')) return { x: 0, y: 0, w: cr.width, h: cr.height };
    const panel = document.querySelector('.panel')?.getBoundingClientRect();
    const head = document.querySelector('.masthead')?.getBoundingClientRect();
    const read = document.querySelector('.readouts')?.getBoundingClientRect();
    const left = head ? Math.min(head.right - cr.left, cr.width * 0.24) * 0.55 : 40;
    const right = panel ? panel.left - cr.left - 12 : cr.width - 280;
    const top = head ? Math.min(head.bottom - cr.top, cr.height * 0.34) * 0.5 : 40;
    const bottom = read ? read.top - cr.top + 24 : cr.height - 100;
    return { x: left, y: top, w: Math.max(200, right - left), h: Math.max(200, bottom - top) };
  }

  followSlice(dt: number) {
    if (this.drag.mode === 'grab') return;
    const c = this.sim.centroid();
    const k = 1 - Math.exp(-dt * 1.4);
    const gx = this.home[0] + (c[0] - 0.0) * 0.7;
    const gz = this.home[2] + (c[2] - 1.01) * 0.7;
    this.target[0] += (gx - this.target[0]) * k;
    this.target[2] += (gz - this.target[2]) * k;
  }

  updateCamera() {
    const W = this.renderer.w, H = this.renderer.h;
    const cr = this.canvas.getBoundingClientRect();
    const r = this.safeRect(W, H);
    const aspect = W / H;
    const hf = r.h / cr.height, wf = r.w / cr.width;
    const Rh = 1.32, Rv = 1.0;
    const t2 = Math.tan(this.FOV / 2);
    const dist = Math.max(Rv / (t2 * 0.9 * hf), Rh / (t2 * 0.92 * aspect * wf)) * this.state.zoom;
    const ce = Math.cos(this.state.el);
    const eye = [
      this.target[0] + dist * Math.sin(this.state.az) * ce,
      this.target[1] + dist * Math.sin(this.state.el),
      this.target[2] + dist * Math.cos(this.state.az) * ce,
    ];
    const view = M4.lookAt(eye, this.target, [0, 1, 0]);
    const proj = M4.perspective(this.FOV, aspect, 0.1, 60);
    const cx = (r.x + r.w / 2) / cr.width, cy = (r.y + r.h / 2) / cr.height;
    proj[8] -= (cx * 2 - 1);
    proj[9] -= -(cy * 2 - 1);
    this.cam.view = view;
    this.cam.viewProj = M4.mul(proj, view);
    this.cam.invViewProj = M4.invert(this.cam.viewProj);
    this.cam.pos = eye;
    const f = [this.target[0] - eye[0], this.target[1] - eye[1], this.target[2] - eye[2]];
    const fl = Math.hypot(...f);
    this.cam.fwd = f.map(v => v / fl);
  }

  updateLight() {
    const c = this.sim.centroid();
    const at = [c[0], 0.3, c[2]];
    this.light.lightVP = M4.mul(
      M4.ortho(-2.4, 2.4, -2.4, 2.4, 0.1, 14),
      M4.lookAt(
        [at[0] + this.keyDir[0] * 7, at[1] + this.keyDir[1] * 7, at[2] + this.keyDir[2] * 7],
        at,
        [0, 0, 1]
      )
    );
    this.light.topVP = M4.mul(
      M4.ortho(-2.6, 2.6, -2.6, 2.6, 0, 5),
      M4.lookAt([at[0], -1, at[2]], [at[0], 1, at[2]], [0, 0, -1])
    );
  }

  skin() {
    const x = this.sim.x;
    const nV = this.world.render.rest.length / 3;
    const skinIdx = this.world.skinIdx, skinW = this.world.skinW;
    const index = this.world.render.index;
    const dyn = this.dyn;

    for (let v = 0; v < nV; v++) {
      const i4 = 4 * v, o = 6 * v;
      const a = 3 * skinIdx[i4], b = 3 * skinIdx[i4 + 1], c = 3 * skinIdx[i4 + 2], d = 3 * skinIdx[i4 + 3];
      const w0 = skinW[i4], w1 = skinW[i4 + 1], w2 = skinW[i4 + 2], w3 = skinW[i4 + 3];
      dyn[o] = x[a] * w0 + x[b] * w1 + x[c] * w2 + x[d] * w3;
      dyn[o + 1] = x[a + 1] * w0 + x[b + 1] * w1 + x[c + 1] * w2 + x[d + 1] * w3;
      dyn[o + 2] = x[a + 2] * w0 + x[b + 2] * w1 + x[c + 2] * w2 + x[d + 2] * w3;
      dyn[o + 3] = 0; dyn[o + 4] = 0; dyn[o + 5] = 0;
    }
    for (let t = 0; t < index.length; t += 3) {
      const a = 6 * index[t], b = 6 * index[t + 1], c = 6 * index[t + 2];
      const e1x = dyn[b] - dyn[a], e1y = dyn[b + 1] - dyn[a + 1], e1z = dyn[b + 2] - dyn[a + 2];
      const e2x = dyn[c] - dyn[a], e2y = dyn[c + 1] - dyn[a + 1], e2z = dyn[c + 2] - dyn[a + 2];
      const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      dyn[a + 3] += nx; dyn[a + 4] += ny; dyn[a + 5] += nz;
      dyn[b + 3] += nx; dyn[b + 4] += ny; dyn[b + 5] += nz;
      dyn[c + 3] += nx; dyn[c + 4] += ny; dyn[c + 5] += nz;
    }
    for (let o = 0; o < dyn.length; o += 6) {
      const l = Math.hypot(dyn[o + 3], dyn[o + 4], dyn[o + 5]) || 1;
      dyn[o + 3] /= l;
      dyn[o + 4] /= l;
      dyn[o + 5] /= l;
    }
  }

  rayFrom(clientX: number, clientY: number) {
    const r = this.canvas.getBoundingClientRect();
    const nx = ((clientX - r.left) / r.width) * 2 - 1, ny = 1 - ((clientY - r.top) / r.height) * 2;
    const m = this.cam.invViewProj;
    const un = (x: number, y: number, z: number) => {
      const w = m[3] * x + m[7] * y + m[11] * z + m[15];
      return [
        (m[0] * x + m[4] * y + m[8] * z + m[12]) / w,
        (m[1] * x + m[5] * y + m[9] * z + m[13]) / w,
        (m[2] * x + m[6] * y + m[10] * z + m[14]) / w,
      ];
    };
    const a = un(nx, ny, 0), b = un(nx, ny, 1);
    const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const l = Math.hypot(...d);
    return { o: a, d: d.map(v => v / l) };
  }

  pick(ray: { o: number[]; d: number[] }): [number, number, number] | null {
    let best = Infinity;
    const { o, d } = ray;
    const bodyTriEnd = this.world.render.body.count;
    const index = this.world.render.index;
    const dyn = this.dyn;

    for (let t = 0; t < bodyTriEnd; t += 3) {
      const a = 6 * index[t], b = 6 * index[t + 1], c = 6 * index[t + 2];
      const e1x = dyn[b] - dyn[a], e1y = dyn[b + 1] - dyn[a + 1], e1z = dyn[b + 2] - dyn[a + 2];
      const e2x = dyn[c] - dyn[a], e2y = dyn[c + 1] - dyn[a + 1], e2z = dyn[c + 2] - dyn[a + 2];
      const px = d[1] * e2z - d[2] * e2y, py = d[2] * e2x - d[0] * e2z, pz = d[0] * e2y - d[1] * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (Math.abs(det) < 1e-12) continue;
      const inv = 1 / det;
      const tx = o[0] - dyn[a], ty = o[1] - dyn[a + 1], tz = o[2] - dyn[a + 2];
      const uu = (tx * px + ty * py + tz * pz) * inv;
      if (uu < 0 || uu > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const vv = (d[0] * qx + d[1] * qy + d[2] * qz) * inv;
      if (vv < 0 || uu + vv > 1) continue;
      const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (tt > 1e-4 && tt < best) {
        best = tt;
        this.lastPickTri = t;
      }
    }
    return best < Infinity ? [o[0] + d[0] * best, o[1] + d[1] * best, o[2] + d[2] * best] : null;
  }

  pickNear(clientX: number, clientY: number, maxPx: number): [number, number, number] | null {
    const r = this.canvas.getBoundingClientRect(), M = this.cam.viewProj, x = this.sim.x;
    let best = -1, bd = maxPx * maxPx;
    for (let i = 0; i < this.sim.n; i++) {
      const X = x[3 * i], Y = x[3 * i + 1], Z = x[3 * i + 2];
      const w = M[3] * X + M[7] * Y + M[11] * Z + M[15];
      const sx = r.left + (((M[0] * X + M[4] * Y + M[8] * Z + M[12]) / w) * 0.5 + 0.5) * r.width;
      const sy = r.top + (0.5 - ((M[1] * X + M[5] * Y + M[9] * Z + M[13]) / w) * 0.5) * r.height;
      const d = (sx - clientX) ** 2 + (sy - clientY) ** 2 - w * 1e-3;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    return best < 0 ? null : [x[3 * best], x[3 * best + 1], x[3 * best + 2]];
  }

  rayPlaneY(ray: { o: number[]; d: number[] }, y: number): [number, number, number] | null {
    if (ray.d[1] > -1e-4) return null;
    const t = (y - ray.o[1]) / ray.d[1];
    return t > 0 ? [ray.o[0] + ray.d[0] * t, y, ray.o[2] + ray.d[2] * t] : null;
  }

  camRightYaw(): number {
    return -this.state.az;
  }

  canonicalYaw(dx: number, dz: number): number {
    let yaw = Math.atan2(dz, dx);
    if (Math.cos(yaw - this.camRightYaw()) < 0) yaw += Math.PI;
    return yaw;
  }

  planCut(ax: number, ay: number, bx: number, by: number): CutPlan {
    if (Math.hypot(bx - ax, by - ay) < 24) return { miss: '' };
    const hit = new Set<number>();
    let yTop = 0, t0 = -1, t1 = -1;
    const vComp = this.world.render.vComp!;
    const index = this.world.render.index;

    for (let k = 0; k <= 48; k++) {
      const t = k / 48;
      const p = this.pick(this.rayFrom(ax + (bx - ax) * t, ay + (by - ay) * t));
      if (p && this.lastPickTri >= 0) {
        hit.add(vComp[index[this.lastPickTri]]);
        yTop = Math.max(yTop, p[1]);
        if (t0 < 0) t0 = t;
        t1 = t;
      }
    }
    if (!hit.size) return { miss: 'Missed — draw the blade across the slice.' };
    if (this.pieces.length + hit.size > MAX_PIECES) {
      return { miss: 'That’s plenty of pieces. Reset to start a fresh slice.' };
    }

    const at = (t: number) => this.rayPlaneY(this.rayFrom(ax + (bx - ax) * t, ay + (by - ay) * t), yTop);
    const A = at(0) || at(t0), B = at(1) || at(t1);
    const A2 = at(t0), B2 = at(t1);
    if (!A || !B || !A2 || !B2) return { miss: 'Missed — draw the blade across the slice.' };

    let dx = B[0] - A[0], dz = B[2] - A[2];
    const dl = Math.hypot(dx, dz);
    if (dl < 0.05) return { miss: '' };
    dx /= dl; dz /= dl;
    const n = [-dz, 0, dx];
    const o: [number, number, number] = [(A2[0] + B2[0]) / 2, yTop, (A2[2] + B2[2]) / 2];
    const next: Piece[] = [], sep: { piece: number; dir: [number, number, number] }[] = [];
    let thin = 0, cut = 0;

    this.pieces.forEach((pc, c) => {
      if (!hit.has(c)) {
        next.push(pc);
        return;
      }
      const F = this.sim.pieceFrame(c), R = F.R;
      const nr = [
        R[0] * n[0] + R[3] * n[1] + R[6] * n[2],
        R[1] * n[0] + R[4] * n[1] + R[7] * n[2],
        R[2] * n[0] + R[5] * n[1] + R[8] * n[2],
      ];
      const d = [o[0] - F.cw[0], o[1] - F.cw[1], o[2] - F.cw[2]];
      const pr = [
        R[0] * d[0] + R[3] * d[1] + R[6] * d[2] + F.cr[0],
        R[1] * d[0] + R[4] * d[1] + R[7] * d[2] + F.cr[1],
        R[2] * d[0] + R[5] * d[1] + R[8] * d[2] + F.cr[2],
      ];
      const L = Math.hypot(nr[0], nr[2]);
      if (L < 0.3) {
        next.push(pc);
        thin++;
        return;
      }
      const cc = (nr[0] * pr[0] + nr[1] * pr[1] + nr[2] * pr[2] - (nr[1] * SHAPE.T) / 2) / L;
      const a = nr[0] / L, b = nr[2] / L;
      if (!lineCrossesPiece(pc.I, a, b, cc)) {
        next.push(pc);
        return;
      }
      const parts = splitPiece(pc.I, a, b, cc);
      if (!parts) {
        next.push(pc);
        thin++;
        return;
      }
      cut++;
      parts.forEach((I, side) => {
        let su = 0, sw = 0;
        for (const p of I) { su += p[0]; sw += p[1]; }
        su /= I.length; sw /= I.length;
        const X = [su - F.cr[0], SHAPE.T / 2 - F.cr[1], sw - F.cr[2]];
        const W = [
          R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + F.cw[0],
          R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + F.cw[1],
          R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + F.cw[2],
        ];
        const sg = Math.sign(n[0] * (W[0] - o[0]) + n[2] * (W[2] - o[2])) || (side ? 1 : -1);
        sep.push({ piece: next.length, dir: [n[0] * sg, 0, n[2] * sg] });
        next.push({ I, cache: null });
      });
    });

    if (!cut) return { miss: thin ? 'Too thin to cut there.' : 'Missed — draw the blade across the slice.' };
    const w = buildWorld(next);
    const ns = new SoftBody(w.sim, { firmness: this.sim.firmness, damping: this.sim.damping });
    return {
      next,
      sep,
      world: w,
      sim: ns,
      M: o,
      yTop,
      yaw: this.canonicalYaw(dx, dz),
    };
  }

  performCut(ax: number, ay: number, bx: number, by: number): boolean {
    const plan = this.planCut(ax, ay, bx, by);
    if (plan.miss !== undefined) {
      if (plan.miss) this.callbacks.onToast?.(plan.miss);
      return false;
    }
    this.installWorld(plan.next!, plan.sep, plan, 0.75, 0.35);
    return true;
  }

  installWorld(
    nextPieces: Piece[],
    sep: { piece: number; dir: [number, number, number] }[] = [],
    pre: CutPlan | null = null,
    sepSpeed = 0.75,
    hop = 0.35
  ) {
    if (this.drag.mode === 'grab') {
      this.sim.endGrab();
      this.drag.mode = null;
    }
    const old = this.sim;
    const w = pre?.world ? pre.world : buildWorld(nextPieces);
    const ns = pre?.sim ? pre.sim : new SoftBody(w.sim, { firmness: old.firmness, damping: old.damping });
    ns.firmness = old.firmness;
    ns.damping = old.damping;

    if (old) {
      const e = (this as any).embedOld(old, ns);
      for (let i = 0; i < ns.n; i++) {
        const t = e.tetOf[i];
        for (let k = 0; k < 3; k++) {
          let xs = 0, vs = 0;
          for (let j = 0; j < 4; j++) {
            const q = old.tets[4 * t + j];
            xs += old.x[3 * q + k] * e.w[4 * i + j];
            vs += old.v[3 * q + k] * e.w[4 * i + j];
          }
          ns.x[3 * i + k] = xs;
          ns.v[3 * i + k] = vs;
        }
        if (ns.x[3 * i + 1] < 0) ns.x[3 * i + 1] = 0;
      }
      for (const sp of sep) {
        for (let i = 0; i < ns.n; i++) {
          if (ns.comp[i] === sp.piece) {
            ns.v[3 * i] += sp.dir[0] * sepSpeed;
            ns.v[3 * i + 1] += Math.abs(sp.dir[1]) * 0.2 * hop + hop;
            ns.v[3 * i + 2] += sp.dir[2] * sepSpeed;
          }
        }
      }
      ns.prev.set(ns.x);
    }

    this.pieces = nextPieces;
    this.world = w;
    this.sim = ns;
    this.sim.blade = this.blade;
    this.blade.side = new Int8Array(this.sim.nComp);
    for (const sp of sep) {
      this.blade.side[sp.piece] = Math.sign(sp.dir[0] * this.blade.n[0] + sp.dir[2] * this.blade.n[2]) || 0;
    }

    this.renderer.setMesh(w.render, ns.edges, ns.n);
    this.dyn = this.renderer.dyn;
    this.hoverDirty = true;

    if ((window as any).__melon) {
      (window as any).__melon.sim = this.sim;
    }
  }

  embedOld(old: SoftBody, ns: SoftBody) {
    return embed(old.rest, old.tets, ns.rest);
  }
  knifeBasis(yaw: number, roll: number, pitch: number): [number[], number[], number[]] {
    const d = [Math.cos(yaw), 0, Math.sin(yaw)], z = [-d[2], 0, d[0]];
    const cr = Math.cos(roll), sr = Math.sin(roll), cp = Math.cos(pitch), sp = Math.sin(pitch);
    const Y1 = [-sr * z[0], cr, -sr * z[2]], Z = [cr * z[0], sr, cr * z[2]];
    const X = [cp * d[0] - sp * Y1[0], cp * d[1] - sp * Y1[1], cp * d[2] - sp * Y1[2]];
    const Y = [sp * d[0] + cp * Y1[0], sp * d[1] + cp * Y1[1], sp * d[2] + cp * Y1[2]];
    return [X, Y, Z];
  }

  poseKnife() {
    const [X, Y, Z] = this.knifeBasis(this.kp.yaw, this.kp.roll, this.kp.pitch);
    const out = this.renderer.knifeDyn, r = this.knifeMesh.rest, nr = this.knifeMesh.nrm, P = this.kp.P, xr = KNIFE.xr;
    for (let v = 0, o = 0; v < r.length; v += 3, o += 6) {
      const lx = r[v] - xr, ly = r[v + 1], lz = r[v + 2];
      out[o] = P[0] + X[0] * lx + Y[0] * ly + Z[0] * lz;
      out[o + 1] = P[1] + X[1] * lx + Y[1] * ly + Z[1] * lz;
      out[o + 2] = P[2] + X[2] * lx + Y[2] * ly + Z[2] * lz;
      const nx = nr[v], ny = nr[v + 1], nz = nr[v + 2];
      out[o + 3] = X[0] * nx + Y[0] * ny + Z[0] * nz;
      out[o + 4] = X[1] * nx + Y[1] * ny + Z[1] * nz;
      out[o + 5] = X[2] * nx + Y[2] * ny + Z[2] * nz;
    }
  }

  hoverPose() {
    let q = this.hoverSeen ? this.rayPlaneY(this.rayFrom(this.hoverX, this.hoverY), SHAPE.T + this.HOVER_H) : null;
    if (q) {
      const dx = q[0] - this.target[0], dz = q[2] - this.target[2], dl = Math.hypot(dx, dz);
      if (dl > 3) {
        q[0] = this.target[0] + (dx / dl) * 3;
        q[2] = this.target[2] + (dz / dl) * 3;
      }
    }
    if (!q) {
      const c = this.sim.centroid();
      q = [c[0] + 0.1, SHAPE.T + this.HOVER_H, c[2] + 0.25];
    }
    return { P: q, yaw: this.camRightYaw(), roll: 0.16, pitch: 0.05 };
  }

  strokePose() {
    const h = SHAPE.T + 0.1;
    const A = this.rayPlaneY(this.rayFrom(this.knifeStroke.ax, this.knifeStroke.ay), h);
    const B = this.rayPlaneY(this.rayFrom(this.knifeStroke.bx, this.knifeStroke.by), h);
    if (!A || !B) return this.hoverPose();
    const dl = Math.hypot(B[0] - A[0], B[2] - A[2]);
    const yaw = dl > 0.08 ? this.canonicalYaw(B[0] - A[0], B[2] - A[2]) : this.kp.yaw;
    return { P: [(A[0] + B[0]) / 2, h, (A[2] + B[2]) / 2], yaw, roll: 0.08, pitch: 0 };
  }

  easeToward(pose: { P: number[]; yaw: number; roll: number; pitch: number }, k: number) {
    this.kp.P = lerp3(this.kp.P, pose.P, k);
    this.kp.yaw += wrapA(pose.yaw - this.kp.yaw) * k;
    this.kp.roll = lerp(this.kp.roll, pose.roll, k);
    this.kp.pitch = lerp(this.kp.pitch, pose.pitch, k);
  }

  startCut(plan: CutPlan) {
    this.knifeAnim = {
      t: 0,
      plan,
      from: { P: this.kp.P.slice(), yaw: this.kp.yaw, roll: this.kp.roll, pitch: this.kp.pitch },
      committed: false,
    };
  }

  runKnifeAnim(dt: number) {
    const KT = { align: 0.16, press: 0.42, through: 0.6, hold: 0.7, lift: 1.08 };
    const a = this.knifeAnim!, pl = a.plan;
    a.t += dt;
    const t = a.t, d = [Math.cos(pl.yaw!), 0, Math.sin(pl.yaw!)], M = pl.M!;
    const at = (s: number, y: number) => [M[0] + d[0] * s, y, M[2] + d[2] * s];
    const P1 = at(-0.12, pl.yTop! + 0.1), P2 = at(0.0, pl.yTop! - 0.13), P3 = at(0.2, 0.004);

    if (t < KT.align) {
      const s = ease(t / KT.align);
      this.kp.P = lerp3(a.from.P, P1, s);
      this.kp.yaw = a.from.yaw + wrapA(pl.yaw! - a.from.yaw) * s;
      this.kp.roll = lerp(a.from.roll, 0, s);
      this.kp.pitch = lerp(a.from.pitch, 0, s);
      this.blade.mode = null;
    } else if (t < KT.press) {
      const s = (t - KT.align) / (KT.press - KT.align);
      this.kp.P = lerp3(P1, P2, s * s);
      this.kp.yaw = pl.yaw!;
      this.kp.roll = 0;
      this.kp.pitch = 0;
      this.blade.mode = 'press';
    } else if (t < KT.hold) {
      if (!a.committed) {
        a.committed = true;
        this.blade.n = [-d[2], 0, d[0]];
        this.blade.holdY = this.kp.P[1];
        this.installWorld(pl.next!, pl.sep, pl, 0.22, 0);
      }
      const s = 1 - Math.pow(1 - Math.min(1, (t - KT.press) / (KT.through - KT.press)), 2);
      this.kp.P = lerp3(P2, P3, s);
      this.blade.halfGap = 0.012 + 0.02 * s;
      this.blade.mode = 'split';
    } else if (t < KT.lift) {
      this.blade.mode = null;
      const s = ease((t - KT.hold) / (KT.lift - KT.hold));
      const h = this.state.tool === 'knife' ? this.hoverPose() : { P: [P3[0], P3[1] + 2.5, P3[2]], yaw: pl.yaw!, roll: 0.3, pitch: 0.05 };
      this.kp.P = lerp3(P3, h.P, s);
      this.kp.yaw = pl.yaw! + wrapA(h.yaw - pl.yaw!) * s;
      this.kp.roll = lerp(0, h.roll, s);
      this.kp.pitch = lerp(0, h.pitch, s);
    } else {
      this.knifeAnim = null;
      this.blade.mode = null;
    }
  }

  updateKnife(dt: number, animDt: number) {
    if (this.state.tool !== 'knife' && !this.knifeAnim) {
      this.renderer.knifeVisible = false;
      this.blade.mode = null;
      return;
    }
    if (this.knifeAnim) this.runKnifeAnim(animDt);
    else this.easeToward(this.drag.mode === 'knife' ? this.strokePose() : this.hoverPose(), 1 - Math.exp(-dt * 16));
    this.blade.p = this.kp.P as [number, number, number];
    this.blade.d = [Math.cos(this.kp.yaw), 0, Math.sin(this.kp.yaw)];
    if (!this.knifeAnim || !this.knifeAnim.committed) this.blade.n = [-this.blade.d[2], 0, this.blade.d[0]];
    this.poseKnife();
    this.renderer.knifeVisible = true;
  }

  // Pointer interactions
  dragTarget(clientX: number, clientY: number): [number, number, number] | null {
    const ray = this.rayFrom(clientX, clientY);
    const { p, n } = this.drag.plane!;
    const den = ray.d[0] * n[0] + ray.d[1] * n[1] + ray.d[2] * n[2];
    if (Math.abs(den) < 1e-5) return null;
    const t = ((p[0] - ray.o[0]) * n[0] + (p[1] - ray.o[1]) * n[1] + (p[2] - ray.o[2]) * n[2]) / den;
    const q: [number, number, number] = [ray.o[0] + ray.d[0] * t, ray.o[1] + ray.d[1] * t, ray.o[2] + ray.d[2] * t];
    const dx = q[0] - this.target[0], dz = q[2] - this.target[2], lim = 3.4, dl = Math.hypot(dx, dz);
    if (dl > lim) {
      q[0] = this.target[0] + (dx / dl) * lim;
      q[2] = this.target[2] + (dz / dl) * lim;
    }
    q[1] = Math.min(Math.max(q[1], 0.02), 3.2);
    return q;
  }

  applyGrab(x: number, y: number) {
    const q = this.dragTarget(x, y);
    if (q) this.sim.moveGrab(q, rotMat(this.cam.fwd, this.drag.twist));
  }

  bindEvents() {
    const canvas = this.canvas;

    canvas.addEventListener('pointerdown', (e: PointerEvent) => {
      if (this.drag.mode === 'grab' && e.pointerId !== this.drag.id && e.pointerType === 'touch') {
        this.drag.second = { id: e.pointerId, x: e.clientX, y: e.clientY };
        this.drag.twist0 = this.drag.twist - Math.atan2(e.clientY - this.drag.lastY, e.clientX - this.drag.lastX);
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
        return;
      }
      if (this.drag.mode) return;
      if (e.button !== undefined && e.button > 0 && e.pointerType === 'mouse') return;

      if (this.state.tool === 'knife') {
        if (this.state.paused || this.knifeAnim) {
          e.preventDefault();
          return;
        }
        this.drag.mode = 'knife';
        this.drag.id = e.pointerId;
        if (e.pointerType !== 'mouse') {
          this.hoverX = e.clientX;
          this.hoverY = e.clientY;
          this.hoverSeen = true;
        }
        this.knifeStroke.ax = this.knifeStroke.bx = e.clientX;
        this.knifeStroke.ay = this.knifeStroke.by = e.clientY;
        try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
        this.callbacks.onStrokeChange?.(this.knifeStroke.ax, this.knifeStroke.ay, this.knifeStroke.bx, this.knifeStroke.by, true, false);
        e.preventDefault();
        return;
      }

      const hit = !this.state.paused
        ? (this.pick(this.rayFrom(e.clientX, e.clientY)) || this.pickNear(e.clientX, e.clientY, e.pointerType === 'touch' ? 34 : 18))
        : null;

      this.drag.id = e.pointerId;
      this.drag.lastX = e.clientX;
      this.drag.lastY = e.clientY;
      try { canvas.setPointerCapture(e.pointerId); } catch (_) {}

      if (hit) {
        this.drag.mode = 'grab';
        this.drag.twist = 0;
        this.drag.plane = { p: hit, n: this.cam.fwd.slice() };
        this.sim.beginGrab(hit, 0.4);
        this.callbacks.onCursorChange?.('grabbing');
      } else {
        this.drag.mode = 'orbit';
        this.callbacks.onCursorChange?.('orbit');
      }
      e.preventDefault();
    });

    canvas.addEventListener('pointermove', (e: PointerEvent) => {
      if (this.drag.second && e.pointerId === this.drag.second.id) {
        this.drag.second.x = e.clientX;
        this.drag.second.y = e.clientY;
        this.drag.twist = this.drag.twist0 + Math.atan2(this.drag.second.y - this.drag.lastY, this.drag.second.x - this.drag.lastX);
        this.drag.twist = Math.max(-1.3, Math.min(1.3, this.drag.twist));
        this.applyGrab(this.drag.lastX, this.drag.lastY);
        return;
      }
      if (e.pointerId !== this.drag.id || !this.drag.mode) {
        this.hoverX = e.clientX;
        this.hoverY = e.clientY;
        this.hoverDirty = true;
        this.hoverSeen = true;
        return;
      }
      if (this.drag.mode === 'knife') {
        this.knifeStroke.bx = e.clientX;
        this.knifeStroke.by = e.clientY;
        this.callbacks.onStrokeChange?.(this.knifeStroke.ax, this.knifeStroke.ay, this.knifeStroke.bx, this.knifeStroke.by, true, false);
        return;
      }
      const dx = e.clientX - this.drag.lastX, dy = e.clientY - this.drag.lastY;
      this.drag.lastX = e.clientX;
      this.drag.lastY = e.clientY;
      if (this.drag.mode === 'grab') {
        this.applyGrab(e.clientX, e.clientY);
      } else {
        this.state.az = Math.max(-0.35, Math.min(1.6, this.state.az - dx * 0.006));
        this.state.el = Math.max(0.12, Math.min(1.25, this.state.el + dy * 0.005));
      }
    });

    const endPointer = (e: PointerEvent) => {
      if (this.drag.second && e.pointerId === this.drag.second.id) {
        this.drag.second = null;
        return;
      }
      if (e.pointerId !== this.drag.id) return;
      if (this.drag.mode === 'knife') {
        if (e.type === 'pointerup') {
          this.knifeStroke.bx = e.clientX;
          this.knifeStroke.by = e.clientY;
          this.callbacks.onStrokeChange?.(this.knifeStroke.ax, this.knifeStroke.ay, this.knifeStroke.bx, this.knifeStroke.by, true, false);
          const plan = this.planCut(this.knifeStroke.ax, this.knifeStroke.ay, this.knifeStroke.bx, this.knifeStroke.by);
          if (plan.miss !== undefined) {
            if (plan.miss) this.callbacks.onToast?.(plan.miss);
          } else {
            this.startCut(plan);
          }
          this.hoverX = e.clientX;
          this.hoverY = e.clientY;
        }
        this.callbacks.onStrokeChange?.(this.knifeStroke.ax, this.knifeStroke.ay, this.knifeStroke.bx, this.knifeStroke.by, false, true);
      }
      if (this.drag.mode === 'grab') {
        this.sim.endGrab();
      }
      this.drag.mode = null;
      this.drag.id = -1;
      this.drag.second = null;
      this.callbacks.onCursorChange?.(this.state.tool === 'knife' ? 'knife' : '');
      this.hoverDirty = true;
    };

    canvas.addEventListener('pointerup', endPointer);
    canvas.addEventListener('pointercancel', endPointer);
    canvas.addEventListener('lostpointercapture', endPointer);

    window.addEventListener(
      'wheel',
      (e: WheelEvent) => {
        if (this.drag.mode === 'grab') {
          e.preventDefault();
          const d = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
          this.drag.twist = Math.max(-1.3, Math.min(1.3, this.drag.twist + d * 0.004));
          this.applyGrab(this.drag.lastX, this.drag.lastY);
        } else if (e.target === canvas) {
          e.preventDefault();
          this.state.zoom = Math.max(0.65, Math.min(1.6, this.state.zoom * Math.exp(e.deltaY * 0.001)));
        }
      },
      { passive: false }
    );

    canvas.addEventListener('dblclick', () => {
      this.state.az = 0.62;
      this.state.el = 0.6;
      this.state.zoom = 1;
    });
  }

  runLoop() {
    let last = performance.now(), fpsStart = last, acc = 0, readoutT = 0, fps = 60, frames = 0, fpsT = 0, first = true;

    const frame = (now: number) => {
      if (!this.running) return;
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;

      if (!this.state.paused) {
        acc += dt * (this.state.slow ? 0.25 : 1);
        let steps = 0;
        while (acc >= this.sim.stepDt && steps < 3) {
          this.sim.step();
          acc -= this.sim.stepDt;
          steps++;
        }
        if (steps === 3) acc = Math.min(acc, this.sim.stepDt);
        this.state.time += dt;
      }

      this.updateKnife(dt, this.state.paused ? 0 : dt * (this.state.slow ? 0.25 : 1));
      this.skin();
      this.followSlice(dt);
      this.updateCamera();
      this.updateLight();

      if (this.hoverDirty && !this.drag.mode) {
        this.hoverDirty = false;
        const cur =
          this.state.tool === 'knife'
            ? 'knife'
            : !this.state.paused && (this.pick(this.rayFrom(this.hoverX, this.hoverY)) || this.pickNear(this.hoverX, this.hoverY, 18))
            ? 'grab'
            : '';
        this.callbacks.onCursorChange?.(cur as any);
      }

      this.renderer.setUniforms(this.cam, this.light, PALETTES[this.currentPalette], {
        time: this.state.time,
        shape: [SHAPE.T, SHAPE.Ro, SHAPE.skin, SHAPE.pale],
        exposure: 1.0,
        meshAlpha: this.state.showMesh ? 0.32 : 0,
        bg: BG_RENDER,
      });

      this.renderer.uploadGeometry(this.sim.x, this.state.showMesh);
      this.renderer.draw(this.state.showMesh);

      if (first) {
        first = false;
        this.callbacks.onStatusChange?.(this.rendererMode, 'live');
        document.body.classList.add('ready');
        document.getElementById('root')?.classList.add('ready');
      }

      readoutT += dt;
      frames++;
      fpsT += dt;
      if (fpsT > 0.5) {
        fps = frames / ((now - fpsStart) / 1000);
        frames = 0;
        fpsT = 0;
        fpsStart = now;
        (this as any)._fps = fps;
      }

      if (readoutT > 0.12) {
        readoutT = 0;
        const massG = this.sim.totalRestVolume * SCALE_CM ** 3 * DENSITY;
        const ke = (this.sim.kineticEnergy() / this.sim.totalMass) * (massG / 1000) * (SCALE_CM / 100) ** 2;
        const uj = ke * 1e6;
        this.callbacks.onStatsChange?.({
          particles: this.sim.n,
          tets: this.sim.nT,
          mass: massG.toFixed(0),
          volume: (this.sim.volumeRatio() * 100).toFixed(1),
          kinetic: uj < 0.05 ? '0.00' : uj < 100 ? uj.toFixed(2) : uj.toFixed(0),
          pieces: this.pieces.length,
          fps: Math.round(fps),
        });
      }

      if (!this.renderer.lost) {
        this.animId = requestAnimationFrame(frame);
      }
    };

    this.animId = requestAnimationFrame(frame);
  }

  destroy() {
    this.running = false;
    if (this.animId) cancelAnimationFrame(this.animId);
    if (this.resizeObserver) this.resizeObserver.disconnect();
  }
}


function wrapA(a: number): number {
  return Math.atan2(Math.sin(a), Math.cos(a));
}
function lerp(a: number, b: number, s: number): number {
  return a + (b - a) * s;
}
function lerp3(a: number[], b: number[], s: number): [number, number, number] {
  return [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s)];
}
function ease(s: number): number {
  s = Math.min(1, Math.max(0, s));
  return s * s * (3 - 2 * s);
}
function rotMat(axis: number[], ang: number): number[] {
  const [x, y, z] = axis, c = Math.cos(ang), s = Math.sin(ang), t = 1 - c;
  return [
    t * x * x + c, t * x * y - s * z, t * x * z + s * y,
    t * x * y + s * z, t * y * y + c, t * y * z - s * x,
    t * x * z - s * y, t * y * z + s * x, t * z * z + c
  ];
}
