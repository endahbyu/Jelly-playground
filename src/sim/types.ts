export interface ShapeDef {
  alpha: number;
  Ri: number;
  rho: number;
  T: number;
  bevel: number;
  skin: number;
  pale: number;
  Ro: number;
}

export interface SimMesh {
  rest: Float32Array;
  tets: Uint32Array;
  region: Uint8Array;
  comp?: Uint16Array;
  nComp?: number;
  n2?: number;
  layers?: number;
}

export interface RangeDef {
  first: number;
  count: number;
}

export interface RenderMesh {
  rest: Float32Array;
  restNormal?: Float32Array;
  mat: Float32Array;
  index: Uint32Array;
  vComp?: Uint16Array;
  body: RangeDef;
  seeds: RangeDef;
  bubbles: RangeDef;
  seedCount?: number;
}

export interface Embedding {
  tetOf: Uint32Array;
  w: Float32Array;
}

export interface PieceCache {
  sim: SimMesh;
  ren: {
    rest: Float32Array;
    mat: Float32Array;
    body: number[];
    seeds: number[];
    bubbles: number[];
  };
  emb: Embedding;
}

export interface Piece {
  I: [number, number][];
  cache: PieceCache | null;
}

export interface WorldData {
  sim: SimMesh;
  render: RenderMesh;
  skinIdx: Uint32Array;
  skinW: Float32Array;
}

export interface KnifeMesh {
  rest: Float32Array;
  nrm: Float32Array;
  mat: Float32Array;
  index: Uint32Array;
}

export interface PaletteDef {
  name: string;
  flesh: [number, number, number];
  fleshDeep: [number, number, number];
  pale: [number, number, number];
  skin: [number, number, number];
  stripe: [number, number, number];
  seed: [number, number, number];
  shadowTint: [number, number, number];
  ui: {
    flesh: [number, number, number];
  };
}

export interface BladeState {
  holdY: number;
  mode: 'press' | 'split' | null;
  p: [number, number, number];
  d: [number, number, number];
  n: [number, number, number];
  a0: number;
  a1: number;
  top: number;
  halfGap: number;
  side: Int8Array | null;
}

export interface CutPlan {
  miss?: string;
  next?: Piece[];
  sep?: { piece: number; dir: [number, number, number] }[];
  world?: WorldData;
  sim?: any;
  M?: [number, number, number];
  yTop?: number;
  yaw?: number;
}
