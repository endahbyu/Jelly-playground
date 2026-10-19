import { RenderMesh, KnifeMesh, PaletteDef } from './types';

// WebGL2 Renderer providing a high-fidelity fallback when WebGPU is unavailable
export class WebGLRenderer {
  canvas: HTMLCanvasElement;
  gl: WebGL2RenderingContext;
  mesh!: RenderMesh;
  nV!: number;
  lost = false;

  w = 0;
  h = 0;

  dyn!: Float32Array;
  dynBuf!: WebGLBuffer;
  staticBuf!: WebGLBuffer;
  indexBuf!: WebGLBuffer;
  lineIdx!: WebGLBuffer;
  lineCount = 0;
  particleBuf!: WebGLBuffer;

  knifeVisible = false;
  knifeDyn!: Float32Array;
  knifeDynBuf!: WebGLBuffer;
  knifeStatic!: WebGLBuffer;
  knifeIdx!: WebGLBuffer;
  knifeCount = 0;

  // Shaders & programs
  programMesh!: WebGLProgram;
  programFloor!: WebGLProgram;
  programLines!: WebGLProgram;

  uniformsMesh: Record<string, WebGLUniformLocation> = {};
  uniformsFloor: Record<string, WebGLUniformLocation> = {};
  uniformsLines: Record<string, WebGLUniformLocation> = {};

  floorQuadBuf!: WebGLBuffer;

  constructor(canvas: HTMLCanvasElement, mesh: RenderMesh, simEdges: Uint32Array, nParticles: number) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) {
      throw new Error('WebGL2 not supported');
    }
    this.gl = gl;

    gl.getExtension('EXT_color_buffer_float');
    gl.getExtension('OES_texture_float_linear');

    this.initPrograms();
    this.setMesh(mesh, simEdges, nParticles);

    // Floor fullscreen quad
    const quad = new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]);
    this.floorQuadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.floorQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  }

  initPrograms() {
    const gl = this.gl;

    // 1. Floor program
    const vsFloor = `#version 300 es
      in vec2 a_pos;
      out vec2 v_ndc;
      void main() {
        v_ndc = a_pos;
        gl_Position = vec4(a_pos, 0.9999, 1.0);
      }
    `;
    const fsFloor = `#version 300 es
      precision highp float;
      in vec2 v_ndc;
      out vec4 fragColor;

      uniform mat4 u_invViewProj;
      uniform vec3 u_camPos;
      uniform vec3 u_bg;
      uniform vec3 u_centroid;
      uniform vec3 u_shadowTint;

      void main() {
        vec4 a = u_invViewProj * vec4(v_ndc, 0.0, 1.0);
        vec4 b = u_invViewProj * vec4(v_ndc, 1.0, 1.0);
        vec3 ro = a.xyz / a.w;
        vec3 rd = normalize(b.xyz / b.w - ro);

        vec3 col = u_bg;
        // Floor at y = 0
        if (rd.y < -1e-4) {
          float t = (0.0 - ro.y) / rd.y;
          if (t > 0.0) {
            vec3 P = ro + rd * t;
            // Soft shadow under centroid of slice
            float dCenter = length(P.xz - u_centroid.xz);
            float shadow = smoothstep(0.05, 1.35, dCenter);
            float coreShadow = smoothstep(0.0, 0.55, dCenter);
            vec3 shadowCol = mix(u_shadowTint * 0.75, vec3(1.0), shadow * 0.7 + coreShadow * 0.3);

            vec3 fl = u_bg * (0.92 + 0.08 * clamp(1.0 - length(P.xz - vec2(0.0, 0.8)) * 0.12, 0.0, 1.0));
            fl *= shadowCol;
            float fade = smoothstep(6.0, 18.0, t);
            col = mix(fl, u_bg, fade);
          }
        }
        col *= 1.0 - 0.08 * smoothstep(0.0, 0.6, rd.y);
        fragColor = vec4(col, 1.0);
      }
    `;
    this.programFloor = this.createProgram(vsFloor, fsFloor);
    this.uniformsFloor = this.getUniforms(this.programFloor, [
      'u_invViewProj', 'u_camPos', 'u_bg', 'u_centroid', 'u_shadowTint'
    ]);

    // 2. Mesh program (Jelly body, seeds, bubbles, knife)
    const vsMesh = `#version 300 es
      in vec3 a_position;
      in vec3 a_normal;
      in vec3 a_rest;
      in float a_mat;

      uniform mat4 u_viewProj;

      out vec3 v_pos;
      out vec3 v_normal;
      out vec3 v_rest;
      out float v_mat;

      void main() {
        v_pos = a_position;
        v_normal = a_normal;
        v_rest = a_rest;
        v_mat = a_mat;
        gl_Position = u_viewProj * vec4(a_position, 1.0);
      }
    `;
    const fsMesh = `#version 300 es
      precision highp float;

      in vec3 v_pos;
      in vec3 v_normal;
      in vec3 v_rest;
      in float v_mat;

      out vec4 fragColor;

      uniform vec3 u_camPos;
      uniform vec4 u_keyDir;
      uniform vec3 u_flesh;
      uniform vec3 u_fleshDeep;
      uniform vec3 u_pale;
      uniform vec3 u_skin;
      uniform vec3 u_stripe;
      uniform vec3 u_seed;
      uniform vec4 u_shape; // T, Ro, skinDepth, paleDepth
      uniform vec3 u_bg;

      // Studio environment reflection approximation
      vec3 studioReflect(vec3 R, float rough) {
        float y = R.y;
        vec3 col = mix(u_bg * 0.65, u_bg * 1.05, smoothstep(-0.35, 0.25, y));
        // Softbox highlights
        vec3 box1 = normalize(vec3(0.05, 1.0, -0.15));
        vec3 box2 = normalize(vec3(-0.8, 0.38, -0.2));
        vec3 box3 = normalize(vec3(0.85, 0.3, 0.55));
        col += vec3(1.0, 0.98, 0.95) * 2.5 * pow(max(dot(R, box1), 0.0), mix(24.0, 4.0, rough));
        col += vec3(1.0, 0.97, 0.93) * 3.0 * pow(max(dot(R, box2), 0.0), mix(32.0, 6.0, rough));
        col += vec3(0.95, 0.97, 1.0) * 1.0 * pow(max(dot(R, box3), 0.0), mix(16.0, 3.0, rough));
        return col;
      }

      void main() {
        vec3 N = normalize(v_normal);
        vec3 V = normalize(u_camPos - v_pos);
        vec3 L = u_keyDir.xyz;
        float key = u_keyDir.w;
        float NdV = max(dot(N, V), 1e-3);
        float NdL = dot(N, L);

        // 1. Jelly body
        if (v_mat < 0.5) {
          float r = length(v_rest.xz);
          float th = atan(v_rest.x, v_rest.z);
          float depth = u_shape.y - r;

          float skinW = 1.0 - smoothstep(u_shape.z - 0.008, u_shape.z + 0.01, depth);
          float paleW = (1.0 - smoothstep(u_shape.w - 0.035, u_shape.w + 0.035, depth)) * (1.0 - skinW);
          float fleshW = 1.0 - skinW - paleW;

          // Stripe on skin
          float stripeF = sin(th * 34.0);
          float stripe = smoothstep(-0.05, 0.35, stripeF) * smoothstep(0.1, 0.02, depth);
          vec3 skinCol = mix(u_skin, u_stripe, stripe);
          skinCol = mix(skinCol, mix(u_skin, u_pale, 0.35), smoothstep(0.02, 0.07, depth) * 0.6);

          // Translucent lighting
          float wrap = clamp((NdL + 0.6) / 1.6, 0.0, 1.0);
          vec3 amb = u_bg * mix(0.35, 0.75, N.y * 0.5 + 0.5);
          vec3 albedo = u_flesh * fleshW + u_pale * paleW + skinCol * skinW;
          vec3 inner = albedo * (amb + vec3(1.0, 0.97, 0.92) * key * 0.5 * wrap);

          // Backlight glow at thin boundaries
          float back = pow(clamp(dot(V, -L), 0.0, 1.0), 3.0) * 1.5 + 0.2;
          vec3 glowTint = u_flesh * fleshW + vec3(0.9, 0.95, 0.78) * paleW + skinCol * 0.6 * skinW;
          vec3 glow = glowTint * back * key * 0.35 * (1.0 - skinW * 0.7);

          // Surface Fresnel reflection & specular
          vec3 R = reflect(-V, N);
          float rough = mix(0.12, 0.22, skinW);
          vec3 env = studioReflect(R, rough);
          vec3 H = normalize(V + L);
          float spec = pow(max(dot(N, H), 0.0), 60.0) * key;

          float F0 = mix(0.035, 0.045, skinW);
          float F = F0 + (1.0 - F0) * pow(1.0 - NdV, 5.0);

          vec3 color = (inner + glow) * (1.0 - F) + env * F + vec3(spec);
          fragColor = vec4(color, 1.0);
        }
        // 2. Seeds
        else if (v_mat < 1.5) {
          vec3 base = u_seed;
          float dif = max(dot(N, L), 0.0) * 0.5 + 0.35;
          float rim = pow(1.0 - NdV, 2.0);
          base += vec3(0.09, 0.035, 0.015) * rim;
          vec3 H = normalize(V + L);
          float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.3;
          fragColor = vec4(base * dif + vec3(spec), 1.0);
        }
        // 3. Bubbles
        else if (v_mat < 2.5) {
          float rim = pow(1.0 - NdV, 2.0);
          vec3 col = mix(vec3(0.9, 0.84, 0.82), vec3(0.62, 0.55, 0.55), rim);
          vec3 H = normalize(V + L);
          col += vec3(pow(max(dot(N, H), 0.0), 40.0) * 0.6);
          fragColor = vec4(col, 1.0);
        }
        // 4. Knife blade
        else if (v_mat < 4.5) {
          float bevel = 1.0 - smoothstep(3.05, 3.4, v_mat);
          float bolster = smoothstep(3.55, 3.65, v_mat);
          vec3 R = reflect(-V, N);
          vec3 env = studioReflect(R, mix(0.15, 0.06, bolster));
          vec3 H = normalize(V + L);
          float spec = pow(max(dot(N, H), 0.0), 48.0) * key * 0.8;
          vec3 steel = mix(vec3(0.72, 0.73, 0.75), vec3(0.88, 0.89, 0.90), bevel);
          vec3 col = env * steel + vec3(spec) + vec3(0.05) * max(dot(N, L), 0.0);
          fragColor = vec4(col, 1.0);
        }
        // 5. Knife handle
        else {
          float p = clamp((-v_rest.x - 0.07) / 0.84, 0.0, 1.0);
          float cy = 0.56 - 0.028 * p * p;
          float rivet = 0.0;
          for (int k = 0; k < 3; k++) {
            float rx = -0.24 - float(k) * 0.25;
            float dd = length(vec2(v_rest.x - rx, v_rest.y - cy));
            rivet = max(rivet, 1.0 - smoothstep(0.017, 0.021, dd));
          }
          vec3 wood = vec3(0.06, 0.03, 0.016);
          float dif = max(dot(N, L), 0.0) * 0.8 + 0.3;
          vec3 R = reflect(-V, N);
          vec3 envW = studioReflect(R, 0.35);
          vec3 col = wood * dif + envW * 0.15;
          col = mix(col, vec3(0.75, 0.77, 0.8), rivet);
          fragColor = vec4(col, 1.0);
        }
      }
    `;
    this.programMesh = this.createProgram(vsMesh, fsMesh);
    this.uniformsMesh = this.getUniforms(this.programMesh, [
      'u_viewProj', 'u_camPos', 'u_keyDir', 'u_flesh', 'u_fleshDeep',
      'u_pale', 'u_skin', 'u_stripe', 'u_seed', 'u_shape', 'u_bg'
    ]);

    // 3. Lines program (wireframe)
    const vsLines = `#version 300 es
      in vec3 a_position;
      uniform mat4 u_viewProj;
      void main() {
        gl_Position = u_viewProj * vec4(a_position, 1.0);
      }
    `;
    const fsLines = `#version 300 es
      precision mediump float;
      out vec4 fragColor;
      uniform float u_meshAlpha;
      void main() {
        fragColor = vec4(0.04, 0.05, 0.06, u_meshAlpha);
      }
    `;
    this.programLines = this.createProgram(vsLines, fsLines);
    this.uniformsLines = this.getUniforms(this.programLines, ['u_viewProj', 'u_meshAlpha']);
  }

  createShader(type: number, src: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error('Shader compile failed: ' + err);
    }
    return shader;
  }

  createProgram(vsSrc: string, fsSrc: string): WebGLProgram {
    const gl = this.gl;
    const p = gl.createProgram()!;
    const vs = this.createShader(gl.VERTEX_SHADER, vsSrc);
    const fs = this.createShader(gl.FRAGMENT_SHADER, fsSrc);
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const err = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error('Program link failed: ' + err);
    }
    return p;
  }

  getUniforms(prog: WebGLProgram, names: string[]): Record<string, WebGLUniformLocation> {
    const gl = this.gl;
    const map: Record<string, WebGLUniformLocation> = {};
    for (const name of names) {
      const loc = gl.getUniformLocation(prog, name);
      if (loc) map[name] = loc;
    }
    return map;
  }

  setKnife(k: KnifeMesh) {
    const gl = this.gl;
    const nK = k.rest.length / 3;
    this.knifeDyn = new Float32Array(nK * 6);
    this.knifeDynBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.knifeDynBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.knifeDyn.byteLength, gl.DYNAMIC_DRAW);

    const st = new Float32Array(nK * 4);
    for (let i = 0; i < nK; i++) {
      st[4 * i] = k.rest[3 * i];
      st[4 * i + 1] = k.rest[3 * i + 1];
      st[4 * i + 2] = k.rest[3 * i + 2];
      st[4 * i + 3] = k.mat[i];
    }
    this.knifeStatic = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.knifeStatic);
    gl.bufferData(gl.ARRAY_BUFFER, st, gl.STATIC_DRAW);

    this.knifeIdx = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.knifeIdx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, k.index, gl.STATIC_DRAW);
    this.knifeCount = k.index.length;
  }

  setMesh(mesh: RenderMesh, simEdges: Uint32Array, nParticles: number) {
    const gl = this.gl;
    this.mesh = mesh;
    this.nV = mesh.rest.length / 3;

    this.dyn = new Float32Array(this.nV * 6);
    this.dynBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.dyn.byteLength, gl.DYNAMIC_DRAW);

    const st = new Float32Array(this.nV * 4);
    for (let i = 0; i < this.nV; i++) {
      st[4 * i] = mesh.rest[3 * i];
      st[4 * i + 1] = mesh.rest[3 * i + 1];
      st[4 * i + 2] = mesh.rest[3 * i + 2];
      st[4 * i + 3] = mesh.mat[i];
    }
    this.staticBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.staticBuf);
    gl.bufferData(gl.ARRAY_BUFFER, st, gl.STATIC_DRAW);

    this.indexBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.index, gl.STATIC_DRAW);

    this.lineIdx = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineIdx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, simEdges, gl.STATIC_DRAW);
    this.lineCount = simEdges.length;

    this.particleBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleBuf);
    gl.bufferData(gl.ARRAY_BUFFER, nParticles * 12, gl.DYNAMIC_DRAW);
  }

  resize(w: number, h: number) {
    w = Math.max(1, Math.floor(w));
    h = Math.max(1, Math.floor(h));
    if (w === this.w && h === this.h) return;
    this.w = w;
    this.h = h;
    this.canvas.width = w;
    this.canvas.height = h;
    this.gl.viewport(0, 0, w, h);
  }

  cachedUniforms: any = {};
  setUniforms(cam: any, light: any, palette: PaletteDef, params: any) {
    this.cachedUniforms = { cam, light, palette, params };
  }

  uploadGeometry(particles: Float32Array, showMesh: boolean) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.dyn);

    if (this.knifeVisible && this.knifeDynBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.knifeDynBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.knifeDyn);
    }

    if (showMesh) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.particleBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, particles);
    }
  }

  capture(): Promise<{ w: number; h: number; px: Uint8ClampedArray }> {
    const gl = this.gl;
    const px = new Uint8ClampedArray(this.w * this.h * 4);
    gl.readPixels(0, 0, this.w, this.h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return Promise.resolve({ w: this.w, h: this.h, px });
  }

  draw(showMesh: boolean) {
    const gl = this.gl;
    const u = this.cachedUniforms;
    if (!u.cam) return;

    gl.viewport(0, 0, this.w, this.h);
    gl.clearColor(u.params.bg[0], u.params.bg[1], u.params.bg[2], 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 1. Draw floor pass
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(this.programFloor);
    gl.uniformMatrix4fv(this.uniformsFloor.u_invViewProj, false, u.cam.invViewProj);
    gl.uniform3fv(this.uniformsFloor.u_camPos, u.cam.pos);
    gl.uniform3fv(this.uniformsFloor.u_bg, u.params.bg);
    gl.uniform3fv(this.uniformsFloor.u_centroid, [0, 0, 1.0]);
    gl.uniform3fv(this.uniformsFloor.u_shadowTint, u.palette.shadowTint);

    const aPosLoc = gl.getAttribLocation(this.programFloor, 'a_pos');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.floorQuadBuf);
    gl.enableVertexAttribArray(aPosLoc);
    gl.vertexAttribPointer(aPosLoc, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(aPosLoc);

    // 2. Draw 3D Jelly & Props
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);

    gl.useProgram(this.programMesh);
    gl.uniformMatrix4fv(this.uniformsMesh.u_viewProj, false, u.cam.viewProj);
    gl.uniform3fv(this.uniformsMesh.u_camPos, u.cam.pos);
    gl.uniform4fv(this.uniformsMesh.u_keyDir, [...u.light.dir, u.light.intensity]);
    gl.uniform3fv(this.uniformsMesh.u_flesh, u.palette.flesh);
    gl.uniform3fv(this.uniformsMesh.u_fleshDeep, u.palette.fleshDeep);
    gl.uniform3fv(this.uniformsMesh.u_pale, u.palette.pale);
    gl.uniform3fv(this.uniformsMesh.u_skin, u.palette.skin);
    gl.uniform3fv(this.uniformsMesh.u_stripe, u.palette.stripe);
    gl.uniform3fv(this.uniformsMesh.u_seed, u.palette.seed);
    gl.uniform4fv(this.uniformsMesh.u_shape, u.params.shape);
    gl.uniform3fv(this.uniformsMesh.u_bg, u.params.bg);

    const aPos = gl.getAttribLocation(this.programMesh, 'a_position');
    const aNorm = gl.getAttribLocation(this.programMesh, 'a_normal');
    const aRest = gl.getAttribLocation(this.programMesh, 'a_rest');
    const aMat = gl.getAttribLocation(this.programMesh, 'a_mat');

    // Bind mesh buffers
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynBuf);
    if (aPos >= 0) {
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 24, 0);
    }
    if (aNorm >= 0) {
      gl.enableVertexAttribArray(aNorm);
      gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 24, 12);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.staticBuf);
    if (aRest >= 0) {
      gl.enableVertexAttribArray(aRest);
      gl.vertexAttribPointer(aRest, 3, gl.FLOAT, false, 16, 0);
    }
    if (aMat >= 0) {
      gl.enableVertexAttribArray(aMat);
      gl.vertexAttribPointer(aMat, 1, gl.FLOAT, false, 16, 12);
    }

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuf);
    // Draw body, seeds, and bubbles
    const totalCount = this.mesh.body.count + this.mesh.seeds.count + this.mesh.bubbles.count;
    gl.drawElements(gl.TRIANGLES, totalCount, gl.UNSIGNED_INT, 0);

    // Draw knife if visible
    if (this.knifeVisible && this.knifeDynBuf) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.knifeDynBuf);
      if (aPos >= 0) gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 24, 0);
      if (aNorm >= 0) gl.vertexAttribPointer(aNorm, 3, gl.FLOAT, false, 24, 12);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.knifeStatic);
      if (aRest >= 0) gl.vertexAttribPointer(aRest, 3, gl.FLOAT, false, 16, 0);
      if (aMat >= 0) gl.vertexAttribPointer(aMat, 1, gl.FLOAT, false, 16, 12);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.knifeIdx);
      gl.drawElements(gl.TRIANGLES, this.knifeCount, gl.UNSIGNED_INT, 0);
    }

    if (aPos >= 0) gl.disableVertexAttribArray(aPos);
    if (aNorm >= 0) gl.disableVertexAttribArray(aNorm);
    if (aRest >= 0) gl.disableVertexAttribArray(aRest);
    if (aMat >= 0) gl.disableVertexAttribArray(aMat);

    // 3. Draw wireframe lines if requested
    if (showMesh) {
      gl.disable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.useProgram(this.programLines);
      gl.uniformMatrix4fv(this.uniformsLines.u_viewProj, false, u.cam.viewProj);
      gl.uniform1f(this.uniformsLines.u_meshAlpha, u.params.meshAlpha);

      const aLinePos = gl.getAttribLocation(this.programLines, 'a_position');
      gl.bindBuffer(gl.ARRAY_BUFFER, this.particleBuf);
      if (aLinePos >= 0) {
        gl.enableVertexAttribArray(aLinePos);
        gl.vertexAttribPointer(aLinePos, 3, gl.FLOAT, false, 12, 0);
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineIdx);
      gl.drawElements(gl.LINES, this.lineCount, gl.UNSIGNED_INT, 0);
      if (aLinePos >= 0) gl.disableVertexAttribArray(aLinePos);
      gl.disable(gl.BLEND);
    }
  }
}
