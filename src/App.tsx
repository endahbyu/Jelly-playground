/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { SimulationController, PALETTES, hex } from './sim/simulation';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const controllerRef = useRef<SimulationController | null>(null);

  const [statusText, setStatusText] = useState('WEBGPU · STARTING');
  const [statusState, setStatusState] = useState<'live' | 'paused' | 'off'>('live');
  const [fallback, setFallback] = useState<{ title: string; detail: string } | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const toastTimeoutRef = useRef<any>(null);

  const [stroke, setStroke] = useState({ x1: 0, y1: 0, x2: 0, y2: 0, on: false, fading: false });
  const [cursor, setCursor] = useState<'grab' | 'grabbing' | 'orbit' | 'knife' | ''>('');

  const [tool, setTool] = useState<'hand' | 'knife'>('hand');
  const [currentPalette, setCurrentPalette] = useState('crimson');
  const [firmness, setFirmness] = useState(40);
  const [damping, setDamping] = useState(45);
  const [slow, setSlow] = useState(false);
  const [showMesh, setShowMesh] = useState(false);
  const [paused, setPaused] = useState(false);

  const [stats, setStats] = useState({
    particles: 0,
    tets: 0,
    mass: '—',
    volume: '—',
    kinetic: '—',
    pieces: 1,
    fps: 60,
  });

  const showToast = useCallback((msg: string) => {
    setToastMessage(msg);
    clearTimeout(toastTimeoutRef.current);
    toastTimeoutRef.current = setTimeout(() => {
      setToastMessage(null);
    }, 2200);
  }, []);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;

    const controller = new SimulationController(canvas, {
      onStatusChange: (text, state) => {
        setStatusText(text);
        setStatusState(state);
      },
      onFallback: (title, detail) => {
        setFallback({ title, detail });
        document.body.classList.add('no-gpu');
      },
      onToast: showToast,
      onStrokeChange: (x1, y1, x2, y2, on, fading) => {
        const r = canvas.getBoundingClientRect();
        setStroke({
          x1: x1 - r.left,
          y1: y1 - r.top,
          x2: x2 - r.left,
          y2: y2 - r.top,
          on,
          fading,
        });
      },
      onCursorChange: cur => {
        setCursor(cur);
      },
      onStatsChange: newStats => {
        setStats(newStats);
      },
    });

    controllerRef.current = controller;
    controller.start();

    return () => {
      controller.destroy();
      controllerRef.current = null;
    };
  }, [showToast]);

  const handleToolChange = (newTool: 'hand' | 'knife') => {
    setTool(newTool);
    document.body.dataset.tool = newTool;
    controllerRef.current?.setTool(newTool);
  };

  const handlePaletteChange = (key: string) => {
    setCurrentPalette(key);
    controllerRef.current?.setPalette(key);
  };

  const handleFirmnessChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value);
    setFirmness(val);
    controllerRef.current?.setFirmness(val / 100);
  };

  const handleDampingChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value);
    setDamping(val);
    controllerRef.current?.setDamping(val / 100);
  };

  const handleNudge = () => {
    controllerRef.current?.nudge();
  };

  const handleReset = () => {
    controllerRef.current?.reset();
    setStats(prev => ({ ...prev, pieces: 1 }));
  };

  const handleSlowChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const checked = e.target.checked;
    setSlow(checked);
    controllerRef.current?.setSlow(checked);
  };

  const handleMeshChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const checked = e.target.checked;
    setShowMesh(checked);
    controllerRef.current?.setShowMesh(checked);
  };

  const handlePauseToggle = () => {
    setPaused(prev => {
      const next = !prev;
      controllerRef.current?.setPaused(next);
      return next;
    });
  };

  // Global keyboard shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input, button, summary')) return;
      if (e.key === ' ') {
        e.preventDefault();
        handlePauseToggle();
      } else if (e.key === 'n' || e.key === 'N') {
        handleNudge();
      } else if (e.key === 'r' || e.key === 'R') {
        handleReset();
      } else if (e.key === 'k' || e.key === 'K') {
        handleToolChange(tool === 'knife' ? 'hand' : 'knife');
      } else if (e.key === 'h' || e.key === 'H') {
        handleToolChange('hand');
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [tool, paused]);

  return (
    <>
      <div className="stage" id="stage">
        <canvas
          ref={canvasRef}
          id="gl"
          aria-label="Interactive 3D watermelon jelly slice. Drag it to stretch it, or switch to the knife and draw across it to cut it into pieces."
          role="img"
          data-cursor={cursor}
        />
        <svg
          className={`stroke ${stroke.on ? 'on' : ''} ${stroke.fading ? 'fading' : ''}`}
          id="stroke"
          aria-hidden="true"
        >
          <line
            id="strokeLine"
            x1={stroke.x1}
            y1={stroke.y1}
            x2={stroke.x2}
            y2={stroke.y2}
          />
        </svg>
        <div
          className={`toast ${toastMessage ? 'on' : ''}`}
          id="toast"
          role="status"
          aria-live="polite"
        >
          {toastMessage}
        </div>
        <div className="fallback" id="fallback" hidden={!fallback}>
          <div>
            <span className="kicker">Renderer offline</span>
            <h2 id="fallbackTitle">{fallback?.title || 'This specimen needs WebGPU.'}</h2>
            <p id="fallbackDetail">{fallback?.detail}</p>
          </div>
        </div>
      </div>

      <header className="masthead">
        <div className="kicker">Material Studies / No. 009</div>
        <h1>
          <span>Melon</span>
          <span>Jelly.</span>
        </h1>
        <p className="caption">
          A slice of summer.
          <br />
          A little wobble.
          <br />
          Too soft to share.
        </p>
      </header>

      <div className="status" id="status" data-state={statusState} role="status" aria-live="polite">
        <span className="dot" aria-hidden="true"></span>
        <span id="statusText">{statusText}</span>
      </div>

      <aside className="panel" aria-label="Specimen controls">
        <header>
          <span className="label">The Specimen</span>
          <span>fig. 9</span>
        </header>

        <div className="group">
          <span className="label">Tool</span>
          <div className="tools" role="group" aria-label="Tool">
            <button
              className="tool"
              type="button"
              data-tool="hand"
              aria-pressed={tool === 'hand'}
              onClick={() => handleToolChange('hand')}
            >
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M6 10V5.5a1.2 1.2 0 0 1 2.4 0V9m0-4.8V3.8a1.2 1.2 0 0 1 2.4 0V9m0-4.2a1.2 1.2 0 0 1 2.4 0V9.5m0-3a1.2 1.2 0 0 1 2.4 0V12c0 3.3-2.4 5.5-5.4 5.5-2.2 0-3.6-1-4.8-2.8L3.3 11.6a1.3 1.3 0 0 1 2-1.6L6 11" />
              </svg>
              Hand
            </button>
            <button
              className="tool"
              type="button"
              data-tool="knife"
              aria-pressed={tool === 'knife'}
              onClick={() => handleToolChange('knife')}
            >
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M3 17l9.2-9.2M12.2 7.8l3.6-4.3c.8 1.6.3 4-1.4 5.7L12.6 11 9 7.4" />
                <path d="M3 17l2.6-.6" />
              </svg>
              Knife
            </button>
          </div>
        </div>

        <div className="group">
          <span className="label">Variety</span>
          <div className="swatches" id="swatches" role="radiogroup" aria-label="Colour variety">
            {Object.entries(PALETTES).map(([key, p]) => (
              <button
                key={key}
                type="button"
                className="swatch"
                data-key={key}
                role="radio"
                aria-checked={currentPalette === key}
                aria-label={`${p.name} variety`}
                onClick={() => handlePaletteChange(key)}
              >
                <span className="chip">
                  <i style={{ background: hex(p.skin) }}></i>
                  <i style={{ background: hex(p.pale.map(c => c * 0.95)) }}></i>
                  <i style={{ background: hex(p.ui.flesh) }}></i>
                </span>
                <span className="swname">{p.name}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="group">
          <div className="slider">
            <div className="row">
              <label className="label" htmlFor="firmness">
                Firmness
              </label>
              <output id="firmnessOut" htmlFor="firmness">
                {(firmness / 100).toFixed(2)}
              </output>
            </div>
            <input
              type="range"
              id="firmness"
              min="0"
              max="100"
              step="1"
              value={firmness}
              onChange={handleFirmnessChange}
            />
            <div className="ends">
              <span>trembling</span>
              <span>set</span>
            </div>
          </div>

          <div className="slider">
            <div className="row">
              <label className="label" htmlFor="damping">
                Internal damping
              </label>
              <output id="dampingOut" htmlFor="damping">
                {(damping / 100).toFixed(2)}
              </output>
            </div>
            <input
              type="range"
              id="damping"
              min="0"
              max="100"
              step="1"
              value={damping}
              onChange={handleDampingChange}
            />
            <div className="ends">
              <span>lively</span>
              <span>syrupy</span>
            </div>
          </div>
        </div>

        <div className="group">
          <div className="buttons">
            <button className="btn" id="nudge" type="button" onClick={handleNudge}>
              Give it a nudge
            </button>
            <button className="btn" id="reset" type="button" onClick={handleReset}>
              Reset
            </button>
          </div>
          <div className="checks">
            <label className="check">
              <input type="checkbox" id="slow" checked={slow} onChange={handleSlowChange} />
              <span className="box" aria-hidden="true"></span>¼ speed
            </label>
            <label className="check">
              <input type="checkbox" id="mesh" checked={showMesh} onChange={handleMeshChange} />
              <span className="box" aria-hidden="true"></span>Show mesh
            </label>
          </div>
          <button
            className="btn wide"
            id="pause"
            type="button"
            aria-pressed={paused}
            onClick={handlePauseToggle}
          >
            {paused ? 'Resume' : 'Pause'}
          </button>
        </div>
      </aside>

      <section className="readouts" aria-label="Live readouts">
        <p className="hint" id="hintText">
          {tool === 'knife' ? (
            <>
              <b>Knife</b>Draw a line across the slice — the knife lines up over it and cuts when you let go. Cut the pieces again, as small as you like.
            </>
          ) : (
            <>
              <b>Hand</b>Grab any piece — tip, corner, flesh or rind — and pull. Scroll, or add a second finger, while holding to twist it.
            </>
          )}
        </p>
        <dl className="stats">
          <div>
            <dt className="label">Mass</dt>
            <dd>
              ≈<span id="massOut">{stats.mass}</span>
              <small>g</small>
            </dd>
          </div>
          <div>
            <dt className="label">Volume</dt>
            <dd>
              <span id="volOut">{stats.volume}</span>
              <small>% of rest</small>
            </dd>
          </div>
          <div>
            <dt className="label">Kinetic</dt>
            <dd>
              <span id="keOut">{stats.kinetic}</span>
              <small>µJ</small>
            </dd>
          </div>
          <div>
            <dt className="label">Pieces</dt>
            <dd>
              <span id="piecesOut">{stats.pieces}</span>
            </dd>
          </div>
        </dl>
        <p className="footnote">
          Illustrative scale: 1 sim unit ≈ 3.5 cm, gummy at 1.3 g/cm³. Volume and energy are summed live over every tetrahedron and particle.
        </p>
      </section>

      <details className="notes">
        <summary>
          <span className="label">Inside the experiment</span>
        </summary>
        <div className="body">
          <h3>A body made of tetrahedra</h3>
          <p>
            The slice is a volumetric mesh of{' '}
            <span className="num" id="nParticles">
              {stats.particles ? stats.particles.toLocaleString('en') : '—'}
            </span>{' '}
            particles and{' '}
            <span className="num" id="nTets">
              {stats.tets ? stats.tets.toLocaleString('en') : '—'}
            </span>{' '}
            tetrahedra, solved with XPBD on a fixed 60&nbsp;Hz step split into 10 substeps. Each tetrahedron is pulled, with finite compliance, toward its best-fit rotated rest shape — a co-rotational elastic constraint that also flips inverted elements back out — and carries a volume constraint that keeps the jelly close to incompressible, so a stretched slice necks in like the real thing.
          </p>
          <p>
            Tetrahedra in the rind are roughly three times stiffer than the flesh. Damping acts only on relative velocity along the mesh edges, so the slice still falls and tumbles freely while its wobble dies away. Ground contact projects particles out of the floor with Coulomb-style friction; the grab is a soft attachment on a patch of particles, bounded in reach.
          </p>
          <h3>Cutting</h3>
          <p>
            Every piece is a convex outline in the slice’s original rest plane, with rounded corners. Your stroke, read on the surface it was drawn over, fixes a vertical blade plane. For each piece it crosses, the piece’s current pose is fitted with a best rotation, the plane is carried back into rest space and becomes a straight line, and the outline is split along it — the two halves are prebuilt as fresh tetrahedral meshes the moment you let go. Then the steel knife comes down: its edge first presses a groove into the jelly, and when it breaks through the new pieces are swapped in, taking over the position and speed of the flesh they came from, while the blade wedges the faces apart on its way to the board. The blade is drawn behind the jelly surface, so the part that is buried in it is seen refracted. Each piece is its own soft body — grab it, stretch it, cut it again — and pieces push against each other instead of passing through.
          </p>
          <h3>A surface that rides along</h3>
          <p>
            The smooth surface, the seeds and the air bubbles are separate meshes, each vertex pinned to a tetrahedron by barycentric weights — so the seeds stretch, turn and travel with the flesh around them.
          </p>
          <h3>Light through jelly</h3>
          <p>
            WGSL shaders render the floor, seeds and bubbles first, then the slice’s back faces to measure how much jelly each view ray crosses. The front faces refract the scene behind by that thickness, absorb it with Beer–Lambert colour, add milky scattering in the pale layer, a back-lit glow at thin edges, Fresnel reflections of a procedural studio and GGX highlights. The floor gets a soft key shadow tinted by the jelly plus contact occlusion from a top-down height map.
          </p>
        </div>
      </details>
    </>
  );
}
