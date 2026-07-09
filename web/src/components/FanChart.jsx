import { useState } from "react";
import { BAND, SIGNAL, HORIZON_DAYS } from "../lib/palette.js";
import { pct } from "../lib/data.js";

// The signature view: recent price on the left, then a quantile band that widens
// from the anchor (day 0) out to the 6-month horizon. Anchor return is 0 by
// construction, so the fan opens from a single point — uncertainty made visible.
const W = 920, H = 400;
const M = { t: 20, r: 62, b: 34, l: 56 };
const HIST_DAYS = 120;

const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");

export default function FanChart({ history, anchorPrice, horizons }) {
  const [hover, setHover] = useState(null);

  const hist = (history?.close || []).slice(-HIST_DAYS);
  const hDates = (history?.dates || []).slice(-HIST_DAYS);
  const hxs = hist.map((_, i) => i - (hist.length - 1)); // -(n-1)..0

  const hzKeys = Object.keys(HORIZON_DAYS).filter((k) => horizons?.[k]);
  const maxDay = Math.max(...hzKeys.map((k) => HORIZON_DAYS[k]), 1);

  // future quantile paths (price levels from the anchor)
  const priceAt = (ret) => anchorPrice * (1 + ret);
  const qPts = (q) => [
    { d: 0, p: anchorPrice },
    ...hzKeys.map((k) => ({ d: HORIZON_DAYS[k], p: priceAt(horizons[k][q]) })),
  ];
  const q10 = qPts("q10"), q50 = qPts("q50"), q90 = qPts("q90");

  const prices = [...hist, ...q10.map((p) => p.p), ...q90.map((p) => p.p)];
  const yMin = Math.min(...prices), yMax = Math.max(...prices);
  const pad = (yMax - yMin) * 0.06 || 1;
  const x = lin(-(hist.length - 1), maxDay, M.l, W - M.r);
  const y = lin(yMin - pad, yMax + pad, H - M.b, M.t);

  const XY = (d, p) => ({ x: x(d), y: y(p) });
  const histPts = hist.map((p, i) => XY(hxs[i], p));
  const band = [...q90.map((p) => XY(p.d, p.p)), ...q10.slice().reverse().map((p) => XY(p.d, p.p))];
  const medPts = q50.map((p) => XY(p.d, p.p));

  // y grid (4 ticks)
  const ticks = Array.from({ length: 4 }, (_, i) => yMin + ((yMax - yMin) * i) / 3);
  const anchorX = x(0);

  return (
    <div className="fan">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Projected price distribution">
        <defs>
          <linearGradient id="bandgrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={BAND} stopOpacity="0.34" />
            <stop offset="100%" stopColor={BAND} stopOpacity="0.06" />
          </linearGradient>
        </defs>

        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} stroke="var(--line-soft)" strokeWidth="1" />
            <text x={M.l - 8} y={y(t) + 4} textAnchor="end" className="fan-ax">{t.toFixed(0)}</text>
          </g>
        ))}

        {/* anchor divider: history | forecast */}
        <line x1={anchorX} x2={anchorX} y1={M.t} y2={H - M.b} stroke="var(--line)" strokeWidth="1" strokeDasharray="3 4" />
        <text x={anchorX} y={M.t - 6} textAnchor="middle" className="fan-ax">now</text>

        {/* history */}
        <path d={path(histPts)} fill="none" stroke="var(--ink-lo)" strokeWidth="1.6" />

        {/* fan band + median (animated reveal via clip) */}
        <g className="fan-reveal">
          <path d={`${path(band)} Z`} fill="url(#bandgrad)" stroke="none" />
          <path d={path(medPts)} fill="none" stroke={SIGNAL} strokeWidth="2.2" strokeLinecap="round" />
        </g>

        <circle cx={anchorX} cy={y(anchorPrice)} r="3.5" fill={SIGNAL} />

        {/* horizon markers + hit targets */}
        {hzKeys.map((k) => {
          const px = x(HORIZON_DAYS[k]);
          const active = hover === k;
          return (
            <g key={k} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)} style={{ cursor: "pointer" }}>
              <rect x={px - 14} y={M.t} width="28" height={H - M.t - M.b} fill="transparent" />
              <line x1={px} x2={px} y1={M.t} y2={H - M.b} stroke="var(--line)" strokeWidth={active ? 1.5 : 1} opacity={active ? 0.9 : 0.4} />
              <circle cx={px} cy={y(priceAt(horizons[k].q50))} r={active ? 5 : 4} fill={SIGNAL} stroke="var(--bg)" strokeWidth="1.5" />
              <text x={px} y={H - M.b + 18} textAnchor="middle" className="fan-ax" fill={active ? "var(--ink-hi)" : "var(--ink-lo)"}>
                {k.toUpperCase()}
              </text>
            </g>
          );
        })}
      </svg>

      {hover && (
        <div className="fan-tip" style={{ left: `${(x(HORIZON_DAYS[hover]) / W) * 100}%` }}>
          <div className="k">{hover.toUpperCase()} · {hDates.length ? "from " + hDates[hDates.length - 1] : ""}</div>
          <div className="row"><span>q90</span><b className="mono">{pct(horizons[hover].q90)}</b></div>
          <div className="row med"><span>median</span><b className="mono">{pct(horizons[hover].q50)}</b></div>
          <div className="row"><span>q10</span><b className="mono">{pct(horizons[hover].q10)}</b></div>
        </div>
      )}
    </div>
  );
}
