import { POS, NEG } from "../lib/palette.js";

// Mini q10–q90 range centered on zero, median tick. `scale` is shared across the
// column so every row is on the same axis — comparable at a glance. Sign of the
// median drives the diverging color (positive vs negative expected return).
const W = 150, HH = 18;

export default function RangeBar({ h, scale }) {
  const x = (v) => W / 2 + (v / (scale || 1)) * (W / 2);
  const c = h.q50 >= 0 ? POS : NEG;
  const lo = Math.max(0, x(h.q10)), hi = Math.min(W, x(h.q90));
  return (
    <svg viewBox={`0 0 ${W} ${HH}`} width={W} height={HH} className="rangebar" aria-hidden="true">
      <line x1={W / 2} x2={W / 2} y1="2" y2={HH - 2} stroke="var(--line)" strokeWidth="1" />
      <rect x={lo} y={HH / 2 - 3} width={Math.max(1, hi - lo)} height="6" rx="3" fill={c} opacity="0.28" />
      <rect x={x(h.q50) - 1} y={HH / 2 - 5} width="2.5" height="10" rx="1" fill={c} />
    </svg>
  );
}
