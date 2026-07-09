import { POS, NEG } from "../lib/palette.js";

// Per-stock out-of-sample track record: each dot is one month — predicted return
// (x) vs realized return (y). Dots in the top-right / bottom-left quadrants got the
// direction right. Honest by design: on a single name the edge is mostly noise.
const W = 480, H = 330, M = { t: 16, r: 16, b: 42, l: 50 };

export default function TrackRecord({ series, ticker }) {
  const { pred, realized } = series;
  const n = pred.length;

  let hits = 0;
  for (let i = 0; i < n; i++) if ((pred[i] >= 0) === (realized[i] >= 0)) hits++;
  const hitRate = hits / n;

  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const mp = mean(pred), mr = mean(realized);
  let cov = 0, vp = 0, vr = 0;
  for (let i = 0; i < n; i++) {
    const dp = pred[i] - mp, dr = realized[i] - mr;
    cov += dp * dr; vp += dp * dp; vr += dr * dr;
  }
  const ic = vp > 0 && vr > 0 ? cov / Math.sqrt(vp * vr) : 0;

  const dom = Math.max(0.02, ...pred.map(Math.abs), ...realized.map(Math.abs));
  const clamp = (v) => Math.max(-dom, Math.min(dom, v));
  const x = (v) => M.l + ((clamp(v) + dom) / (2 * dom)) * (W - M.l - M.r);
  const y = (v) => (H - M.b) - ((clamp(v) + dom) / (2 * dom)) * (H - M.t - M.b);
  const x0 = x(0), y0 = y(0);

  const verdict = hitRate >= 0.55
    ? `The model has called ${ticker}'s monthly direction right ${(hitRate * 100).toFixed(0)}% of ${n} months — a mild but real edge on this name.`
    : hitRate >= 0.45
      ? `Roughly a coin flip on ${ticker} (${(hitRate * 100).toFixed(0)}% of ${n} months). That's expected — the model's edge is a portfolio effect across many names, not a single-stock oracle.`
      : `Below a coin flip on ${ticker} (${(hitRate * 100).toFixed(0)}% of ${n} months) — it hasn't been reliable on this specific name.`;

  return (
    <div>
      <div className="tiles" style={{ marginBottom: 16 }}>
        <div className="panel tile">
          <div className="k">Directional hit rate</div>
          <div className={`v ${hitRate >= 0.5 ? "pos" : "neg"}`}>{(hitRate * 100).toFixed(0)}%</div>
          <div className="note">{hits} of {n} months correct</div>
        </div>
        <div className="panel tile">
          <div className="k">Per-stock IC</div>
          <div className={`v ${ic >= 0 ? "pos" : "neg"}`}>{ic.toFixed(3)}</div>
          <div className="note">predicted vs realized correlation</div>
        </div>
        <div className="panel tile">
          <div className="k">Months evaluated</div>
          <div className="v">{n}</div>
          <div className="note">out-of-sample, monthly</div>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="track-svg" role="img" aria-label={`predicted vs realized returns for ${ticker}`}>
        <rect x={x0} y={M.t} width={W - M.r - x0} height={y0 - M.t} fill={POS} opacity="0.05" />
        <rect x={M.l} y={y0} width={x0 - M.l} height={H - M.b - y0} fill={POS} opacity="0.05" />
        <line x1={x0} x2={x0} y1={M.t} y2={H - M.b} stroke="var(--line)" strokeWidth="1" />
        <line x1={M.l} x2={W - M.r} y1={y0} y2={y0} stroke="var(--line)" strokeWidth="1" />
        {pred.map((p, i) => {
          const correct = (p >= 0) === (realized[i] >= 0);
          return <circle key={i} cx={x(p)} cy={y(realized[i])} r="3.4" fill={correct ? POS : NEG} opacity="0.62" />;
        })}
        <text x={(M.l + W - M.r) / 2} y={H - 10} textAnchor="middle" className="fan-ax">predicted return →</text>
        <text x={16} y={(M.t + H - M.b) / 2} textAnchor="middle" className="fan-ax"
          transform={`rotate(-90 16 ${(M.t + H - M.b) / 2})`}>realized return →</text>
      </svg>

      <p className="cap" style={{ marginTop: 12 }}>
        <span style={{ color: "var(--pos)" }}>●</span> correct direction &nbsp;
        <span style={{ color: "var(--neg)" }}>●</span> wrong direction &nbsp;— {verdict}
      </p>
    </div>
  );
}
