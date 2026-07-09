import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip,
} from "recharts";
import { useAsync, loadMetrics, num } from "../lib/data.js";
import { MODELS, MODEL_COLOR, MODEL_LABEL } from "../lib/palette.js";

const METRICS = [
  { key: "ic", label: "Information coeff.", hint: "rank corr. of median vs realized — higher is better", fmt: (v) => num(v, 3) },
  { key: "pinball", label: "Pinball loss", hint: "quantile loss — lower is better", fmt: (v) => num(v, 4) },
  { key: "coverage", label: "Coverage", hint: "share of outcomes inside the 80% band — target 0.80", fmt: (v) => num(v, 2), ref: 0.8 },
  { key: "hit_rate", label: "Hit rate", hint: "directional accuracy — above 0.50 beats a coin flip", fmt: (v) => num(v, 3), ref: 0.5 },
];

function Tip({ active, payload, label, metric }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="fan-tip" style={{ position: "static", transform: "none", minWidth: 150 }}>
      <div className="k">{label}</div>
      {payload.map((p) => (
        <div className="row" key={p.dataKey}>
          <span style={{ color: p.fill }}>{MODEL_LABEL[p.dataKey]}</span>
          <b className="mono">{p.value == null ? "—" : metric.fmt(p.value)}</b>
        </div>
      ))}
    </div>
  );
}

export default function Dashboard() {
  const { data: m, error, loading } = useAsync(loadMetrics);
  const [metricKey, setMetricKey] = useState("ic");
  if (loading) return <div className="state">Loading metrics…</div>;
  if (error) return <div className="state">Could not load metrics.json — run the export step.</div>;

  const metric = METRICS.find((x) => x.key === metricKey);
  const HZ = Object.keys(m.baseline); // ["y_1m","y_3m","y_6m"]
  const data = HZ.map((hz) => {
    const row = { hz: hz.replace("y_", "").toUpperCase() };
    MODELS.forEach((mod) => { row[mod] = m[mod]?.[hz]?.[metricKey] ?? null; });
    return row;
  });
  const ens = m.ensemble;
  const bw = m.blend_weight_chronos || {};

  return (
    <>
      <div className="page-head">
        <span className="eyebrow">Out-of-sample · walk-forward CV</span>
        <h1>Model performance</h1>
        <p>
          Five models scored on held-out folds across three horizons. The ensemble blends
          GBM and zero-shot Chronos per horizon, then conformal calibration corrects the bands.
        </p>
      </div>

      <div className="tiles" style={{ marginBottom: 20 }}>
        {HZ.map((hz) => (
          <div className="panel tile" key={hz}>
            <div className="k">Ensemble IC · {hz.replace("y_", "").toUpperCase()}</div>
            <div className="v">{num(ens[hz].ic, 3)}</div>
            <div className="note">coverage {num(ens[hz].coverage, 2)} · hit {num(ens[hz].hit_rate, 2)}</div>
          </div>
        ))}
      </div>

      <div className="panel chart-card">
        <div className="controls" style={{ marginBottom: 6 }}>
          <div className="seg" role="tablist" aria-label="metric">
            {METRICS.map((x) => (
              <button key={x.key} className={x.key === metricKey ? "on" : ""}
                onClick={() => setMetricKey(x.key)} aria-selected={x.key === metricKey}>
                {x.label}
              </button>
            ))}
          </div>
        </div>
        <p className="cap">{metric.hint}</p>
        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} barCategoryGap="22%">
            <CartesianGrid stroke="#172136" vertical={false} />
            <XAxis dataKey="hz" tick={{ fill: "#7b89a3", fontFamily: "IBM Plex Mono", fontSize: 12 }} axisLine={{ stroke: "#1e2a40" }} tickLine={false} />
            <YAxis tick={{ fill: "#7b89a3", fontFamily: "IBM Plex Mono", fontSize: 11 }} axisLine={false} tickLine={false} width={48} />
            <Tooltip content={<Tip metric={metric} />} cursor={{ fill: "#ffffff0a" }} />
            {metric.ref != null && <ReferenceLine y={metric.ref} stroke="#f5c451" strokeDasharray="4 4" strokeOpacity={0.7} />}
            {MODELS.map((mod) => (
              <Bar key={mod} dataKey={mod} fill={MODEL_COLOR[mod]} radius={[3, 3, 0, 0]} maxBarSize={26} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        <div className="legend">
          {MODELS.map((mod) => (
            <span key={mod}><i style={{ background: MODEL_COLOR[mod] }} />{MODEL_LABEL[mod]}</span>
          ))}
          {metric.ref != null && <span><i style={{ background: "#f5c451" }} />target {metric.ref}</span>}
        </div>
      </div>

      <div className="panel chart-card" style={{ marginTop: 16 }}>
        <h3>Ensemble composition</h3>
        <p className="cap">Per-horizon Chronos weight (rest is GBM) and the conformal band shift applied to reach 80% coverage.</p>
        <table className="tbl">
          <thead><tr><th>Horizon</th><th className="right">Chronos weight</th><th className="right">GBM weight</th><th className="right">Conformal Δ</th></tr></thead>
          <tbody>
            {HZ.map((hz) => {
              const k = hz.replace("y_", "");
              const w = bw[k] ?? 1;
              const d = (m.conformal_delta || {})[k];
              return (
                <tr key={hz} style={{ cursor: "default" }}>
                  <td className="tk">{k.toUpperCase()}</td>
                  <td className="right mono">{num(w, 2)}</td>
                  <td className="right mono">{num(1 - w, 2)}</td>
                  <td className="right mono">{d == null ? "—" : (d >= 0 ? "+" : "") + num(d, 4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}
