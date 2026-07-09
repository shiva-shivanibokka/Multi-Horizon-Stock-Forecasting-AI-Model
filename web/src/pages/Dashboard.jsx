import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip,
} from "recharts";
import { useAsync, loadMetrics, num } from "../lib/data.js";
import { MODELS, MODEL_COLOR, MODEL_LABEL } from "../lib/palette.js";
import Info from "../components/Info.jsx";

const METRICS = [
  { key: "ic", label: "Skill (IC)", hint: "Information coefficient — how well the model ranks winners vs losers. 0 = no skill, higher is better. On daily equities, even 0.03–0.06 is a real edge.", fmt: (v) => num(v, 3) },
  { key: "pinball", label: "Pinball loss", hint: "Overall quality of the whole predicted distribution — lower is better.", fmt: (v) => num(v, 4) },
  { key: "coverage", label: "Calibration", hint: "How often the real outcome landed inside the 80% band. Perfect = 0.80.", fmt: (v) => num(v, 2), ref: 0.8 },
  { key: "hit_rate", label: "Direction", hint: "How often the model got the up/down direction right. Above 0.50 beats a coin flip.", fmt: (v) => num(v, 3), ref: 0.5 },
];

function Tip({ active, payload, label, metric }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="fan-tip" style={{ position: "static", transform: "none", minWidth: 160 }}>
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
  const HZ = Object.keys(m.baseline);
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
        <span className="eyebrow">Performance</span>
        <h1>How good are these forecasts, honestly?</h1>
        <p className="lead">
          Every model is scored on data it never saw during training (<b>walk-forward cross-validation</b>). We compare five
          models across three horizons. The <b>Ensemble</b> is the one powering the Forecast and Screener tabs — it blends a
          gradient-boosted model with a foundation time-series model, then calibrates the uncertainty bands.
        </p>
      </div>

      <div className="tiles" style={{ marginBottom: 22 }}>
        {HZ.map((hz) => (
          <div className="panel tile" key={hz}>
            <div className="k">Ensemble skill · {hz.replace("y_", "").toUpperCase()}
              <Info text="Information coefficient (IC): rank correlation between the forecast and what actually happened. Small positive numbers are genuinely useful on stocks." />
            </div>
            <div className="v">{num(ens[hz].ic, 3)}</div>
            <div className="note">calibration {num(ens[hz].coverage, 2)} · direction {num(ens[hz].hit_rate, 2)}</div>
          </div>
        ))}
      </div>

      <div className="panel chart-card">
        <h3>Model comparison
          <Info text="Compare all five models on one metric at a time. Switch metrics with the buttons — each measures a different thing." />
        </h3>
        <div className="controls" style={{ margin: "14px 0 2px" }}>
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
        <ResponsiveContainer width="100%" height={370}>
          <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }} barCategoryGap="24%">
            <CartesianGrid stroke="#141d31" vertical={false} />
            <XAxis dataKey="hz" tick={{ fill: "#8695b2", fontFamily: "JetBrains Mono", fontSize: 13 }} axisLine={{ stroke: "#1c2740" }} tickLine={false} />
            <YAxis tick={{ fill: "#8695b2", fontFamily: "JetBrains Mono", fontSize: 12 }} axisLine={false} tickLine={false} width={52} />
            <Tooltip content={<Tip metric={metric} />} cursor={{ fill: "#ffffff0c" }} />
            {metric.ref != null && <ReferenceLine y={metric.ref} stroke="#8f7bff" strokeDasharray="5 4" strokeOpacity={0.8} />}
            {MODELS.map((mod) => (
              <Bar key={mod} dataKey={mod} fill={MODEL_COLOR[mod]} radius={[4, 4, 0, 0]} maxBarSize={30} isAnimationActive={false} />
            ))}
          </BarChart>
        </ResponsiveContainer>
        <div className="legend">
          {MODELS.map((mod) => (
            <span key={mod}><i style={{ background: MODEL_COLOR[mod] }} />{MODEL_LABEL[mod]}</span>
          ))}
          {metric.ref != null && <span><i style={{ background: "#8f7bff" }} />target {metric.ref}</span>}
        </div>
      </div>

      <div className="panel chart-card" style={{ marginTop: 18 }}>
        <h3>What's inside the ensemble
          <Info text="For each horizon, how much weight goes to the Chronos foundation model vs the gradient-boosted model, plus the calibration nudge applied to the bands." />
        </h3>
        <p className="cap">Fit on a validation fold, then applied to held-out test data.</p>
        <table className="tbl">
          <thead><tr><th>Horizon</th><th className="right">Chronos weight</th><th className="right">GBM weight</th><th className="right">Calibration Δ</th></tr></thead>
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
