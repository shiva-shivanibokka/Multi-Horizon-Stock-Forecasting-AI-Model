import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAsync, loadForecasts, pct, signClass, HORIZON_LABEL } from "../lib/data.js";
import RangeBar from "../components/RangeBar.jsx";
import Info from "../components/Info.jsx";
import Combo from "../components/Combo.jsx";

const HORIZONS = ["1m", "3m", "6m"];

export default function Screener() {
  const { data, error, loading } = useAsync(loadForecasts);
  const nav = useNavigate();
  const [hz, setHz] = useState("3m");
  const [sector, setSector] = useState("All");
  const [q, setQ] = useState("");
  const [sortDir, setSortDir] = useState("desc");

  const sectors = useMemo(() => {
    if (!data) return [];
    return ["All", ...Array.from(new Set(data.tickers.map((t) => t.sector))).sort()];
  }, [data]);

  const rows = useMemo(() => {
    if (!data) return [];
    const ql = q.trim().toUpperCase();
    const filtered = data.tickers.filter(
      (t) => (sector === "All" || t.sector === sector) &&
        (!ql || t.ticker.includes(ql) || t.name.toUpperCase().includes(ql))
    );
    filtered.sort((a, b) => (b.horizons[hz].q50 - a.horizons[hz].q50) * (sortDir === "desc" ? 1 : -1));
    return filtered;
  }, [data, hz, sector, q, sortDir]);

  const scale = useMemo(() => {
    let s = 0.01;
    for (const t of rows) s = Math.max(s, Math.abs(t.horizons[hz].q10), Math.abs(t.horizons[hz].q90));
    return s;
  }, [rows, hz]);

  if (loading) return <div className="state">Loading forecasts…</div>;
  if (error) return <div className="state">Could not load forecasts.json — run the export step.</div>;

  return (
    <>
      <div className="page-head">
        <span className="eyebrow">Screener</span>
        <h1>Which stocks does the model favor right now?</h1>
        <p className="lead">
          Every S&amp;P 500 company ranked by its expected <b>{HORIZON_LABEL[hz]}</b> return. The colored bar shows each stock's
          full 10–90% range around zero — so you can see conviction <em>and</em> uncertainty at a glance. Click any row to open its forecast.
        </p>
      </div>

      <div className="controls">
        <div className="seg" role="tablist" aria-label="horizon">
          {HORIZONS.map((h) => (
            <button key={h} className={h === hz ? "on" : ""} onClick={() => setHz(h)}>{h.toUpperCase()}</button>
          ))}
        </div>
        <Combo value={sector} options={sectors.map((s) => ({ value: s, label: s }))} onChange={setSector}
          width={220} searchable={false} placeholder="Sector" />
        <input className="field" placeholder="Search ticker or company…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="search" style={{ flex: "1 1 220px", maxWidth: 300 }} />
        <span className="mono" style={{ color: "var(--ink-faint)", fontSize: 13 }}>{rows.length} companies</span>
      </div>

      <div className="panel" style={{ overflowX: "auto" }}>
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: 40 }}>#</th>
              <th>Company</th>
              <th>Sector</th>
              <th className="right sortable" onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}>
                Expected <span className="arw">{sortDir === "desc" ? "▼" : "▲"}</span>
                <Info text={`Median forecast return over ${HORIZON_LABEL[hz]}. Click to flip the sort order.`} />
              </th>
              <th>Range
                <Info text="The 10th–90th percentile of the return, centered on zero. Longer bar = more uncertainty; color = expected direction." />
              </th>
              <th className="right">Low</th>
              <th className="right">High</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 150).map((t, i) => {
              const h = t.horizons[hz];
              return (
                <tr key={t.ticker} onClick={() => nav(`/forecast/${t.ticker}`)}>
                  <td className="mono" style={{ color: "var(--ink-faint)" }}>{i + 1}</td>
                  <td>
                    <div className="tk">{t.ticker}</div>
                    <div className="nm">{t.name}</div>
                  </td>
                  <td className="sec">{t.sector}</td>
                  <td className={`right mono ${signClass(h.q50)}`} style={{ fontSize: 15.5, fontWeight: 600 }}>{pct(h.q50)}</td>
                  <td><RangeBar h={h} scale={scale} /></td>
                  <td className="right mono" style={{ color: "var(--ink-lo)" }}>{pct(h.q10)}</td>
                  <td className="right mono" style={{ color: "var(--ink-lo)" }}>{pct(h.q90)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length > 150 && <p className="cap" style={{ marginTop: 12 }}>Showing the top 150 of {rows.length}. Narrow with search or sector.</p>}
    </>
  );
}
