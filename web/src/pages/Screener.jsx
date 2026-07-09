import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAsync, loadForecasts, pct, signClass, HORIZON_LABEL } from "../lib/data.js";
import RangeBar from "../components/RangeBar.jsx";

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
      (t) => (sector === "All" || t.sector === sector) && (!ql || t.ticker.includes(ql))
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
        <span className="eyebrow">{data.tickers.length} tickers · anchor {data.tickers[0]?.anchor_date}</span>
        <h1>Screener</h1>
        <p>Expected {HORIZON_LABEL[hz]} return per stock, ranked by the ensemble median. The bar shows the full 10–90% band — width is uncertainty, not conviction.</p>
      </div>

      <div className="controls">
        <div className="seg" role="tablist" aria-label="horizon">
          {HORIZONS.map((h) => (
            <button key={h} className={h === hz ? "on" : ""} onClick={() => setHz(h)}>{h.toUpperCase()}</button>
          ))}
        </div>
        <select className="field" value={sector} onChange={(e) => setSector(e.target.value)} aria-label="sector">
          {sectors.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <input className="field" placeholder="Search ticker…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="search ticker" style={{ flex: "1 1 160px", maxWidth: 220 }} />
        <span className="mono" style={{ color: "var(--ink-faint)", fontSize: 12 }}>{rows.length} match</span>
      </div>

      <div className="panel" style={{ overflowX: "auto" }}>
        <table className="tbl">
          <thead>
            <tr>
              <th style={{ width: 40 }}>#</th>
              <th>Ticker</th>
              <th>Sector</th>
              <th className="right" onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}>
                Median <span className="arw">{sortDir === "desc" ? "▼" : "▲"}</span>
              </th>
              <th>10–90% band</th>
              <th className="right">Low</th>
              <th className="right">High</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 120).map((t, i) => {
              const h = t.horizons[hz];
              return (
                <tr key={t.ticker} onClick={() => nav(`/forecast/${t.ticker}`)}>
                  <td className="mono" style={{ color: "var(--ink-faint)" }}>{i + 1}</td>
                  <td className="tk">{t.ticker}</td>
                  <td className="sec">{t.sector}</td>
                  <td className={`right mono ${signClass(h.q50)}`}>{pct(h.q50)}</td>
                  <td><RangeBar h={h} scale={scale} /></td>
                  <td className="right mono" style={{ color: "var(--ink-lo)" }}>{pct(h.q10)}</td>
                  <td className="right mono" style={{ color: "var(--ink-lo)" }}>{pct(h.q90)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length > 120 && <p className="cap" style={{ marginTop: 10 }}>Showing top 120 of {rows.length}. Narrow with search or sector.</p>}
    </>
  );
}
