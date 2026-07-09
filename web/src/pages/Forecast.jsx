import { useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAsync, loadForecasts, loadPrices, pct, signClass, HORIZON_LABEL } from "../lib/data.js";
import FanChart from "../components/FanChart.jsx";

const HZ = ["1m", "3m", "6m"];
const both = () => Promise.all([loadForecasts(), loadPrices()]);

export default function Forecast() {
  const { data, error, loading } = useAsync(both);
  const { ticker } = useParams();
  const nav = useNavigate();

  const [forecasts, prices] = data || [];
  const byTicker = useMemo(() => {
    const map = {};
    forecasts?.tickers.forEach((t) => (map[t.ticker] = t));
    return map;
  }, [forecasts]);

  if (loading) return <div className="state">Loading forecasts…</div>;
  if (error) return <div className="state">Could not load data — run the export step.</div>;

  const sel = byTicker[ticker?.toUpperCase()] || forecasts.tickers[0];
  const hist = prices[sel.ticker];

  return (
    <>
      <div className="page-head" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 20, flexWrap: "wrap" }}>
        <div>
          <span className="eyebrow">{sel.sector} · anchor {sel.anchor_date}</span>
          <h1 style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
            <span className="mono">{sel.ticker}</span>
            <span className="mono" style={{ fontSize: 18, color: "var(--ink-lo)" }}>${sel.anchor_price.toFixed(2)}</span>
          </h1>
        </div>
        <input
          className="field" list="tickers" defaultValue="" placeholder={`Jump to ticker (e.g. ${sel.ticker})`}
          onChange={(e) => { const v = e.target.value.toUpperCase(); if (byTicker[v]) nav(`/forecast/${v}`); }}
          aria-label="select ticker" style={{ width: 220 }}
        />
        <datalist id="tickers">{forecasts.tickers.map((t) => <option key={t.ticker} value={t.ticker} />)}</datalist>
      </div>

      <div className="panel chart-card">
        <h3>Projected price distribution</h3>
        <p className="cap">Recent close, then the ensemble's 10–90% return band converted to price and widened over the horizon. The gold line is the median path.</p>
        {hist
          ? <FanChart history={hist} anchorPrice={sel.anchor_price} horizons={sel.horizons} />
          : <div className="state">No price history for {sel.ticker}.</div>}
        <div className="legend">
          <span><i style={{ background: "#f5c451" }} />median</span>
          <span><i style={{ background: "#56c7dc" }} />10–90% band</span>
          <span><i style={{ background: "#7b89a3" }} />history</span>
        </div>
      </div>

      <div className="tiles" style={{ marginTop: 16 }}>
        {HZ.map((h) => {
          const q = sel.horizons[h];
          return (
            <div className="panel tile" key={h}>
              <div className="k">{HORIZON_LABEL[h]}</div>
              <div className={`v ${signClass(q.q50)}`}>{pct(q.q50)}</div>
              <div className="note mono">band {pct(q.q10)} … {pct(q.q90)}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}
