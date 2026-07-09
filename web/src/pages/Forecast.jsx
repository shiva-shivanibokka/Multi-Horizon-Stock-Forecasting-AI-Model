import { useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAsync, loadForecasts, loadPrices, pct, signClass, HORIZON_LABEL } from "../lib/data.js";
import FanChart from "../components/FanChart.jsx";
import Info from "../components/Info.jsx";

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

  const sel = byTicker[ticker?.toUpperCase()] || byTicker.AAPL || forecasts.tickers[0];
  const hist = prices[sel.ticker];

  return (
    <>
      <div className="page-head">
        <span className="eyebrow">Forecast</span>
        <h1>Where might a stock go — and how sure are we?</h1>
        <p className="lead">
          Pick any S&amp;P 500 company. The chart projects a <b>range</b> of possible returns at 1, 3, and 6 months —
          not a single guess. The gold line is the most likely path; the shaded band is where the price should land
          <b> 80% of the time</b>. A wider band means more uncertainty.
        </p>
      </div>

      <div className="controls" style={{ justifyContent: "space-between" }}>
        <div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
            <span className="mono" style={{ fontFamily: "var(--font-display)", fontWeight: 800, fontSize: 34, color: "var(--ink-hi)" }}>{sel.ticker}</span>
            <span className="mono" style={{ fontSize: 20, color: "var(--brand)" }}>${sel.anchor_price.toFixed(2)}</span>
          </div>
          <div style={{ color: "var(--ink-lo)", fontSize: 16, marginTop: 2 }}>{sel.name} · {sel.sector}</div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <input
            className="field" list="tickers" defaultValue="" placeholder="Search a ticker…"
            onChange={(e) => { const v = e.target.value.toUpperCase(); if (byTicker[v]) nav(`/forecast/${v}`); }}
            aria-label="select ticker" style={{ width: 240 }}
          />
          <datalist id="tickers">{forecasts.tickers.map((t) => <option key={t.ticker} value={t.ticker}>{t.name}</option>)}</datalist>
          <span style={{ fontSize: 12.5, color: "var(--ink-faint)" }}>anchored {sel.anchor_date}</span>
        </div>
      </div>

      <div className="panel chart-card">
        <h3>Projected price range
          <Info text="Left of the dashed line is the real recent price. Right of it is the model's forecast: the gold line is the median outcome and the cyan band spans the 10th–90th percentile of returns, converted to price." />
        </h3>
        <p className="cap">Hover the 1M / 3M / 6M markers to read the exact numbers.</p>
        {hist
          ? <FanChart history={hist} anchorPrice={sel.anchor_price} horizons={sel.horizons} />
          : <div className="state">No price history for {sel.ticker}.</div>}
        <div className="legend">
          <span><i style={{ background: "var(--signal)" }} />median (most likely)</span>
          <span><i style={{ background: "var(--band)" }} />10–90% range</span>
          <span><i style={{ background: "var(--ink-lo)" }} />recent price</span>
        </div>
      </div>

      <div className="tiles" style={{ marginTop: 18 }}>
        {HZ.map((h) => {
          const q = sel.horizons[h];
          return (
            <div className="panel tile" key={h}>
              <div className="k">Expected in {HORIZON_LABEL[h]}
                <Info text={`Median forecast return over ${HORIZON_LABEL[h]}. The band below is the 10–90% range — the realized return should fall inside it 80% of the time.`} />
              </div>
              <div className={`v ${signClass(q.q50)}`}>{pct(q.q50)}</div>
              <div className="note mono">range {pct(q.q10)} to {pct(q.q90)}</div>
            </div>
          );
        })}
      </div>
    </>
  );
}
