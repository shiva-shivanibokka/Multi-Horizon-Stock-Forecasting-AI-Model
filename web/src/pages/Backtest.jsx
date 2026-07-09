import { useNavigate } from "react-router-dom";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip,
} from "recharts";
import { useAsync, loadBacktest, loadForecasts, pct, num, signClass } from "../lib/data.js";
import Info from "../components/Info.jsx";

const LINES = [
  { key: "long_short", label: "Long-short (top − bottom decile)", color: "#a78bff", w: 2.6 },
  { key: "long_only", label: "Long-only (top decile)", color: "#37c6ec", w: 1.8 },
  { key: "benchmark", label: "Equal-weight universe", color: "#8695b2", w: 1.8 },
];
const noSign = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const both = () => Promise.all([loadBacktest(), loadForecasts()]);

function Tip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="fan-tip" style={{ position: "static", transform: "none", minWidth: 220 }}>
      <div className="k">{label}</div>
      {payload.map((p) => (
        <div className="row" key={p.dataKey}>
          <span style={{ color: p.color }}>{LINES.find((l) => l.key === p.dataKey)?.label}</span>
          <b className="mono">{p.value.toFixed(2)}×</b>
        </div>
      ))}
    </div>
  );
}

function Tile({ k, v, note, info, cls = "" }) {
  return (
    <div className="panel tile">
      <div className="k">{k}<Info text={info} /></div>
      <div className={`v ${cls}`}>{v}</div>
      <div className="note">{note}</div>
    </div>
  );
}

function HoldList({ rows, nav }) {
  return (
    <div className="hold-list">
      {rows.map((t) => (
        <div className="hold-row" key={t.ticker} onClick={() => nav(`/forecast/${t.ticker}`)}>
          <div className="h-l">
            <span className="h-tk">{t.ticker}</span>
            <span className="h-nm">{t.name}</span>
          </div>
          <span className={`mono ${signClass(t.horizons["1m"].q50)}`} style={{ fontWeight: 600 }}>
            {pct(t.horizons["1m"].q50)}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function Backtest() {
  const { data, error, loading } = useAsync(both);
  const nav = useNavigate();
  if (loading) return <div className="state">Loading backtest…</div>;
  if (error) return <div className="state">Could not load backtest.json — run <code>python -m mhf.eval.backtest</code>.</div>;

  const [b, forecasts] = data;
  const ls = b.stats.long_short;
  const net10 = b.cost_sensitivity.find((c) => c.bps === 10);
  const data2 = b.dates.map((d, i) => ({
    date: d.slice(0, 7),
    long_short: b.equity.long_short[i],
    long_only: b.equity.long_only[i],
    benchmark: b.equity.benchmark[i],
  }));

  const ranked = [...forecasts.tickers].sort((a, c) => c.horizons["1m"].q50 - a.horizons["1m"].q50);
  const k = Math.max(1, Math.round(ranked.length * (b.decile || 0.1)));
  const longs = ranked.slice(0, k);
  const shorts = ranked.slice(ranked.length - k).reverse();

  return (
    <>
      <div className="page-head">
        <span className="eyebrow">Backtest</span>
        <h1>If you'd actually traded the signal</h1>
        <p className="lead">
          Each month, rank the S&amp;P 500 by the model's predicted 1-month return, go <b>long the top decile</b> and
          <b> short the bottom decile</b>, hold a month, repeat — using only out-of-sample, walk-forward predictions.
          Here's the equity curve, the risk-adjusted return, and whether the edge survives trading costs.
        </p>
      </div>

      <div className="tiles" style={{ marginBottom: 22 }}>
        <Tile k="Sharpe ratio" v={num(ls.sharpe, 2)} cls={ls.sharpe >= 0 ? "pos" : "neg"}
          note={net10 ? `${num(net10.sharpe, 2)} net of 10 bps/side` : ""}
          info="Annualized return divided by volatility — return per unit of risk. Above 1 is good, above 2 is strong. Shown before costs." />
        <Tile k="Annual return" v={pct(ls.cagr, 1)} cls={ls.cagr >= 0 ? "pos" : "neg"}
          note={`volatility ${noSign(ls.ann_vol)}`}
          info="Compound annual growth rate of the long-short book over the backtest window." />
        <Tile k="Max drawdown" v={pct(ls.max_drawdown, 1)} cls="neg"
          note={`${Math.round(ls.hit_rate * 100)}% winning months`}
          info="Largest peak-to-trough loss — the worst stretch you would have had to sit through." />
        <Tile k="Monthly turnover" v={noSign(ls.avg_turnover)}
          note="of the book, per rebalance"
          info="Share of holdings replaced each month. Higher turnover means the edge is more exposed to trading costs." />
      </div>

      <div className="panel chart-card">
        <h3>Growth of $1
          <Info text="Cumulative value of $1 invested at the start, out-of-sample. Long-short is market-neutral (top decile minus bottom decile); the benchmark is the equal-weight universe." />
        </h3>
        <p className="cap">{b.n_months} months · {b.period?.[0]} → {b.period?.[1]} · monthly rebalance · {b.signal}</p>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={data2} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#141d31" vertical={false} />
            <XAxis dataKey="date" tick={{ fill: "#aab6d2", fontFamily: "JetBrains Mono", fontSize: 12 }} axisLine={{ stroke: "#1c2740" }} tickLine={false} minTickGap={44} />
            <YAxis tick={{ fill: "#aab6d2", fontFamily: "JetBrains Mono", fontSize: 12 }} axisLine={false} tickLine={false} width={46} tickFormatter={(v) => `${v}×`} />
            <Tooltip content={<Tip />} />
            <ReferenceLine y={1} stroke="#5a688a" strokeDasharray="4 4" />
            {LINES.map((l) => (
              <Line key={l.key} type="monotone" dataKey={l.key} stroke={l.color} strokeWidth={l.w} dot={false} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
        <div className="legend">
          {LINES.map((l) => <span key={l.key}><i style={{ background: l.color }} />{l.label}</span>)}
        </div>
      </div>

      <div className="panel chart-card" style={{ marginTop: 18 }}>
        <h3>What it holds this month
          <Info text="The exact stocks the strategy would trade at the latest rebalance: long the top decile by predicted 1-month return, short the bottom decile. Click any name to open its forecast." />
        </h3>
        <p className="cap">Long the top {k} names, short the bottom {k}, ranked by the 1-month forecast (as of {forecasts.tickers[0]?.anchor_date}). The backtest repeats this every month.</p>
        <div className="holdings">
          <div>
            <div className="hold-head pos">▲ Long · top decile ({k})</div>
            <HoldList rows={longs} nav={nav} />
          </div>
          <div>
            <div className="hold-head neg">▼ Short · bottom decile ({k})</div>
            <HoldList rows={shorts} nav={nav} />
          </div>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(330px, 1fr))", marginTop: 18 }}>
        <div className="panel chart-card">
          <h3>Does the edge survive costs?
            <Info text="Transaction cost charged per side, in basis points. At each level, the long-short strategy's net Sharpe and annual return after paying to trade the monthly turnover." />
          </h3>
          <p className="cap">Charged per side, applied to monthly turnover.</p>
          <table className="tbl">
            <thead><tr><th>Cost / side</th><th className="right">Net Sharpe</th><th className="right">Net annual</th></tr></thead>
            <tbody>
              {b.cost_sensitivity.map((c) => (
                <tr key={c.bps} style={{ cursor: "default" }}>
                  <td className="tk">{c.bps} bps</td>
                  <td className="right mono">{num(c.sharpe, 2)}</td>
                  <td className={`right mono ${c.ann_return >= 0 ? "pos" : "neg"}`}>{pct(c.ann_return, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel chart-card">
          <h3>Read this honestly</h3>
          <div className="prose" style={{ fontSize: 15.5 }}>
            <p>The signal is the <strong>GBM cross-sectional model</strong>, evaluated strictly out-of-sample with purged, embargoed walk-forward folds — it never sees the future.</p>
            <p><strong>Caveats that matter:</strong> the universe is <em>today's</em> S&amp;P 500, so there is survivorship bias; shorting frictions and borrow costs beyond the simple per-side charge aren't modeled; monthly rebalancing ignores intramonth risk. Treat this as an honest research backtest, not a live track record.</p>
          </div>
        </div>
      </div>
    </>
  );
}
