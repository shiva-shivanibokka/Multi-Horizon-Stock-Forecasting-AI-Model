import { useAsync, loadMetrics, num } from "../lib/data.js";

export default function Method() {
  const { data: m } = useAsync(loadMetrics);
  const ens6 = m?.ensemble?.y_6m;

  return (
    <>
      <div className="page-head">
        <span className="eyebrow">How it works</span>
        <h1>Method</h1>
        <p>A probabilistic forecaster: for each stock it predicts a <em>distribution</em> of returns at 1, 3, and 6 months — not a single number.</p>
      </div>

      <div className="prose">
        <h2>Reading a forecast</h2>
        <p>
          Every forecast is three numbers per horizon: the 10th, 50th, and 90th percentile of the
          return. The median (50th) is the central estimate; the 10–90% band is the range the model
          expects the outcome to fall in 80% of the time. A wide band means high uncertainty — it is
          not a weak signal, it is an honest one.
        </p>

        <h2>The models</h2>
        <p>
          Two <strong>baselines</strong> set the bar: a random walk (flat forecast) and the historical
          return distribution. A <strong>gradient-boosted model</strong> (LightGBM quantile regression)
          learns from 35 technical and market features. A zero-shot <strong>Chronos-Bolt</strong>
          foundation model reads each price series directly. The <strong>ensemble</strong> blends GBM
          and Chronos with a weight fit <em>per horizon</em>, then applies conformal calibration to
          correct the bands.
        </p>
        <div className="callout">
          Chronos is used <strong>zero-shot</strong>, on purpose. Fine-tuning it on price levels
          collapsed its long-horizon forecasts; the out-of-the-box model is well-calibrated, so the
          ensemble backs it and calibration handles the rest.
        </div>

        <h2>How it's validated</h2>
        <p>
          Everything is scored with purged, embargoed <strong>walk-forward cross-validation</strong> —
          the model only ever sees the past. The blend weight and the conformal correction are fit on a
          validation fold and never on the test rows they're scored against, so the reported numbers
          aren't leaking their own answers.
        </p>

        <h2>What the numbers mean — honestly</h2>
        <p>
          Daily equity returns are close to unpredictable, so the edge here is small by nature. The
          ensemble's 6-month information coefficient is {ens6 ? <code>{num(ens6.ic, 3)}</code> : "small"} —
          a weak but real rank signal, with band coverage of {ens6 ? <code>{num(ens6.coverage, 2)}</code> : "~0.80"} against
          an 0.80 target. The unconditional baseline is genuinely hard to beat on calibration; the learned
          models earn their keep on <em>discrimination</em> (ranking stocks), which is what the screener uses.
        </p>
        <div className="callout">
          Research project, not investment advice. Forecasts are as of the last training anchor and are
          not updated live.
        </div>
      </div>
    </>
  );
}
