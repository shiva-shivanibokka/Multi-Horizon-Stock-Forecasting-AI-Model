"use client";

function Row({ label, value }) {
  if (value == null || value === "N/A" || value === 0) return null;
  return (
    <div className="flex justify-between py-2 border-b border-[hsl(var(--border))] last:border-0 text-sm">
      <span className="text-[hsl(var(--muted))]">{label}</span>
      <span className="font-medium">{String(value)}</span>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div className="rounded-lg border border-[hsl(var(--border))] p-4">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-[hsl(var(--muted))] mb-3">
        {title}
      </h3>
      {children}
    </div>
  );
}

export default function Fundamentals({ data }) {
  if (!data || data.error)
    return <p className="text-sm text-[hsl(var(--muted))]">{data?.error || "No fundamental data."}</p>;

  const info = data.company_info || {};
  const val  = data.valuation    || {};
  const fin  = data.financials   || {};
  const grow = data.growth       || {};
  const trd  = data.trading      || {};

  return (
    <div className="space-y-4">
      {info.name && (
        <div className="mb-2">
          <p className="font-semibold text-base">{info.name}</p>
          <p className="text-sm text-[hsl(var(--muted))]">
            {[info.sector, info.industry, info.country].filter(Boolean).join(" · ")}
          </p>
          {info.website && info.website !== "N/A" && (
            <a href={info.website} target="_blank" rel="noopener noreferrer"
               className="text-xs text-blue-400 hover:underline">
              {info.website}
            </a>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <Section title="Valuation">
          <Row label="Market Cap"     value={info.market_cap} />
          <Row label="P/E Ratio"      value={val.pe_ratio} />
          <Row label="Forward P/E"    value={val.forward_pe} />
          <Row label="Price / Book"   value={val.price_to_book} />
          <Row label="Price / Sales"  value={val.price_to_sales} />
        </Section>

        <Section title="Financials">
          <Row label="Revenue"        value={fin.revenue} />
          <Row label="Net Income"     value={fin.net_income} />
          <Row label="Cash"           value={fin.total_cash} />
          <Row label="Debt"           value={fin.total_debt} />
          <Row label="Debt / Equity"  value={fin.debt_equity} />
          <Row label="Current Ratio"  value={fin.current_ratio} />
        </Section>

        <Section title="Growth & Trading">
          <Row label="Revenue Growth"   value={grow.revenue_growth  != null ? `${grow.revenue_growth}%`  : null} />
          <Row label="Earnings Growth"  value={grow.earnings_growth != null ? `${grow.earnings_growth}%` : null} />
          <Row label="Profit Margin"    value={grow.profit_margin   != null ? `${grow.profit_margin}%`   : null} />
          <Row label="Beta"             value={trd.beta} />
          <Row label="52-Week Change"   value={trd["52w_change_pct"] != null ? `${trd["52w_change_pct"]}%` : null} />
          <Row label="Dividend Yield"   value={trd.dividend_yield_pct != null ? `${trd.dividend_yield_pct}%` : null} />
        </Section>
      </div>
    </div>
  );
}
