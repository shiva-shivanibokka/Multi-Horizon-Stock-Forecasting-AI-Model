"use client";
import { fmt, pct } from "@/lib/utils";

const HORIZONS = {
  "1w": "1 Week",
  "1m": "1 Month",
  "6m": "6 Months",
};

export default function ForecastTable({ predictions, p10, p90, currentPrice }) {
  if (!predictions) return null;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[hsl(var(--border))]">
            <th className="text-left py-2 pr-6 text-[hsl(var(--muted))] font-medium">Horizon</th>
            <th className="text-right py-2 px-4 text-[hsl(var(--muted))] font-medium">p50 (median)</th>
            <th className="text-right py-2 px-4 text-[hsl(var(--muted))] font-medium">p10 (low)</th>
            <th className="text-right py-2 px-4 text-[hsl(var(--muted))] font-medium">p90 (high)</th>
            <th className="text-right py-2 pl-4 text-[hsl(var(--muted))] font-medium">Change</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(HORIZONS).map(([key, label]) => {
            const price  = predictions?.[key];
            const low    = p10?.[key];
            const high   = p90?.[key];
            const change = price != null ? ((price - currentPrice) / currentPrice) * 100 : null;
            const up     = change != null && change >= 0;

            return (
              <tr key={key} className="border-b border-[hsl(var(--border))] last:border-0">
                <td className="py-3 pr-6 font-medium">{label}</td>
                <td className="py-3 px-4 text-right font-mono">${fmt(price)}</td>
                <td className="py-3 px-4 text-right font-mono text-[hsl(var(--muted))]">
                  {low != null ? `$${fmt(low)}` : "—"}
                </td>
                <td className="py-3 px-4 text-right font-mono text-[hsl(var(--muted))]">
                  {high != null ? `$${fmt(high)}` : "—"}
                </td>
                <td className={`py-3 pl-4 text-right font-mono font-medium ${up ? "text-emerald-400" : "text-red-400"}`}>
                  {change != null ? pct(change) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
