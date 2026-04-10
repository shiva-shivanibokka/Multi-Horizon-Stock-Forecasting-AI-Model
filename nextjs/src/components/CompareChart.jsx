"use client";
import {
  ResponsiveContainer, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";

const HORIZON_LABELS = {
  "1d": "1 Day",
  "1w": "1 Week",
  "1m": "1 Month",
  "6m": "6 Months",
  "1y": "1 Year",
};

const COLORS = {
  transformer: "#60a5fa",
  lstm:        "#34d399",
  rnn:         "#fbbf24",
  rf:          "#c084fc",
};

export default function CompareChart({ allData }) {
  if (!allData?.predictions) return null;

  const rows = Object.keys(HORIZON_LABELS).map((h) => {
    const row = { horizon: HORIZON_LABELS[h] };
    Object.keys(allData.predictions).forEach((m) => {
      const p = allData.predictions[m]?.p50?.[h];
      if (p != null) row[m] = Number(p);
    });
    return row;
  });

  const models = Object.keys(allData.predictions);

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="horizon" tick={{ fontSize: 12, fill: "hsl(var(--muted))" }} />
          <YAxis
            tick={{ fontSize: 12, fill: "hsl(var(--muted))" }}
            tickFormatter={(v) => `$${v}`}
            width={70}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v, n) => [`$${Number(v).toFixed(2)}`, n.toUpperCase()]}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 8 }}
            formatter={(v) => v.toUpperCase()}
          />
          {models.map((m) => (
            <Bar key={m} dataKey={m} name={m} fill={COLORS[m] || "#94a3b8"} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
