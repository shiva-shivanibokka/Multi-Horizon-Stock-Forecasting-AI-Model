"use client";
import {
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";

export default function PriceChart({ techData }) {
  if (!techData?.dates?.length) return <p className="text-sm text-[hsl(var(--muted))]">No chart data.</p>;

  const points = techData.dates.map((d, i) => ({
    date:    new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    close:   Number(techData.close[i])   || null,
    sma50:   Number(techData.sma_50[i])  || null,
    sma200:  Number(techData.sma_200[i]) || null,
  }));

  const skip = Math.max(1, Math.floor(points.length / 8));

  return (
    <div className="h-80">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={points} margin={{ top: 5, right: 10, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 11, fill: "hsl(var(--muted))" }}
            interval={skip - 1}
            angle={-35}
            textAnchor="end"
            height={55}
          />
          <YAxis
            tick={{ fontSize: 11, fill: "hsl(var(--muted))" }}
            tickFormatter={(v) => `$${v}`}
            width={65}
          />
          <Tooltip
            contentStyle={{
              background: "hsl(var(--card))",
              border: "1px solid hsl(var(--border))",
              borderRadius: 8,
              fontSize: 12,
            }}
            formatter={(v, n) => [`$${Number(v).toFixed(2)}`, n]}
          />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          <Line type="monotone" dataKey="close"  name="Close"   stroke="#60a5fa" strokeWidth={2}   dot={false} />
          <Line type="monotone" dataKey="sma50"  name="SMA 50"  stroke="#34d399" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
          <Line type="monotone" dataKey="sma200" name="SMA 200" stroke="#fbbf24" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
