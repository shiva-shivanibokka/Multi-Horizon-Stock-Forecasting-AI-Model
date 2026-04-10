"use client";

const LABEL_STYLE = {
  positive: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  negative: "text-red-400     bg-red-500/10     border-red-500/30",
  neutral:  "text-yellow-400  bg-yellow-500/10  border-yellow-500/30",
};

export default function Sentiment({ data }) {
  if (!data) return <p className="text-sm text-[hsl(var(--muted))]">No sentiment data.</p>;

  const style = LABEL_STYLE[data.sentiment] || LABEL_STYLE.neutral;
  const score = Number(data.score);
  const barW  = Math.round(((score + 1) / 2) * 100);

  return (
    <div className="space-y-6">
      {/* Aggregate */}
      <div className="flex flex-wrap items-center gap-4">
        <span className={`px-3 py-1 rounded-full border text-sm font-semibold capitalize ${style}`}>
          {data.sentiment}
        </span>
        <div className="flex-1 min-w-40">
          <div className="flex justify-between text-xs text-[hsl(var(--muted))] mb-1">
            <span>Negative</span>
            <span>Score: {score.toFixed(3)}</span>
            <span>Positive</span>
          </div>
          <div className="h-2 rounded-full bg-[hsl(var(--border))]">
            <div
              className="h-2 rounded-full bg-blue-500 transition-all"
              style={{ width: `${barW}%` }}
            />
          </div>
        </div>
        <p className="text-xs text-[hsl(var(--muted))]">{data.articles_analyzed} articles analyzed</p>
      </div>

      {/* Articles */}
      <div className="space-y-3 max-h-80 overflow-y-auto pr-1">
        {data.articles?.map((a, i) => {
          const s = Number(a.sentiment_score);
          const c = s > 0.05 ? "text-emerald-400" : s < -0.05 ? "text-red-400" : "text-yellow-400";
          return (
            <div key={i} className="border border-[hsl(var(--border))] rounded-lg p-3 space-y-1">
              <div className="flex items-start justify-between gap-3">
                <a
                  href={a.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium hover:text-blue-400 transition-colors"
                >
                  {a.title}
                </a>
                <span className={`text-xs font-mono font-semibold shrink-0 ${c}`}>
                  {s >= 0 ? "+" : ""}{s.toFixed(3)}
                </span>
              </div>
              {a.summary && (
                <p className="text-xs text-[hsl(var(--muted))] line-clamp-2">{a.summary}</p>
              )}
              <p className="text-xs text-[hsl(var(--muted))]">{a.date}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
