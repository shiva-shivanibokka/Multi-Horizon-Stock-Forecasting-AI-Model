"use client";

const COLOR = {
  BUY:  { badge: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30", dot: "bg-emerald-400" },
  HOLD: { badge: "bg-yellow-500/15  text-yellow-400  border-yellow-500/30",  dot: "bg-yellow-400"  },
  SELL: { badge: "bg-red-500/15     text-red-400     border-red-500/30",     dot: "bg-red-400"     },
};

export default function Recommendation({ rec }) {
  if (!rec) return null;
  const style = COLOR[rec.recommendation] || COLOR.HOLD;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <span className={`inline-flex items-center gap-2 px-4 py-1.5 rounded-full border text-sm font-semibold ${style.badge}`}>
          <span className={`w-2 h-2 rounded-full ${style.dot}`} />
          {rec.recommendation}
        </span>
        <span className="text-sm text-[hsl(var(--muted))]">
          Confidence: <span className="text-white font-medium">{rec.confidence}</span>
          &nbsp;&nbsp;·&nbsp;&nbsp;Score: <span className="text-white font-medium">{rec.score}</span>
        </span>
      </div>

      {rec.reasons?.length > 0 && (
        <ul className="space-y-1.5">
          {rec.reasons.map((r, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-[hsl(var(--muted))]">
              <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-[hsl(var(--muted))] shrink-0" />
              {r}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
