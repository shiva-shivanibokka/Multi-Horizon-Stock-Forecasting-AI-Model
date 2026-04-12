"use client";

import { useState } from "react";
import axios from "axios";
import { Search, TrendingUp, Loader2 } from "lucide-react";
import PriceChart     from "@/components/PriceChart";
import ForecastTable  from "@/components/ForecastTable";
import CompareChart   from "@/components/CompareChart";
import Sentiment      from "@/components/Sentiment";
import Fundamentals   from "@/components/Fundamentals";
import Recommendation from "@/components/Recommendation";
import { cn }         from "@/lib/utils";

const MODELS = ["transformer", "lstm", "rnn", "rf"];
const TABS   = ["Forecast", "Comparison", "Chart", "Sentiment", "Fundamentals"];

export default function Home() {
  const [ticker,       setTicker]       = useState("");
  const [model,        setModel]        = useState("transformer");
  const [tab,          setTab]          = useState("Forecast");
  const [data,         setData]         = useState(null);
  const [allData,      setAllData]      = useState(null);
  const [sentiment,    setSentiment]    = useState(null);
  const [fundamentals, setFundamentals] = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState("");

  async function search() {
    if (!ticker.trim()) { setError("Enter a ticker symbol."); return; }
    setLoading(true);
    setError("");
    setData(null);
    setAllData(null);
    setSentiment(null);
    setFundamentals(null);

    try {
      const [single, all, sent, fund] = await Promise.all([
        axios.get(`/api/predict/${ticker.trim()}?model=${model}`),
        axios.get(`/api/predict/all/${ticker.trim()}`),
        axios.get(`/api/sentiment/${ticker.trim()}`),
        axios.get(`/api/fundamentals/${ticker.trim()}`),
      ]);
      if (single.data.error) { setError(single.data.error); return; }
      setData(single.data);
      setAllData(all.data);
      setSentiment(sent.data);
      setFundamentals(fund.data);
      setTab("Forecast");
    } catch (e) {
      setError(e.response?.data?.error || "Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[hsl(var(--background))]">

      {/* Top bar */}
      <header className="border-b border-[hsl(var(--border))] bg-[hsl(var(--card))]">
        <div className="max-w-7xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <TrendingUp className="w-6 h-6 text-blue-400" />
            <span className="text-lg font-semibold tracking-tight">
              Stock Prediction Dashboard
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Model selector */}
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-[hsl(var(--background))] border border-[hsl(var(--border))] text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              {MODELS.map((m) => (
                <option key={m} value={m}>
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </option>
              ))}
            </select>

            {/* Ticker input */}
            <div className="relative">
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                onKeyDown={(e) => e.key === "Enter" && search()}
                placeholder="AAPL"
                className="w-40 bg-[hsl(var(--background))] border border-[hsl(var(--border))] rounded-lg px-4 py-2 pr-10 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <button
                onClick={search}
                disabled={loading}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-[hsl(var(--muted))] hover:text-blue-400 disabled:opacity-40 transition-colors"
              >
                {loading
                  ? <Loader2 className="w-4 h-4 animate-spin" />
                  : <Search className="w-4 h-4" />
                }
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">

        {/* Error */}
        {error && (
          <div className="mb-6 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-24 gap-3 text-[hsl(var(--muted))]">
            <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
            <p className="text-sm">Fetching data and running models...</p>
          </div>
        )}

        {data && !loading && (
          <>
            {/* Ticker header */}
            <div className="mb-6 flex flex-wrap items-end justify-between gap-2">
              <div>
                <h1 className="text-4xl font-bold tracking-tight">{data.ticker}</h1>
                <p className="text-[hsl(var(--muted))] text-sm mt-1">
                  Last close&nbsp;
                  <span className="text-white font-medium">${data.current_price}</span>
                  &nbsp;&nbsp;·&nbsp;&nbsp;
                  Model&nbsp;
                  <span className="text-blue-400 font-medium capitalize">{data.model}</span>
                </p>
              </div>
              <p className="text-xs text-[hsl(var(--muted))]">
                {new Date().toLocaleDateString("en-US", { dateStyle: "medium" })}
              </p>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 border-b border-[hsl(var(--border))] mb-6">
              {TABS.map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={cn(
                    "px-4 py-2 text-sm font-medium rounded-t-lg transition-colors",
                    tab === t
                      ? "border border-b-[hsl(var(--card))] border-[hsl(var(--border))] bg-[hsl(var(--card))] text-white"
                      : "text-[hsl(var(--muted))] hover:text-white"
                  )}
                >
                  {t}
                </button>
              ))}
            </div>

            {/* Tab panels */}
            {tab === "Forecast" && (
              <div className="grid gap-6">
                <Card title="Price Forecasts">
                  <ForecastTable
                    predictions={data.predictions}
                    p10={data.p10}
                    p90={data.p90}
                    currentPrice={data.current_price}
                  />
                  {data.p10 && (
                    <p className="mt-3 text-xs text-[hsl(var(--muted))]">
                      p10 / p90 are Monte Carlo Dropout confidence bounds (50 forward passes).
                    </p>
                  )}
                </Card>
                <Card title="Recommendation">
                  <Recommendation rec={data.recommendation} />
                </Card>
              </div>
            )}

            {tab === "Comparison" && allData && (
              <Card title="All 4 Models — p50 Forecasts">
                <CompareChart allData={allData} />
              </Card>
            )}

            {tab === "Chart" && (
              <Card title="Price History with Moving Averages">
                <PriceChart techData={data.technical_data} />
              </Card>
            )}

            {tab === "Sentiment" && (
              <Card title="News Sentiment">
                <Sentiment data={sentiment} />
              </Card>
            )}

            {tab === "Fundamentals" && (
              <Card title="Company Fundamentals">
                <Fundamentals data={fundamentals} />
              </Card>
            )}
          </>
        )}

        {/* Empty state */}
        {!data && !loading && !error && (
          <div className="flex flex-col items-center justify-center py-32 gap-3 text-[hsl(var(--muted))]">
            <TrendingUp className="w-12 h-12" />
            <p className="text-sm">Enter a ticker symbol above to get started.</p>
          </div>
        )}
      </main>
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div className="rounded-xl border border-[hsl(var(--border))] bg-[hsl(var(--card))] p-6">
      <h2 className="text-base font-semibold mb-4">{title}</h2>
      {children}
    </div>
  );
}
