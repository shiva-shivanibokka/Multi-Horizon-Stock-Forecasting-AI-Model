import { useEffect, useState } from "react";

// Static JSON emitted by `python -m mhf.serve.export`. Fetched once, cached as
// promises so every view shares one download.
const BASE = import.meta.env.BASE_URL || "./";
const cache = {};

function load(name) {
  if (!cache[name]) {
    cache[name] = fetch(`${BASE}data/${name}`).then((r) => {
      if (!r.ok) throw new Error(`failed to load ${name} (${r.status})`);
      return r.json();
    });
  }
  return cache[name];
}

export const loadForecasts = () => load("forecasts.json");
export const loadMetrics = () => load("metrics.json");
export const loadPrices = () => load("prices.json");

export function useAsync(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    loader()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error, loading: false }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

// ---- formatting helpers (shared across views) ----
export const HORIZON_LABEL = { "1m": "1 month", "3m": "3 months", "6m": "6 months" };

export function pct(x, digits = 1) {
  if (x == null || Number.isNaN(x)) return "—";
  return `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`;
}

export function num(x, digits = 4) {
  if (x == null || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

export const signClass = (x) => (x == null ? "" : x >= 0 ? "pos" : "neg");
