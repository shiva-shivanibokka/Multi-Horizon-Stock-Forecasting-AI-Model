// Categorical model colors — fixed order, validated CVD-safe on the dark surface
// (dataviz validator: lightness/chroma/CVD/contrast all pass). Never cycle or
// reassign by rank; a model keeps its hue everywhere.
export const MODELS = ["random_walk", "baseline", "gbm", "chronos", "ensemble"];

export const MODEL_COLOR = {
  random_walk: "#4784c9",
  baseline: "#1f9a6c",
  gbm: "#b8862a",
  chronos: "#c164a6",
  ensemble: "#8264cc",
};

export const MODEL_LABEL = {
  random_walk: "Random walk",
  baseline: "Baseline",
  gbm: "GBM",
  chronos: "Chronos",
  ensemble: "Ensemble",
};

// Semantic (non-categorical) encodings, from styles.css tokens.
export const SIGNAL = "#f5c451"; // median needle / brand
export const BAND = "#56c7dc"; // uncertainty band
export const POS = "#46c7a8"; // positive return (diverging warm-pole)
export const NEG = "#e8746a"; // negative return (diverging cool-pole)

// trading-day offset of each horizon (mirrors settings.horizons on the Python side)
export const HORIZON_DAYS = { "1m": 21, "3m": 63, "6m": 126 };
