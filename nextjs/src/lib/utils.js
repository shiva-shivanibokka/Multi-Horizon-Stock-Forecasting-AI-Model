import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export function fmt(n, decimals = 2) {
  if (n == null) return "—";
  return Number(n).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function pct(n) {
  if (n == null) return "—";
  const v = Number(n);
  return (v >= 0 ? "+" : "") + v.toFixed(2) + "%";
}
