import { HashRouter, NavLink, Routes, Route, Navigate } from "react-router-dom";
import Forecast from "./pages/Forecast.jsx";
import Screener from "./pages/Screener.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Backtest from "./pages/Backtest.jsx";
import Method from "./pages/Method.jsx";

const NAV = [
  { to: "/", label: "Forecast", end: true },
  { to: "/screener", label: "Screener" },
  { to: "/backtest", label: "Backtest" },
  { to: "/performance", label: "Performance" },
  { to: "/method", label: "Method" },
];

function Logo() {
  return (
    <div className="logo">
      <svg className="glyph" viewBox="0 0 30 30" aria-hidden="true">
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#b061ff" />
            <stop offset="0.5" stopColor="#6b74ff" />
            <stop offset="1" stopColor="#29d3f0" />
          </linearGradient>
        </defs>
        <path d="M5 15 L27 4 L27 26 Z" fill="url(#lg)" opacity="0.92" />
        <path d="M5 15 L27 15" stroke="#f0f4fc" strokeWidth="2" strokeLinecap="round" opacity="0.9" />
        <circle cx="5" cy="15" r="2.6" fill="#f0f4fc" />
      </svg>
      <div>
        <div className="word grad-text">MHF</div>
        <div className="tag">Forecaster</div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <HashRouter>
      <div className="aurora" aria-hidden="true"><span /><span /><span /></div>
      <header className="header">
        <Logo />
        <nav className="topnav">
          {NAV.map((n) => (
            <NavLink key={n.to} to={n.to} end={n.end}>{n.label}</NavLink>
          ))}
        </nav>
        <span className="spacer" />
        <span className="env"><span className="live-dot" />497 companies · 1m / 3m / 6m</span>
      </header>
      <main className="main">
        <div className="wrap">
          <Routes>
            <Route path="/" element={<Forecast />} />
            <Route path="/forecast/:ticker" element={<Forecast />} />
            <Route path="/screener" element={<Screener />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/performance" element={<Dashboard />} />
            <Route path="/method" element={<Method />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </main>
    </HashRouter>
  );
}
