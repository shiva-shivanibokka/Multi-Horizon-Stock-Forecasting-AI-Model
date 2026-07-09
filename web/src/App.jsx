import { HashRouter, NavLink, Routes, Route, Navigate } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import Screener from "./pages/Screener.jsx";
import Forecast from "./pages/Forecast.jsx";
import Method from "./pages/Method.jsx";

const NAV = [
  { to: "/", label: "Performance", end: true },
  { to: "/screener", label: "Screener" },
  { to: "/forecast", label: "Forecast" },
  { to: "/method", label: "Method" },
];

function Rail() {
  return (
    <aside className="rail">
      <div className="brand">
        <span className="mark">MHF<b>.</b></span>
        <span className="sub">Probabilistic Forecaster</span>
      </div>
      <nav className="nav">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.end}>
            {n.label}
          </NavLink>
        ))}
      </nav>
      <div className="foot">
        S&amp;P 500 · 1m/3m/6m<br />
        distributions, not points
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <HashRouter>
      <div className="app">
        <Rail />
        <main className="main">
          <div className="wrap">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/screener" element={<Screener />} />
              <Route path="/forecast" element={<Forecast />} />
              <Route path="/forecast/:ticker" element={<Forecast />} />
              <Route path="/method" element={<Method />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>
    </HashRouter>
  );
}
