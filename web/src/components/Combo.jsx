import { useEffect, useRef, useState } from "react";

// Styled, searchable dropdown that matches the dark theme (native <select> can't
// be styled and renders an OS-light option list). options: [{value, tag?, label, sub?}].
export default function Combo({ value, options, onChange, searchable = true, placeholder = "Search…", width = 260 }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [hi, setHi] = useState(0);
  const ref = useRef(null);
  const searchRef = useRef(null);

  const sel = options.find((o) => o.value === value);
  const ql = q.trim().toLowerCase();
  const filtered = ql
    ? options.filter((o) =>
        (o.tag || "").toLowerCase().includes(ql) ||
        o.label.toLowerCase().includes(ql) ||
        (o.sub || "").toLowerCase().includes(ql))
    : options;

  useEffect(() => {
    function onDoc(e) { if (ref.current && !ref.current.contains(e.target)) setOpen(false); }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  useEffect(() => {
    if (open && searchable) searchRef.current?.focus();
    setHi(0);
  }, [open, ql, searchable]);

  function choose(o) { onChange(o.value); setOpen(false); setQ(""); }

  function onKey(e) {
    if (e.key === "ArrowDown") { e.preventDefault(); setHi((h) => Math.min(h + 1, filtered.length - 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setHi((h) => Math.max(h - 1, 0)); }
    else if (e.key === "Enter") { e.preventDefault(); if (filtered[hi]) choose(filtered[hi]); }
    else if (e.key === "Escape") { setOpen(false); }
  }

  return (
    <div className={`combo${open ? " open" : ""}`} ref={ref} style={{ width }} onKeyDown={onKey}>
      <button type="button" className="combo-btn" onClick={() => setOpen((o) => !o)} aria-haspopup="listbox" aria-expanded={open}>
        <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {sel ? (sel.tag ? `${sel.tag} · ${sel.label}` : sel.label) : placeholder}
        </span>
        <svg className="chev" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
          <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div className="combo-panel" role="listbox">
          {searchable && (
            <input ref={searchRef} className="combo-search" placeholder={placeholder} value={q}
              onChange={(e) => setQ(e.target.value)} aria-label="search options" />
          )}
          <div className="combo-list">
            {filtered.length === 0 && <div className="combo-empty">No matches</div>}
            {filtered.slice(0, 300).map((o, i) => (
              <div key={o.value} role="option" aria-selected={o.value === value}
                className={`combo-opt${o.value === value ? " sel" : ""}${i === hi ? " hi" : ""}`}
                onMouseEnter={() => setHi(i)} onClick={() => choose(o)}>
                <span className="o1">{o.tag && <b>{o.tag}</b>}{o.label}</span>
                {o.sub && <span className="o2">{o.sub}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
