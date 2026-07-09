// Brief hover/focus "?" explainer. Keyboard-accessible (tabbable, shows on focus).
export default function Info({ text }) {
  return (
    <span className="info" tabIndex={0} role="note" aria-label={text}>
      ?<span className="info-pop">{text}</span>
    </span>
  );
}
