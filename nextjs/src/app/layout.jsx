import "./globals.css";

export const metadata = {
  title: "Stock Prediction Dashboard",
  description: "Multi-horizon stock forecasting with Transformer, LSTM, RNN and Random Forest models.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
