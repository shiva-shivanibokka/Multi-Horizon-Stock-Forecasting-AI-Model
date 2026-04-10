/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card:       "hsl(var(--card))",
        border:     "hsl(var(--border))",
        muted:      "hsl(var(--muted))",
        accent:     "hsl(var(--accent))",
      },
    },
  },
  plugins: [],
};
