/** @type {import('next').NextConfig} */
const nextConfig = {
  // Proxy all /api calls to the Flask backend.
  // In production on Render, set NEXT_PUBLIC_API_URL to the Flask service URL.
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000"}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
