import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      { source: "/auth/:path*", destination: "http://localhost:8000/auth/:path*" },
      { source: "/api/:path*", destination: "http://localhost:8000/api/:path*" },
      { source: "/voice/:path*", destination: "http://localhost:8000/voice/:path*" },
    ];
  },
};

export default nextConfig;
