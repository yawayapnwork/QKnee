/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL: (process.env.NEXT_PUBLIC_API_URL || "https://qknee-8dv8.onrender.com").trim(),
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
