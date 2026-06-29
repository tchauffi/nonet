/** @type {import('next').NextConfig} */

// On GitHub Pages the site is served from https://<user>.github.io/nonet/, so
// assets need the `/nonet` prefix. Locally (next dev) we want no prefix.
const isProd = process.env.NODE_ENV === 'production';
const basePath = isProd ? '/nonet' : '';

const nextConfig = {
  output: 'export',                 // static HTML/JS, no Node server (GitHub Pages)
  basePath,
  images: { unoptimized: true },    // no image optimizer in a static export
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  webpack: (config) => {
    // onnxruntime-web references node builtins it never uses in the browser.
    config.resolve.fallback = { ...config.resolve.fallback, fs: false, path: false, crypto: false };
    return config;
  },
};

export default nextConfig;
