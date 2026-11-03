import type { NextConfig } from "next";

const config: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  // Keep the Next.js workspace independent of the Python project.
  turbopack: { root: process.cwd() },
};

export default config;
