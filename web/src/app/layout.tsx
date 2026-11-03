import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "AnamnesisLM · Clinical note review",
  description: "Local synthetic clinical note extraction demo.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
