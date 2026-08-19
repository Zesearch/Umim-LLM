import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "UMIM — Distilling Sequential Computation",
  description:
    "UMIM distills multi-token computation into compact surrogate embeddings for efficient Transformer inference.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
