import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "NBA Prediction Platform | AI-Powered Game & Player Projections",
  description:
    "Real-time NBA game predictions powered by XGBoost ML models. Daily matchup scores, spreads, totals, win probabilities, and player prop projections.",
  keywords: ["NBA predictions", "basketball analytics", "player props", "game spread", "AI sports"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr" className="dark">
      <body className={`${inter.variable} font-sans antialiased bg-[#070B14] text-slate-100 min-h-screen`}>
        {children}
      </body>
    </html>
  );
}
