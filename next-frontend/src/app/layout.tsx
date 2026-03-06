import type { Metadata } from "next";
import { Outfit, Inter } from "next/font/google";
import { Toaster } from "react-hot-toast";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Kaudal - Inversión Inteligente",
  description: "Portafolio Markowitz Trimestral",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body
        className={`${inter.variable} ${outfit.variable} font-sans antialiased`}
      >
        <Toaster />
        {children}
      </body>
    </html >
  );
}
