import type { Metadata } from "next";
import { Outfit, Inter } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";
import HelpWidget from "@/components/HelpWidget";
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
  title: "Kaudal - Aprendizaje Financiero Inteligente",
  description: "Portafolio Markowitz Trimestral",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body
        suppressHydrationWarning
        className={`${inter.variable} ${outfit.variable} font-sans antialiased`}
      >
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
          <Toaster
            theme="dark"
            position="top-center"
            richColors
            closeButton
            toastOptions={{ duration: 4000 }}
          />
          {children}
          <HelpWidget />
        </ThemeProvider>
      </body>
    </html>
  );
}
