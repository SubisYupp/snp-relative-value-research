import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'SNP / Relative Value Research', description: 'Chronological options research, volatility regimes and strike-level explanations.' };
export default function RootLayout({children}:{children:React.ReactNode}) { return <html lang="en"><body>{children}</body></html>; }
