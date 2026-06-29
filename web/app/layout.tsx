import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'SudokuDiT — masked-diffusion Sudoku solver',
  description:
    'A 1.28M-parameter Diffusion Transformer solving Sudoku as masked discrete diffusion, running entirely in your browser via ONNX.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
