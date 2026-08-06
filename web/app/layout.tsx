import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Exercise Detection from Wrist Motion",
  description:
    "Identify which strength exercise a wrist-worn IMU recording contains.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
