import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = { title: "pr-review" };

// Wraps every page. Each page's content is passed in as `children`.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <main>{children}</main>
      </body>
    </html>
  );
}
