import type { Topic } from "./types";
export const COLORS = [
  "#3159de",
  "#cf6744",
  "#8371b1",
  "#3d8b9a",
  "#b29639",
  "#bb7090",
];
export const pct = (v: number | null) =>
  v === null ? "Unavailable" : `${(v * 100).toFixed(1)}%`;
export const date = (s: string) =>
  new Date(s).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "America/Los_Angeles",
  });
export const fullDate = (s: string) =>
  new Date(s).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "America/Los_Angeles",
  });
export const title = (t: Topic) => t.keywords.slice(0, 2).join(" + ");
