import React, { useMemo, useState } from "react";
import { COLORS, date } from "../format";
import type { Report } from "../types";
export function Stat({
  label,
  value,
  note,
  icon,
}: {
  label: string;
  value: string;
  note: string;
  icon?: React.ReactNode;
}) {
  return (
    <section className="card stat">
      <div>
        <span className="eyebrow">{label}</span>
        {icon}
      </div>
      <strong className={value.length > 12 ? "stat-value word" : "stat-value"}>
        {value}
      </strong>
      <p>{note}</p>
    </section>
  );
}
export function CardHeader({
  kicker,
  title,
  aside,
}: {
  kicker: string;
  title: string;
  aside?: React.ReactNode;
}) {
  return (
    <div className="card-header">
      <div>
        <p className="eyebrow">{kicker}</p>
        <h2>{title}</h2>
      </div>
      {aside}
    </div>
  );
}
export function Timeline({
  report,
  topicId,
}: {
  report: Report;
  topicId: number | null;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const days = useMemo(() => {
    const map = new Map<string, number[]>();
    for (
      let t = new Date(report.data_start);
      t <= new Date(report.data_end);
      t = new Date(+t + 86400000)
    ) {
      const key = t.toLocaleDateString("en-CA", {
        timeZone: "America/Los_Angeles",
      });
      map.set(key, Array(report.summary.topic_count).fill(0));
    }
    for (const p of report.timeline) {
      if (!map.has(p.date))
        map.set(p.date, Array(report.summary.topic_count).fill(0));
      map.get(p.date)![p.topic_id] = p.count;
    }
    return [...map.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [report]);
  const max = Math.max(
    1,
    ...days.map(([, v]) =>
      topicId === null ? v.reduce((a, b) => a + b, 0) : v[topicId] || 0,
    ),
  );
  const w = 760,
    h = 206,
    left = 30,
    right = 12,
    top = 16,
    bottom = 28;
  const chartW = w - left - right,
    chartH = h - top - bottom;
  const values = days.map(([, v]) =>
    topicId === null ? v.reduce((a, b) => a + b, 0) : v[topicId] || 0,
  );
  const x = (i: number) => left + (i / Math.max(1, days.length - 1)) * chartW,
    y = (n: number) => top + chartH - (n / max) * chartH;
  const points = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  return (
    <div className="timeline">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`${topicId === null ? "All topics" : "Selected topic"} daily post counts`}
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const index = Math.round(
            ((((e.clientX - rect.left) / rect.width) * w - left) / chartW) *
              (days.length - 1),
          );
          setHover(Math.max(0, Math.min(days.length - 1, index)));
        }}
      >
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line
              x1={left}
              x2={w - right}
              y1={y(f * max)}
              y2={y(f * max)}
              stroke="#e9ece6"
              strokeDasharray="3 4"
            />
            <text
              x={left - 10}
              y={y(f * max) + 4}
              textAnchor="end"
              className="chart-label"
            >
              {Math.round(f * max)}
            </text>
          </g>
        ))}
        <polygon
          points={`${left},${y(0)} ${points} ${x(days.length - 1)},${y(0)}`}
          fill={topicId === null ? "#3159de10" : `${COLORS[topicId % 6]}18`}
        />
        <polyline
          points={points}
          fill="none"
          stroke={topicId === null ? COLORS[0] : COLORS[topicId % 6]}
          strokeWidth="2"
          strokeLinejoin="round"
        />
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const i = Math.round(f * (days.length - 1));
          return (
            <text
              key={f}
              x={x(i)}
              y={h - 5}
              textAnchor={f === 0 ? "start" : f === 1 ? "end" : "middle"}
              className="chart-label"
            >
              {date(days[i][0] + "T12:00:00-07:00")}
            </text>
          );
        })}
        {hover !== null && (
          <g>
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1={top}
              y2={top + chartH}
              stroke="#6b7c71"
              strokeDasharray="3 3"
            />
            <circle
              cx={x(hover)}
              cy={y(values[hover])}
              r="4"
              fill={COLORS[0]}
              stroke="white"
              strokeWidth="2"
            />
          </g>
        )}
      </svg>
      <div className="chart-readout" aria-live="polite">
        {hover !== null
          ? `${date(days[hover][0] + "T12:00:00-07:00")} · ${values[hover]} posts`
          : "Move across the chart to inspect a day"}
      </div>
    </div>
  );
}
