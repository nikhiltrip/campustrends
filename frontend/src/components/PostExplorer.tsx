import { ArrowUpRight, Search } from "lucide-react";
import { COLORS, date, title } from "../format";
import type { Post, Report, Topic } from "../types";
export type Filters = {
  topics: Topic[];
  topic: string;
  setTopic: (s: string) => void;
  query: string;
  setQuery: (s: string) => void;
  platform: string;
  setPlatform: (s: string) => void;
  report: Report;
  sort: string;
  setSort: (s: string) => void;
  period: string;
  setPeriod: (s: string) => void;
};
export function PostFilters(p: Filters) {
  return (
    <div className="filters">
      <label className="search">
        <Search size={16} />
        <input
          aria-label="Search posts"
          placeholder="Search posts…"
          value={p.query}
          onChange={(e) => p.setQuery(e.target.value)}
        />
      </label>
      <select
        aria-label="Filter topic"
        value={p.topic}
        onChange={(e) => p.setTopic(e.target.value)}
      >
        <option value="all">All topics</option>
        {p.topics.map((t) => (
          <option key={t.id} value={t.id}>
            {title(t)}
          </option>
        ))}
      </select>
      <select
        aria-label="Filter platform"
        value={p.platform}
        onChange={(e) => p.setPlatform(e.target.value)}
      >
        <option value="all">All platforms</option>
        {p.report.summary.platforms.map((s) => (
          <option key={s}>{s}</option>
        ))}
      </select>
      <select
        aria-label="Filter period"
        value={p.period}
        onChange={(e) => p.setPeriod(e.target.value)}
      >
        <option value="all">All dates</option>
        <option value="recent">Recent 7 days</option>
      </select>
      <select
        aria-label="Sort posts"
        value={p.sort}
        onChange={(e) => p.setSort(e.target.value)}
      >
        <option value="recent">Newest first</option>
        <option value="upvotes">Most upvotes</option>
      </select>
    </div>
  );
}
export function Posts({
  posts,
  topics,
  onSelect,
}: {
  posts: Post[];
  topics: Topic[];
  onSelect: (p: Post) => void;
}) {
  return posts.length ? (
    <div className="table-scroll">
      <table className="posts">
        <thead>
          <tr>
            <th>Post</th>
            <th>Topic</th>
            <th>Upvotes</th>
            <th>Date</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {posts.map((p) => (
            <tr key={p.id}>
              <td>
                <button className="post-link" onClick={() => onSelect(p)}>
                  {p.text}
                  <small>{p.platform}</small>
                </button>
              </td>
              <td>
                <span className="topic-tag">
                  <i style={{ background: COLORS[p.topic_id % 6] }} />
                  {topics.find((t) => t.id === p.topic_id)?.keywords[0] ||
                    "Unknown"}
                </span>
              </td>
              <td>
                <span className="vote">
                  <ArrowUpRight size={13} />
                  {p.upvotes}
                </span>
              </td>
              <td className="date-cell">{date(p.timestamp)}</td>
              <td>
                <button
                  aria-label={`Inspect post ${p.id}`}
                  className="row-open"
                  onClick={() => onSelect(p)}
                >
                  <ArrowUpRight size={16} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ) : (
    <div className="empty">
      <Search size={24} />
      <h3>No matching posts</h3>
      <p>Try another topic, platform, date range, or search term.</p>
    </div>
  );
}
