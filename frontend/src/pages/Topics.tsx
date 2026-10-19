import { X } from "lucide-react";
import type { Filters } from "../components/PostExplorer";
import { PostFilters, Posts } from "../components/PostExplorer";
import { CardHeader, Timeline } from "../components/ReportWidgets";
import { COLORS, date, pct, title } from "../format";
import type { Post, Report } from "../types";
type Props = {
  report: Report;
  filters: Filters;
  posts: Post[];
  setSelected: (post: Post) => void;
  setPage: (page: string) => void;
};
export function Topics({ report, filters, posts, setSelected }: Props) {
  const {
    topic,
    setTopic,
    query,
    setQuery,
    platform,
    setPlatform,
    sort,
    setSort,
    period,
    setPeriod,
  } = filters;
  const topics = report.trends.topics;
  const activeTopic = topics.find((t) => t.id === Number(topic));
  return (
    <>
      <div className="window-note">
        <span className="status-dot" /> Recent:{" "}
        {date(report.trends.recent_start)} – {date(report.trends.end)} (
        {report.trends.recent_total} posts) <span className="divider">/</span>{" "}
        Previous: {date(report.trends.previous_start)} –{" "}
        {date(report.trends.recent_start)} ({report.trends.previous_total}{" "}
        posts)
      </div>
      <div className="topic-grid">
        {topics.map((t) => (
          <button
            className={`card topic-card ${topic === String(t.id) ? "chosen" : ""}`}
            key={t.id}
            onClick={() =>
              setTopic(topic === String(t.id) ? "all" : String(t.id))
            }
          >
            <div className="topic-top">
              <span
                className="topic-dot"
                style={{ background: COLORS[t.id % 6] }}
              />{" "}
              <span className={t.status === "rising" ? "pill green" : "pill"}>
                {t.status}
              </span>
            </div>
            <h2>{title(t)}</h2>
            <p>{t.keywords.slice(2, 6).join(" · ")}</p>
            <div className="share-track">
              <span
                style={{
                  width: `${Math.max(1, t.recent_share * 100)}%`,
                  background: COLORS[t.id % 6],
                }}
              />
            </div>
            <div className="topic-bottom">
              <strong>
                {pct(t.recent_share)}
                <small>{t.recent_count} recent posts</small>
              </strong>
              <span className={t.share_change > 0 ? "change up" : "change"}>
                {t.share_change > 0 ? "+" : ""}
                {(t.share_change * 100).toFixed(1)} pp
              </span>
            </div>
          </button>
        ))}
      </div>
      <section className="card">
        <CardHeader
          kicker="DAILY TOPIC VOLUME"
          title={activeTopic ? title(activeTopic) : "All conversations"}
          aside={
            topic !== "all" ? (
              <button onClick={() => setTopic("all")}>
                Clear selection <X size={14} />
              </button>
            ) : null
          }
        />
        <Timeline
          report={report}
          topicId={topic === "all" ? null : Number(topic)}
        />
        <p className="fine">
          Keyword names are automatic. Topic assignments indicate similar
          wording, not verified campus subcultures.
        </p>
      </section>
      <section className="card">
        <CardHeader kicker="TOPIC POSTS" title="Read the context" />
        <PostFilters
          {...{
            topics,
            topic,
            setTopic,
            query,
            setQuery,
            platform,
            setPlatform,
            report,
            sort,
            setSort,
            period,
            setPeriod,
          }}
        />
        <Posts
          posts={posts.slice(0, 12)}
          topics={topics}
          onSelect={setSelected}
        />
      </section>
    </>
  );
}
