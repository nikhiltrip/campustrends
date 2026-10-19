import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Database,
  Layers3,
} from "lucide-react";
import type { Filters } from "../components/PostExplorer";
import { PostFilters, Posts } from "../components/PostExplorer";
import { CardHeader, Stat, Timeline } from "../components/ReportWidgets";
import { date, pct, title } from "../format";
import type { Post, Report } from "../types";
type Props = {
  report: Report;
  filters: Filters;
  posts: Post[];
  setSelected: (post: Post) => void;
  setPage: (page: string) => void;
};
export function Overview({
  report,
  filters,
  posts,
  setSelected,
  setPage,
}: Props) {
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
  const rising = topics.filter(
    (t) => t.status === "rising" || t.status === "new in window",
  );
  return (
    <>
      <div className="stats-grid">
        <Stat
          label="POSTS ANALYZED"
          value={report.summary.total_posts.toLocaleString()}
          note={`${date(report.data_start)} – ${date(report.data_end)}, ${new Date(report.data_end).getFullYear()}`}
          icon={<Database size={18} />}
        />
        <Stat
          label="TOPICS DISCOVERED"
          value={String(report.summary.topic_count)}
          note="LDA · fitted on training posts"
          icon={<Layers3 size={18} />}
        />
        <Stat
          label="RISING TOPICS"
          value={String(rising.length)}
          note="New or +3 percentage points"
          icon={<Activity size={18} />}
        />
        <Stat
          label="HELD-OUT TOP-DECILE PRECISION"
          value={pct(report.evaluation.test.precision_at_10)}
          note={
            report.is_sample
              ? "Sample test split only"
              : `${report.evaluation.test.top_k} highest-scored test posts`
          }
          icon={<BarChart3 size={18} />}
        />
      </div>
      <div className="content-grid">
        <section className="card chart-card">
          <CardHeader
            kicker="CONVERSATION OVER TIME"
            title="Campus pulse"
            aside={<span className="pill">Daily post volume</span>}
          />
          <Timeline report={report} topicId={null} />
          <div className="chart-footer">
            <span>
              {date(report.data_start)} – {date(report.data_end)} · Los Angeles
              dates
            </span>
            <button onClick={() => setPage("Topics & trends")}>
              Explore topics <ArrowRight size={14} />
            </button>
          </div>
        </section>
        <section className="card rising-card">
          <CardHeader
            kicker="LAST 7 DAYS VS. PREVIOUS 7"
            title="Gaining ground"
          />
          <div className="ranked-topics">
            {topics.slice(0, 4).map((t, i) => (
              <button
                key={t.id}
                onClick={() => {
                  setTopic(String(t.id));
                  setPage("Topics & trends");
                }}
              >
                <span className="rank">0{i + 1}</span>
                <span className="topic-text">
                  <strong>{title(t)}</strong>
                  <small>
                    {t.recent_count} posts · {pct(t.recent_share)} of recent
                    volume
                  </small>
                </span>
                <span className={t.share_change > 0 ? "change up" : "change"}>
                  {t.share_change > 0 ? "+" : ""}
                  {(t.share_change * 100).toFixed(1)}
                  <small>pp</small>
                </span>
              </button>
            ))}
          </div>
          <p className="fine">
            Share change measures percentage points, not raw volume growth.
          </p>
        </section>
      </div>
      <section className="card">
        <CardHeader
          kicker="POST EXPLORER"
          title="A closer look"
          aside={
            <span className="muted">
              Click a post to inspect its context <ArrowUpRight size={14} />
            </span>
          }
        />
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
          posts={posts.slice(0, 8)}
          topics={topics}
          onSelect={setSelected}
        />
        <div className="chart-footer">
          <span>
            {posts.length.toLocaleString()} matching posts · showing{" "}
            {Math.min(8, posts.length)}
          </span>
          <button onClick={() => setPage("Dataset")}>
            View dataset <ArrowRight size={14} />
          </button>
        </div>
      </section>
    </>
  );
}
