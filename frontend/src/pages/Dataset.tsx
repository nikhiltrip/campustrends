import { LoaderCircle, RefreshCw, Upload } from "lucide-react";
import type { Filters } from "../components/PostExplorer";
import { PostFilters, Posts } from "../components/PostExplorer";
import { CardHeader } from "../components/ReportWidgets";
import { date } from "../format";
import type { Post, Report } from "../types";

type Props = {
  report: Report;
  filters: Filters;
  posts: Post[];
  setSelected: (post: Post) => void;
  local: boolean;
  loading: boolean;
  calendarText: string;
  setCalendarText: (text: string) => void;
  onDataUpload: () => void;
  onReportImport: () => void;
  reset: () => void;
};
export function Dataset({
  report,
  filters,
  posts,
  setSelected,
  local,
  loading,
  calendarText,
  setCalendarText,
  onDataUpload,
  onReportImport,
  reset,
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
  return (
    <>
      <div className="content-grid">
        <section className="card">
          <CardHeader
            kicker="SOURCE & COVERAGE"
            title={
              report.is_sample
                ? "A reproducible sample"
                : "Your imported export"
            }
          />
          <dl>
            <div>
              <dt>Origin</dt>
              <dd>
                {report.is_sample
                  ? "Seeded fictional posts"
                  : "User-supplied CSV / JSON"}
              </dd>
            </div>
            <div>
              <dt>Coverage</dt>
              <dd>
                {date(report.data_start)} – {date(report.data_end)}
              </dd>
            </div>
            <div>
              <dt>Platforms</dt>
              <dd>{report.summary.platforms.join(", ")}</dd>
            </div>
            <div>
              <dt>Academic calendar</dt>
              <dd>
                {report.calendar.length
                  ? `${report.calendar.length} configured terms${report.is_sample ? " (illustrative dates)" : ""}`
                  : "Not supplied; term features disabled"}
              </dd>
            </div>
            <div>
              <dt>Engagement target</dt>
              <dd>
                Upvotes above the prior 7-day 90th percentile, with at least 20
                earlier posts. Equal-time posts excluded.
              </dd>
            </div>
          </dl>
          <p className="fine">
            The warm-up period has no label. Threshold ties count as negative. A
            rolling reference target need not label exactly 10% of posts.
          </p>
        </section>
        <section className="card">
          <CardHeader kicker="DATA QUALITY" title="Clean-up summary" />
          {Object.entries(report.quality).map(([key, value]) => (
            <div className="quality-row" key={key}>
              <span>{key.replaceAll("_", " ")}</span>
              <strong>{value.toLocaleString()}</strong>
            </div>
          ))}
        </section>
      </div>
      <section className="card import-card">
        <CardHeader kicker="BRING YOUR OWN DATA" title="Run an analysis" />
        <p>
          Use a local CSV or JSON export with{" "}
          <code>post_id, text, upvotes, timestamp, school</code>. Add{" "}
          <code>platform</code> if available. Store upvote counts at a
          consistent post age. Remove personal information from text before
          importing.
        </p>
        <label htmlFor="calendar">
          Academic terms <small>(optional JSON; dates in Los Angeles)</small>
        </label>
        <textarea
          id="calendar"
          rows={3}
          value={calendarText}
          onChange={(e) => setCalendarText(e.target.value)}
          disabled={!local}
        />
        <div className="import-actions">
          <button
            className="primary"
            disabled={!local || loading}
            onClick={() => onDataUpload()}
          >
            {loading ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <Upload size={16} />
            )}{" "}
            {loading ? "Training local models…" : "Analyze CSV / JSON"}
          </button>
          <button className="secondary" onClick={() => onReportImport()}>
            <Upload size={16} /> Open saved report
          </button>
          <button className="text-button" onClick={reset}>
            <RefreshCw size={15} />{" "}
            {local ? "Reload startup dataset" : "Reset sample"}
          </button>
        </div>
        <p className="fine">
          {local
            ? "Files stay on this computer. Uploaded analysis lives in memory until you restart or reset; export its report to keep it. Maximum 5 MB / 10,000 rows. Use the CLI for larger exports."
            : "This preview can open completed report.json files. Run the local app to train on raw CSV/JSON and score drafts."}{" "}
          <a
            href="https://github.com/nikhiltrip/campustrends"
            target="_blank"
            rel="noreferrer"
          >
            Instructions ↗
          </a>
        </p>
      </section>
      <section className="card">
        <CardHeader kicker="CLEANED POSTS" title="Inspect the dataset" />
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
          posts={posts.slice(0, 40)}
          topics={topics}
          onSelect={setSelected}
        />
        <p className="fine">
          Showing {Math.min(40, posts.length)} of {posts.length} matches. Export
          the report for the complete dataset.
        </p>
      </section>
    </>
  );
}
