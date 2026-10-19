import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronRight,
  Database,
  Download,
  FlaskConical,
  Info,
  Layers3,
  X,
} from "lucide-react";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { PostDialog } from "./components/PostDialog";
import type { Filters } from "./components/PostExplorer";
import { fullDate } from "./format";
import { Archetypes } from "./pages/Archetypes";
import { Dataset } from "./pages/Dataset";
import { DraftLab } from "./pages/DraftLab";
import { Evaluation } from "./pages/Evaluation";
import { Overview } from "./pages/Overview";
import { Topics } from "./pages/Topics";
import { validateReport } from "./report";
import "./style.css";
import type { Post, Report, Score } from "./types";

const download = (report: Report) => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(
    new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
  );
  a.download = "campustrends-report.json";
  a.click();
  URL.revokeObjectURL(a.href);
};
function App() {
  const [report, setReport] = useState<Report | null>(null),
    [page, setPage] = useState("Overview"),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false),
    [local, setLocal] = useState(false),
    [modelReady, setModelReady] = useState(false);
  const [topic, setTopic] = useState("all"),
    [query, setQuery] = useState(""),
    [platform, setPlatform] = useState("all"),
    [sort, setSort] = useState("recent"),
    [selected, setSelected] = useState<Post | null>(null),
    [period, setPeriod] = useState("all");
  const [draft, setDraft] = useState(
      "Anyone else looking for a quiet place to study during finals week?",
    ),
    [postTime, setPostTime] = useState("2026-10-18T18:00"),
    [score, setScore] = useState<Score | null>(null),
    [scoring, setScoring] = useState(false);
  const reportFile = useRef<HTMLInputElement>(null),
    dataFile = useRef<HTMLInputElement>(null);
  const [calendarText, setCalendarText] = useState("[]");
  const installReport = (r: Report) => {
    setReport(r);
    setTopic("all");
    setQuery("");
    setPlatform("all");
    setSelected(null);
    setScore(null);
    setPeriod("all");
    setCalendarText(JSON.stringify(r.calendar, null, 2));
  };
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const response = await fetch("/api/report");
        if (
          response.ok &&
          response.headers.get("content-type")?.includes("application/json")
        ) {
          const r = validateReport(await response.json());
          if (live) {
            installReport(r);
            setLocal(true);
            setModelReady(true);
          }
          return;
        }
      } catch {
        /* Hosted demo loads the static report below. */
      }
      try {
        const response = await fetch("/demo-report.json");
        if (!response.ok)
          throw Error(
            "Could not load the demo. Start the local app or refresh the page.",
          );
        const r = validateReport(await response.json());
        if (live) installReport(r);
      } catch (e) {
        if (live) setError(String(e));
      }
    })();
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    const escape = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSelected(null);
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, []);
  const posts = useMemo(() => {
    if (!report) return [];
    return report.posts
      .filter(
        (p) =>
          (topic === "all" || p.topic_id === Number(topic)) &&
          (platform === "all" || p.platform === platform) &&
          p.text.toLowerCase().includes(query.toLowerCase()) &&
          (period === "all" ||
            new Date(p.timestamp) >= new Date(report.trends.recent_start)),
      )
      .sort((a, b) =>
        sort === "upvotes"
          ? b.upvotes - a.upvotes
          : Date.parse(b.timestamp) - Date.parse(a.timestamp),
      );
  }, [report, topic, platform, query, sort, period]);
  async function importReport(file?: File) {
    if (!file) return;
    try {
      if (file.size > 15 * 1024 * 1024)
        throw Error("Reports must be smaller than 15 MB.");
      installReport(validateReport(JSON.parse(await file.text())));
      setModelReady(false);
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  async function analyze(file?: File) {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      if (file.size > 5 * 1024 * 1024)
        throw Error("Data files must be smaller than 5 MB.");
      const form = new FormData();
      form.append("file", file);
      form.append("calendar", calendarText);
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: form,
      });
      const r = await response.json();
      if (!response.ok) throw Error(r.detail || "Analysis failed.");
      installReport(validateReport(r));
      setModelReady(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }
  async function scoreDraft(e: React.FormEvent) {
    e.preventDefault();
    setScoring(true);
    setError("");
    setScore(null);
    try {
      const response = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text: draft,
          timestamp: new Date(postTime).toISOString(),
        }),
      });
      const data = await response.json();
      if (!response.ok)
        throw Error(
          typeof data.detail === "string"
            ? data.detail
            : "Check your draft and posting time.",
        );
      setScore(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setScoring(false);
    }
  }
  async function reset() {
    try {
      if (local) {
        const response = await fetch("/api/reset", { method: "POST" });
        if (!response.ok) throw Error("Could not reload the local report.");
        installReport(validateReport(await response.json()));
        setModelReady(true);
      } else {
        const response = await fetch("/demo-report.json");
        installReport(validateReport(await response.json()));
      }
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  if (!report)
    return (
      <main className="loading">
        <Activity size={30} />
        <h1>
          CampusTrends<span>ML</span>
        </h1>
        <p>{error || "Loading the campus post analysis…"}</p>
        <button onClick={() => window.location.reload()}>Reload</button>
      </main>
    );
  const topics = report.trends.topics;
  const filters: Filters = {
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
  };
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("Overview");
          }}
        >
          <span className="brand-icon">
            <Activity size={23} />
          </span>
          <span>
            CampusTrends<span className="ml">ML</span>
            <small>FIELD NOTES FROM CAMPUS</small>
          </span>
        </a>
        <div className="workspace">
          <div className="school-mark">U</div>
          <div>
            UCLA workspace<small>Local research project</small>
          </div>
          <span className="status-dot" />
        </div>
        <p className="nav-caption">EXPLORE</p>
        <nav aria-label="Main navigation">
          {[
            { name: "Overview", icon: BarChart3 },
            { name: "Topics & trends", icon: Activity },
            { name: "Post archetypes", icon: Layers3 },
            { name: "Draft lab", icon: FlaskConical },
            { name: "Evaluation", icon: Check },
            { name: "Dataset", icon: Database },
          ].map(({ name, icon: Icon }) => (
            <button
              key={name}
              className={page === name ? "nav-item active" : "nav-item"}
              onClick={() => {
                setPage(name);
                setSelected(null);
              }}
            >
              <Icon size={18} />
              {name}
              {page === name && <ChevronRight size={15} />}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <span className="eyebrow">ABOUT THIS WORKSPACE</span>
          <p>
            Find the topics people are talking about and the patterns in posts
            that gain attention.
          </p>
          <a
            href="https://github.com/nikhiltrip/campustrends"
            target="_blank"
            rel="noreferrer"
          >
            View the code <ArrowUpRight size={14} />
          </a>
          <span className="local-badge">
            <span className="status-dot" />
            {local ? "Local analysis connected" : "Offline sample preview"}
          </span>
        </div>
      </aside>
      <div className="main-wrap">
        <header className="topbar">
          <div>
            Workspace <ChevronRight size={14} /> <strong>{page}</strong>
          </div>
          <span className="top-meta">
            {local ? "LOCAL ANALYSIS" : "SAMPLE PREVIEW"} <span> / </span> UCLA
          </span>
        </header>
        <main>
          <div className={report.is_sample ? "notice sample" : "notice"}>
            <Info size={17} />
            <span>
              {report.is_sample ? (
                <>
                  <strong>Sample dataset.</strong> Fictional posts and upvotes
                  demonstrate the workflow. These results do not measure real
                  campus behavior.
                </>
              ) : (
                <>
                  <strong>Imported dataset.</strong> Results describe this
                  export. Confirm consistent upvote observation ages before
                  interpreting predictions.
                </>
              )}
            </span>
            <button onClick={() => setPage("Dataset")}>
              Data details <ArrowRight size={14} />
            </button>
          </div>
          {error && (
            <div role="alert" className="error">
              {error}
              <button onClick={() => setError("")} aria-label="Dismiss error">
                <X size={16} />
              </button>
            </div>
          )}
          <section className="page-heading">
            <div>
              <p className="eyebrow">
                CAMPUS INTELLIGENCE /{" "}
                {page === "Overview"
                  ? "01"
                  : page === "Topics & trends"
                    ? "02"
                    : page === "Post archetypes"
                      ? "03"
                      : page === "Draft lab"
                        ? "04"
                        : page === "Evaluation"
                          ? "05"
                          : "06"}
              </p>
              <h1>
                {page === "Overview"
                  ? "The campus conversation."
                  : page === "Topics & trends"
                    ? "Follow the conversation."
                    : page === "Post archetypes"
                      ? "Patterns in high performers."
                      : page === "Draft lab"
                        ? "Test a post in context."
                        : page === "Evaluation"
                          ? "Check the evidence."
                          : "Start with the data."}
              </h1>
              <p>
                {page === "Overview"
                  ? "Topics, shifts, and engagement across anonymous campus posts."
                  : page === "Topics & trends"
                    ? "Compare topic shares across two adjacent weeks, accounting for overall post volume."
                    : page === "Post archetypes"
                      ? "Text clusters fitted on high-engagement training posts, then tracked in the recent window."
                      : page === "Draft lab"
                        ? "Score draft text and a posting time with the saved engagement model."
                        : page === "Evaluation"
                          ? "Older posts train the models. Newer posts test how they generalize."
                          : "Inspect the source, clean-up counts, and import options."}
              </p>
            </div>
            <button
              className="secondary export"
              onClick={() => download(report)}
            >
              <Download size={15} /> Export report
            </button>
          </section>
          {page === "Overview" && (
            <Overview
              report={report}
              filters={filters}
              posts={posts}
              setSelected={setSelected}
              setPage={setPage}
            />
          )}
          {page === "Topics & trends" && (
            <Topics
              report={report}
              filters={filters}
              posts={posts}
              setSelected={setSelected}
              setPage={setPage}
            />
          )}
          {page === "Post archetypes" && <Archetypes report={report} />}
          {page === "Draft lab" && (
            <DraftLab
              {...{
                local,
                modelReady,
                scoring,
                draft,
                setDraft,
                postTime,
                setPostTime,
                score,
                scoreDraft,
              }}
            />
          )}
          {page === "Evaluation" && <Evaluation report={report} />}
          {page === "Dataset" && (
            <Dataset
              report={report}
              filters={filters}
              posts={posts}
              setSelected={setSelected}
              local={local}
              loading={loading}
              calendarText={calendarText}
              setCalendarText={setCalendarText}
              onDataUpload={() => dataFile.current?.click()}
              onReportImport={() => reportFile.current?.click()}
              reset={reset}
            />
          )}
          <footer>
            <span>
              CampusTrendsML <span> / </span> A UCLA project by Nikhil Tripathi
            </span>
            <span>As of {fullDate(report.data_end)} PT</span>
          </footer>
          <input
            type="file"
            hidden
            ref={reportFile}
            accept=".json"
            onChange={(e) => {
              importReport(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
          <input
            type="file"
            hidden
            ref={dataFile}
            accept=".csv,.json"
            onChange={(e) => {
              analyze(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </main>
      </div>
      {selected && (
        <PostDialog
          post={selected}
          topic={topics.find((t) => t.id === selected.topic_id)}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
