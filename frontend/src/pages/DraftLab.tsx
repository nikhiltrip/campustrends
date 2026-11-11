import { FlaskConical, LoaderCircle } from "lucide-react";
import type { FormEvent } from "react";
import { CardHeader } from "../components/ReportWidgets";
import { pct } from "../format";
import type { Score } from "../types";

type Props = {
  local: boolean;
  modelReady: boolean;
  scoring: boolean;
  draft: string;
  setDraft: (s: string) => void;
  postTime: string;
  setPostTime: (s: string) => void;
  score: Score | null;
  scoreDraft: (e: FormEvent) => void;
};
export function DraftLab({
  local,
  modelReady,
  scoring,
  draft,
  setDraft,
  postTime,
  setPostTime,
  score,
  scoreDraft,
}: Props) {
  return (
    <div className="lab-grid">
      <section className="card lab-card">
        <CardHeader kicker="POSTING-TIME FEATURES" title="Your draft" />
        <form onSubmit={scoreDraft}>
          <label htmlFor="draft">Post text</label>
          <textarea
            id="draft"
            maxLength={20000}
            minLength={5}
            rows={6}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            required
          />
          <div className="input-meta">
            {draft.length} characters · Text and time only, no upvote input
          </div>
          <label htmlFor="time">
            Planned posting time <small>(your device time zone)</small>
          </label>
          <input
            id="time"
            type="datetime-local"
            required
            value={postTime}
            onChange={(e) => setPostTime(e.target.value)}
          />
          <button
            type="submit"
            className="primary"
            disabled={!local || !modelReady || scoring}
          >
            {scoring ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <FlaskConical size={16} />
            )}{" "}
            {scoring ? "Scoring…" : "Score draft"}
          </button>
        </form>
        {(!local || !modelReady) && (
          <p className="fine">
            {!local
              ? "Draft scoring runs in the local Python app. Start it using the README instructions."
              : "This imported report has no connected model. Analyze its source CSV/JSON locally to enable scoring."}{" "}
            <a
              href="https://github.com/nikhiltrip/campustrends#run-the-dashboard"
              target="_blank"
              rel="noreferrer"
            >
              Setup guide ↗
            </a>
          </p>
        )}
      </section>
      <section className="card score-card">
        <span className="eyebrow">MODEL OUTPUT</span>
        {score ? (
          <>
            <div className="score-number">
              {(score.score * 100).toFixed(1)}
              <small>/ 100</small>
            </div>
            <h2>Engagement model score</h2>
            <p>{score.note}</p>
            <dl>
              <div>
                <dt>Closest topic</dt>
                <dd>{score.topic.keywords.slice(0, 2).join(" + ")}</dd>
              </div>
              <div>
                <dt>Academic context</dt>
                <dd>
                  {score.term}
                  {score.week_of_term > 0 ? `, week ${score.week_of_term}` : ""}
                </dd>
              </div>
              <div>
                <dt>Known word coverage</dt>
                <dd>{pct(score.known_word_fraction)}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd>{score.model}</dd>
              </div>
            </dl>
            {score.known_word_fraction < 0.3 && (
              <p className="fine">
                Most draft words are outside the training vocabulary. Treat this
                score with extra caution.
              </p>
            )}
            <p className="fine">
              A higher score means stronger resemblance to learned
              high-engagement patterns. This is not a guaranteed outcome or a
              calibrated probability.
            </p>
          </>
        ) : (
          <div className="score-empty">
            <FlaskConical size={38} />
            <h2>A draft, a time, a model.</h2>
            <p>
              Inspect the predicted score and topic after you run a local
              analysis.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
