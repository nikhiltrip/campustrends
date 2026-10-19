import { CardHeader, Stat } from "../components/ReportWidgets";
import { COLORS, fullDate, pct } from "../format";
import type { Report } from "../types";
export function Evaluation({ report }: { report: Report }) {
  return (
    <>
      <div className="stats-grid">
        <Stat
          label="SELECTED MODEL"
          value={report.evaluation.selected_model}
          note="Selected by validation average precision"
        />
        <Stat
          label="TEST PRECISION @ TOP 10%"
          value={pct(report.evaluation.test.precision_at_10)}
          note={`${report.evaluation.test.top_k} highest-scored held-out posts`}
        />
        <Stat
          label="TEST ROC-AUC"
          value={report.evaluation.test.roc_auc?.toFixed(3) || "Unavailable"}
          note="0.5 corresponds to random ranking"
        />
        <Stat
          label="TEST POSITIVE RATE"
          value={pct(report.evaluation.test.positive_rate)}
          note="Expected precision for random selection"
        />
      </div>
      <section className="card">
        <CardHeader
          kicker="CHRONOLOGICAL HOLDOUT"
          title="Train on the past. Test on newer posts."
        />
        <div className="split-bar">
          {report.evaluation.splits.map((s, i) => (
            <div key={s.name} style={{ flex: s.count, background: COLORS[i] }}>
              <strong>{s.name}</strong>
              <span>{s.count} posts</span>
            </div>
          ))}
        </div>
        <div className="split-details">
          {report.evaluation.splits.map((s) => (
            <div key={s.name}>
              <strong>{s.name}</strong>
              <p>
                {fullDate(s.start)}
                <br />
                to {fullDate(s.end)}
              </p>
              <small>{pct(s.positive_rate)} high engagement</small>
            </div>
          ))}
        </div>
        <p className="fine">
          A {report.evaluation.embargo_days}-day gap separates splits for a
          48-hour outcome observation horizon. Vocabulary, scaling, LDA, and
          archetype clustering use training data only. Model selection uses
          validation average precision. The final test remains held out.
        </p>
      </section>
      <div className="content-grid">
        <section className="card">
          <CardHeader kicker="MODEL SELECTION" title="Validation results" />
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Average precision</th>
                  <th>Top 10% precision</th>
                  <th>ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(report.evaluation.validation).map(
                  ([name, m]) => (
                    <tr key={name}>
                      <td>
                        {name}
                        {name === report.evaluation.selected_model && (
                          <span className="pill green">selected</span>
                        )}
                      </td>
                      <td>{m.average_precision.toFixed(3)}</td>
                      <td>{pct(m.precision_at_10)}</td>
                      <td>{m.roc_auc?.toFixed(3)}</td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
          <p className="fine">
            Top-decile precision ranks scores and inspects the highest 10%. It
            differs from precision at a 0.5 decision threshold.
          </p>
        </section>
        <section className="card">
          <CardHeader kicker="TEST / THRESHOLD 0.5" title="Confusion matrix" />
          <div className="matrix">
            {report.evaluation.test.confusion_matrix.flat().map((n, i) => (
              <div key={i} className={i === 0 || i === 3 ? "correct" : ""}>
                <strong>{n}</strong>
                <small>
                  {
                    [
                      "True negative",
                      "False positive",
                      "False negative",
                      "True positive",
                    ][i]
                  }
                </small>
              </div>
            ))}
          </div>
          <p className="fine">
            Class imbalance can cause the model to predict few positives at 0.5
            even when its ranking beats chance.
          </p>
        </section>
      </div>
      <section className="card limitations">
        <CardHeader kicker="INTERPRETATION" title="Limits of this analysis" />
        <ul>
          {report.evaluation.limitations.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
        <p className="fine">
          The project’s original 70% precision goal remains a target. This
          report does not establish that performance on real UCLA data.
        </p>
      </section>
    </>
  );
}
