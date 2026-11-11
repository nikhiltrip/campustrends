import { Info, Layers3 } from "lucide-react";
import { pct } from "../format";
import type { Report } from "../types";
export function Archetypes({ report }: { report: Report }) {
  return (
    <>
      <div className="window-note">
        <Info size={16} /> K-means on TF-IDF vectors from high-engagement
        training posts. Names use cluster keywords; representative examples come
        from training data.
      </div>
      <div className="archetype-grid">
        {report.archetypes.map((a, i) => (
          <section className="card archetype" key={a.id}>
            <span className="eyebrow">
              PATTERN 0{i + 1} <Layers3 size={16} />
            </span>
            <h2>{a.name.replace(" / ", " + ")}</h2>
            <div className="tags">
              {a.keywords.slice(0, 4).map((k) => (
                <span key={k}>{k}</span>
              ))}
            </div>
            <blockquote>“{a.examples[0]}”</blockquote>
            <div className="archetype-stats">
              <span>
                <strong>{a.training_high_posts}</strong>training high performers
              </span>
              <span>
                <strong>{a.recent_posts}</strong>recent posts assigned
              </span>
              <span>
                <strong>{pct(a.recent_high_rate)}</strong>recent high-engagement
                rate
              </span>
            </div>
            <p className="fine">
              Recent rate: {a.recent_high_posts} / {a.recent_labeled_posts}{" "}
              labeled posts. Small groups produce unstable estimates.
            </p>
          </section>
        ))}
      </div>
    </>
  );
}
