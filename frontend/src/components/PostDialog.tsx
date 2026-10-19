import { X } from "lucide-react";
import { useEffect, useRef } from "react";
import { fullDate, pct } from "../format";
import type { Post, Topic } from "../types";
export function PostDialog({
  post,
  topic,
  onClose,
}: {
  post: Post;
  topic?: Topic;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-header">
        <span className="eyebrow">POST CONTEXT</span>
        <button onClick={onClose} autoFocus aria-label="Close post">
          <X size={20} />
        </button>
      </div>
      <p className="modal-post">{post.text}</p>
      <div className="tags">
        {topic?.keywords.slice(0, 4).map((k) => (
          <span key={k}>{k}</span>
        ))}
      </div>
      <dl>
        <div>
          <dt>Platform</dt>
          <dd>{post.platform}</dd>
        </div>
        <div>
          <dt>Posted</dt>
          <dd>{fullDate(post.timestamp)} PT</dd>
        </div>
        <div>
          <dt>Upvotes</dt>
          <dd>{post.upvotes}</dd>
        </div>
        <div>
          <dt>Historical threshold</dt>
          <dd>
            {post.threshold === null
              ? "Warm-up: insufficient prior posts"
              : post.threshold.toFixed(1)}
          </dd>
        </div>
        <div>
          <dt>High engagement</dt>
          <dd>
            {post.high_engagement === null
              ? "Unlabeled"
              : post.high_engagement
                ? "Yes"
                : "No"}
          </dd>
        </div>
        <div>
          <dt>Split</dt>
          <dd>{post.split}</dd>
        </div>
        <div>
          <dt>Held-out model score</dt>
          <dd>{post.score === null ? "Not a test post" : pct(post.score)}</dd>
        </div>
        <div>
          <dt>Term</dt>
          <dd>
            {post.term}
            {post.week_of_term ? ` · week ${post.week_of_term}` : ""}
          </dd>
        </div>
      </dl>
      <p className="fine">
        The model never uses this post’s upvotes or engagement label as input.
      </p>
    </dialog>
  );
}
