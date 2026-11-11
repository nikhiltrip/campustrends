export type Topic = {
  id: number;
  name: string;
  keywords: string[];
  recent_count: number;
  previous_count: number;
  recent_share: number;
  previous_share: number;
  share_change: number;
  status: string;
  recent_high_count: number;
  examples: string[];
};
export type Post = {
  id: string;
  text: string;
  upvotes: number;
  timestamp: string;
  platform: string;
  topic_id: number;
  term: string;
  week_of_term: number;
  high_engagement: boolean | null;
  threshold: number | null;
  split: string;
  score: number | null;
};
export type Metrics = {
  posts: number;
  positive_rate: number;
  precision_at_10: number;
  top_k: number;
  roc_auc: number | null;
  average_precision: number;
  precision_at_half: number;
  recall_at_half: number;
  confusion_matrix: number[][];
};
export type Report = {
  schema_version: 1;
  title: string;
  provenance: string;
  is_sample: boolean;
  school: string;
  data_start: string;
  data_end: string;
  calendar: { name: string; start: string; end: string }[];
  quality: Record<string, number>;
  summary: {
    total_posts: number;
    labeled_posts: number;
    high_posts: number;
    topic_count: number;
    platforms: string[];
  };
  trends: {
    window_days: number;
    end: string;
    recent_start: string;
    previous_start: string;
    recent_total: number;
    previous_total: number;
    topics: Topic[];
  };
  timeline: { date: string; topic_id: number; count: number }[];
  posts: Post[];
  archetypes: {
    id: number;
    name: string;
    keywords: string[];
    training_high_posts: number;
    examples: string[];
    recent_posts: number;
    recent_labeled_posts: number;
    recent_high_posts: number;
    recent_high_rate: number | null;
  }[];
  evaluation: {
    selected_model: string;
    selection_metric: string;
    validation: Record<string, Metrics>;
    test: Metrics;
    baseline: Metrics;
    splits: {
      name: string;
      count: number;
      start: string;
      end: string;
      positive_rate: number;
    }[];
    embargo_days: number;
    feature_weights: { feature: string; weight: number }[];
    limitations: string[];
  };
};
export type Score = {
  score: number;
  topic: Topic;
  term: string;
  week_of_term: number;
  known_word_fraction: number;
  is_sample: boolean;
  model: string;
  outside_training_dates: boolean;
  note: string;
};
