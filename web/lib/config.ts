/**
 * Backend location. Defaults to the deployed API; set NEXT_PUBLIC_BACKEND_URL at
 * build time to point elsewhere - e.g. NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000
 * when running the backend locally.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "https://etrak.onrender.com";

export interface Recording {
  filename: string;
  activity: string;
}

export interface RecordingsResponse {
  count: number;
  classes: string[];
  recordings: Recording[];
}

export interface PredictResponse {
  filename: string;
  predicted: string;
  confidence: number;
  probabilities: Record<string, number>;
  true_label: string;
  correct: boolean;
  n_samples: number | null;
}

/** Uploaded recordings have no known label, so no true_label/correct. */
export interface UploadPredictResponse {
  filename: string;
  predicted: string;
  confidence: number;
  probabilities: Record<string, number>;
  report: {
    rows_in: number;
    rows_out: number;
    leading_warmup_dropped: number;
    has_secondsElapsed: boolean;
    partial_rows_dropped: number;
  };
}
