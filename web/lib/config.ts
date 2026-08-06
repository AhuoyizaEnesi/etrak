/**
 * Backend location. Swap this one constant (or set NEXT_PUBLIC_API_BASE_URL at
 * build time) to point the frontend at a deployed API instead of localhost.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

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
}
