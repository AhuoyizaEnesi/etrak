"use client";

import { useEffect, useState } from "react";
import {
  API_BASE_URL,
  type PredictResponse,
  type Recording,
  type RecordingsResponse,
} from "@/lib/config";

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;

/* The model works in class codes; people read exercise names. */
const EXERCISE_NAMES: Record<string, string> = {
  APULL: "Cable Lat Pulldown",
  IDBC: "Incline Dumbbell Curl",
  "30DBP": "Incline Dumbbell Press",
  SACLR: "Single-Arm Cable Lateral Raise",
  SAOCTE: "Single-Arm Overhead Cable Triceps Extension",
  MTE: "Overhead Triceps Extension",
  NGCR: "Narrow-Grip Cable Row",
};

/* Fall back to the raw code if a new class ever ships before this map does. */
const exerciseName = (code: string) => EXERCISE_NAMES[code] ?? code;

const STEPS = [
  {
    index: "01",
    title: "Wrist motion recorded",
    note: "A wrist-worn IMU logs acceleration, rotation rate and orientation at 100 Hz for one set.",
  },
  {
    index: "02",
    title: "Motion features extracted",
    note: "The set is reduced to a fixed vector: magnitudes, per-axis spread, orientation, spectral content, periodicity and cross-axis correlation.",
  },
  {
    index: "03",
    title: "Exercise classified",
    note: "A random forest scores the vector against seven exercises and returns a probability for each.",
  },
];

export default function Home() {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [selected, setSelected] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageBroken, setImageBroken] = useState(false);

  useEffect(() => {
    let cancelled = false;

    fetch(`${API_BASE_URL}/recordings`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<RecordingsResponse>;
      })
      .then((data) => {
        if (cancelled) return;
        setRecordings(data.recordings);
        setSelected(data.recordings[0]?.filename ?? "");
      })
      .catch((err: Error) => {
        if (!cancelled) {
          // The backend sleeps on the free tier, so the first request after a
          // quiet spell usually fails while the instance spins back up.
          setError(
            `Could not reach the backend (${err.message}). The server sleeps after a period of inactivity, and the first request can take up to a minute to wake it. Wait a moment, then reload the page.`,
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  async function predict() {
    if (!selected) return;
    setLoading(true);
    setError(null);
    setImageBroken(false);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: selected }),
      });
      // A status code means the server answered, so it is awake - report the
      // status rather than blaming a cold start.
      if (!response.ok) throw new Error(`the server responded with HTTP ${response.status}`);
      setResult((await response.json()) as PredictResponse);
    } catch (err) {
      setResult(null);
      // fetch() rejects with a TypeError when the request never lands, which on
      // the free tier usually means the instance is still spinning up.
      setError(
        err instanceof TypeError
          ? `Could not reach the backend (${err.message}). The server sleeps after a period of inactivity, and the first request can take up to a minute to wake it. Wait a moment, then try again.`
          : `Prediction failed: ${(err as Error).message}.`,
      );
    } finally {
      setLoading(false);
    }
  }

  // Highest probability first, so the winning class leads the list.
  const ranked = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <main className="shell">
      <header className="masthead">
        <h1>Exercise Detection</h1>
        <p>
          Identifies which strength exercise is being performed from wrist-worn
          motion sensor data.
        </p>
      </header>

      <section className="controls">
        <div className="select-wrap">
          <select
            value={selected}
            disabled={recordings.length === 0}
            onChange={(event) => {
              setSelected(event.target.value);
              setResult(null);
            }}
            aria-label="Recording"
          >
            {recordings.length === 0 && <option value="">No recordings loaded</option>}
            {recordings.map((item) => (
              <option key={item.filename} value={item.filename}>
                {item.activity} — {item.filename}
              </option>
            ))}
          </select>
        </div>

        <button
          className="predict"
          onClick={predict}
          disabled={!selected || loading}
        >
          {loading ? "Working" : "Predict"}
        </button>
      </section>

      {error && <p className="status error">{error}</p>}

      {result && (
        <>
          <section className="result">
            <div className="verdict">
              <span className="label">Detected exercise</span>
              <h2>{exerciseName(result.predicted)}</h2>
              <span className="label code-line">Class code — {result.predicted}</span>

              <div className="readout">
                <div>
                  <span className="label">Confidence</span>
                  <div className="value">{pct(result.confidence)}</div>
                </div>
                <div>
                  <span className="label">True label</span>
                  <div className="value value-name">
                    {exerciseName(result.true_label)}
                  </div>
                </div>
                <div>
                  <span className="label">Result</span>
                  <div
                    className={`value ${result.correct ? "match" : "mismatch"}`}
                  >
                    {result.correct ? "Match" : "Mismatch"}
                  </div>
                </div>
              </div>
            </div>

            <div>
              <div className="plate">
                {imageBroken ? (
                  <span className="label">No image for {result.predicted}</span>
                ) : (
                  /* eslint-disable-next-line @next/next/no-img-element */
                  <img
                    src={`/images/${result.predicted}.png`}
                    alt={`Reference demonstration of ${result.predicted}`}
                    onError={() => setImageBroken(true)}
                  />
                )}
              </div>
              <p className="plate-caption label">{result.predicted} — reference</p>
            </div>
          </section>

          <section className="dist">
            <div className="dist-head">
              <span className="label">Probability distribution</span>
              <span className="label">{ranked.length} classes</span>
            </div>

            {ranked.map(([name, value], index) => (
              <div
                key={name}
                className={`bar-row${index === 0 ? " top" : ""}`}
              >
                <span className="name">{name}</span>
                <div className="track">
                  <div className="fill" style={{ width: `${value * 100}%` }} />
                </div>
                <span className="pct">{pct(value)}</span>
              </div>
            ))}
          </section>
        </>
      )}

      <section className="steps" aria-label="How it works">
        <span className="label steps-title">How it works</span>
        <ol className="steps-grid">
          {STEPS.map((step) => (
            <li key={step.index} className="step">
              <span className="step-index">{step.index}</span>
              <span className="step-title">{step.title}</span>
              <p className="step-note">{step.note}</p>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
