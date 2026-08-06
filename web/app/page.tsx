"use client";

import { useEffect, useRef, useState } from "react";
import {
  API_BASE_URL,
  type PredictResponse,
  type Recording,
  type RecordingsResponse,
  type UploadPredictResponse,
} from "@/lib/config";

const SAMPLE_RATE_HZ = 100;

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

/* Library filenames carry a long timestamp suffix; the leading id is enough to
   identify a recording, and the meta line only shows it once. */
const shortId = (filename: string) =>
  filename.replace(/\.csv$/i, "").replace(/-\d{4}-\d{2}-\d{2}_[\d-]+$/, "");

/* The backend sleeps on the free tier, so a request that never lands usually
   means the instance is still spinning up rather than that anything is broken. */
const coldStartMessage = (detail: string) =>
  `Could not reach the backend (${detail}). The server sleeps after a period of inactivity, and the first request can take up to a minute to wake it. Wait a moment, then try again.`;

/** Read FastAPI's {"detail": ...} body, which may be a string or a validation list. */
async function backendDetail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) {
      return body.detail.map((d: { msg?: string }) => d.msg ?? "invalid input").join("; ");
    }
  } catch {
    /* non-JSON error body; fall through to the status code */
  }
  return `the server responded with HTTP ${response.status}`;
}

/* One shape for both prediction sources, so the result renders the same way.
   Uploads simply carry no trueLabel, and those fields stay hidden. */
interface Outcome {
  filename: string;
  predicted: string;
  confidence: number;
  probabilities: Record<string, number>;
  samples?: number;
  trueLabel?: string;
  correct?: boolean;
}

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
  const [outcome, setOutcome] = useState<Outcome | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageBroken, setImageBroken] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

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
    setUploadName("");

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: selected }),
      });
      // A status code means the server answered, so it is awake - report the
      // status rather than blaming a cold start.
      if (!response.ok) throw new Error(await backendDetail(response));
      const data = (await response.json()) as PredictResponse;
      setOutcome({
        filename: data.filename,
        predicted: data.predicted,
        confidence: data.confidence,
        probabilities: data.probabilities,
        samples: data.n_samples ?? undefined,
        trueLabel: data.true_label,
        correct: data.correct,
      });
    } catch (err) {
      setOutcome(null);
      // fetch() rejects with a TypeError when the request never lands, which on
      // the free tier usually means the instance is still spinning up.
      setError(
        err instanceof TypeError
          ? coldStartMessage((err as Error).message)
          : `Prediction failed: ${(err as Error).message}.`,
      );
    } finally {
      setLoading(false);
    }
  }

  async function upload(file: File) {
    setUploading(true);
    setError(null);
    setImageBroken(false);
    setUploadName(file.name);

    const body = new FormData();
    body.append("file", file);

    try {
      const response = await fetch(`${API_BASE_URL}/predict-upload`, {
        method: "POST",
        body,
      });
      // The backend explains exactly why a file was rejected - surface that
      // verbatim rather than replacing it with a generic failure.
      if (!response.ok) throw new Error(await backendDetail(response));
      const data = (await response.json()) as UploadPredictResponse;
      setOutcome({
        filename: data.filename,
        predicted: data.predicted,
        confidence: data.confidence,
        probabilities: data.probabilities,
        samples: data.report?.rows_out,
      });
    } catch (err) {
      setOutcome(null);
      setError(
        err instanceof TypeError
          ? coldStartMessage((err as Error).message)
          : `Upload rejected: ${(err as Error).message}`,
      );
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  // Highest probability first, so the winning class leads the list.
  const ranked = outcome
    ? Object.entries(outcome.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  const busy = loading || uploading;
  const seconds = outcome?.samples
    ? Math.round(outcome.samples / SAMPLE_RATE_HZ)
    : null;

  return (
    <div
      className={`page${dragging ? " dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        if (!busy) setDragging(true);
      }}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const dropped = event.dataTransfer.files?.[0];
        if (dropped && !busy) void upload(dropped);
      }}
    >
      <header>
        <span className="wordmark">Exercise Detection</span>
        <nav>
          <a href="#how-it-works">How it works</a>
        </nav>
      </header>

      <main>
        <section aria-label="Classification result">
          <p className="eyebrow">
            {outcome ? "Session analysis" : "Wrist sensor classification"}
          </p>

          {outcome ? (
            <>
              <h1 className="serif">{exerciseName(outcome.predicted)}.</h1>

              <p className="confidence">
                {pct(outcome.confidence)} <span>model confidence</span>
              </p>

              <p className="meta">
                <b>{shortId(outcome.filename)}</b>
                {outcome.samples ? (
                  <>
                    {" · "}
                    {seconds} s{" · "}
                    {SAMPLE_RATE_HZ} Hz{" · "}
                    {outcome.samples.toLocaleString()} samples
                  </>
                ) : null}
                {" · "}
                {outcome.predicted}
                {outcome.trueLabel !== undefined && (
                  <>
                    {" · actual "}
                    {outcome.trueLabel}
                    {" · "}
                    <span className={outcome.correct ? "ok" : "off"}>
                      {outcome.correct ? "Match" : "Mismatch"}
                    </span>
                  </>
                )}
              </p>

              <p className="prob-label">
                Probability across {ranked.length} exercises
              </p>

              {ranked.map(([code, value], index) => (
                <div
                  key={code}
                  className={`prob-row${index === 0 ? " winner" : ""}`}
                >
                  <span className="name">{exerciseName(code)}</span>
                  <span className="track">
                    <span className="fill" style={{ width: `${value * 100}%` }} />
                  </span>
                  <span className="val">{(value * 100).toFixed(1)}</span>
                </div>
              ))}
            </>
          ) : (
            <>
              <h1 className="serif">Read the lift from the wrist.</h1>
              <p className="lede">
                Identifies which strength exercise is being performed from
                wrist-worn motion sensor data. Pick a recording from the library,
                or bring your own.
              </p>
            </>
          )}

          <div className="cta-row">
            <div className="field-wrap">
              <select
                className="field"
                value={selected}
                disabled={recordings.length === 0 || busy}
                onChange={(event) => {
                  setSelected(event.target.value);
                  setOutcome(null);
                  setUploadName("");
                }}
                aria-label="Recording"
              >
                {recordings.length === 0 && (
                  <option value="">No recordings loaded</option>
                )}
                {recordings.map((item) => (
                  <option key={item.filename} value={item.filename}>
                    {exerciseName(item.activity)} — {shortId(item.filename)}
                  </option>
                ))}
              </select>
            </div>

            <button
              className="btn btn-solid"
              onClick={predict}
              disabled={!selected || busy}
            >
              {loading ? "Analysing" : "Predict"}
            </button>

            <button
              className="btn"
              onClick={() => fileInput.current?.click()}
              disabled={busy}
            >
              {uploading ? "Reading" : "Upload CSV"}
            </button>

            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              className="file-input"
              onChange={(event) => {
                const chosen = event.target.files?.[0];
                if (chosen) void upload(chosen);
              }}
            />
          </div>

          <p
            className="cta-note"
            title="Requires columns wristMotion_accelerationX/Y/Z, wristMotion_rotationRateX/Y/Z and wristMotion_gravityX/Y/Z"
          >
            {uploadName ? (
              <>
                Uploaded <b>{uploadName}</b>
              </>
            ) : (
              <>Apple Watch wrist CSV, 100 Hz — or drag a file anywhere on this page.</>
            )}
          </p>

          {error && <p className="alert">{error}</p>}
        </section>

        <aside className="arch" aria-label="Reference illustration">
          {outcome && !imageBroken ? (
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={`/images/${outcome.predicted}.png`}
              alt={`Reference demonstration of ${exerciseName(outcome.predicted)}`}
              onError={() => setImageBroken(true)}
            />
          ) : (
            <div className="arch-empty" />
          )}

          <p className="caption-label">Reference</p>
          <p className="caption serif">
            {outcome ? exerciseName(outcome.predicted) : "Awaiting analysis"}
          </p>
        </aside>
      </main>

      <section className="steps" id="how-it-works" aria-label="How it works">
        <span className="steps-title">How it works</span>
        <ol className="steps-grid">
          {STEPS.map((step) => (
            <li key={step.index}>
              <span className="step-index serif">{step.index}</span>
              <span className="step-title">{step.title}</span>
              <p className="step-note">{step.note}</p>
            </li>
          ))}
        </ol>
      </section>

      <footer>
        <span>Exercise Detection · Wrist sensor exercise recognition</span>
        <span>Random forest · 7 classes</span>
      </footer>
    </div>
  );
}
