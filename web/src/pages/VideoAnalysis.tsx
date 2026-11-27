import axios from "axios";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import useSWR from "swr";
import { baseUrl } from "@/api/baseUrl";
import { Button } from "@/components/ui/button";

type Analysis = {
  id: string;
  name: string;
  model?: string;
  model_id?: string;
  sample_fps?: number;
  completed_at: number;
  sample_count: number;
  box_count: number;
  unique_tracks_total?: number;
  unique_tracks_by_label?: Record<string, number>;
  boxes_by_label: Record<string, number>;
  median_inference_ms: number;
  end_to_end_fps?: number;
  incidents?: Array<{
    event_id: string;
    timestamp_s: number;
    category: string;
    state: string;
    severity: string;
    evidence_score: number;
    contributing_evidence: string[];
    snapshot_file?: string | null;
    clip_file?: string | null;
  }>;
  local_test_sos?: {
    incident_id: string;
    status: string;
    notification_status: string;
    note: string;
  } | null;
  risk_events?: Array<{ risk_id: string; near_miss: boolean; risk_level: string }>;
  safety_alerts?: Array<{ alert_id: string; category: string }>;
};

type AnalysisState = {
  id?: string;
  status:
    | "idle"
    | "running"
    | "completed"
    | "invalid"
    | "failed"
    | "interrupted"
    | "cancelled";
  progress?: number;
  stage?: string;
};

type RegistryModel = {
  id: string;
  name: string;
  input_width: number;
  availability: { available: boolean; status: string };
};

export default function VideoAnalysis() {
  const { t, i18n } = useTranslation(["views/exports", "objects"]);
  const { data, error, mutate } = useSWR<Analysis[]>("drishti/analysis");
  const { data: modelResp } = useSWR<{ models: RegistryModel[] }>(
    "drishti/v2/models",
  );
  const {
    data: job,
    error: jobError,
    mutate: updateJob,
  } = useSWR<AnalysisState>("drishti/analysis/status", {
    refreshInterval: (state) => (state?.status === "running" ? 2000 : 10000),
  });

  useEffect(() => {
    if (job?.status === "completed") {
      void mutate();
    }
  }, [job?.id, job?.status, mutate]);

  const [selection, select] = useState("");
  const [file, setFile] = useState<File>();
  const [modelId, setModelId] = useState("yolox-nano-416");
  const [sampleFps, setSampleFps] = useState(10);
  const [sending, setSending] = useState(false);
  const busy = sending || job?.status === "running";
  const [issue, setIssue] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const item = data?.find((x) => x.id === selection) ?? data?.[0];
  const url = item && `${baseUrl}api/drishti/analysis/${item.id}`;

  useEffect(() => {
    document.title = t("analysis.title") + " | DRISHTI AI";
  }, [t]);

  async function submit() {
    if (!file || busy) return;
    setIssue("");
    if (file.size > 32 * 1024 * 1024) {
      setIssue("size");
      return;
    }
    setSending(true);
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await axios.post<AnalysisState>("drishti/analysis", body, {
        params: {
          background: true,
          model_id: modelId,
          sample_fps: sampleFps,
        },
      });
      await updateJob(res.data, { revalidate: true });
      select(res.data.id ?? "");
      setFile(undefined);
      if (input.current) input.current.value = "";
    } catch (e) {
      const status = axios.isAxiosError(e) ? e.response?.status : undefined;
      setIssue(
        status === 409
          ? "conflict"
          : status === 413
            ? "size"
            : status === 422
              ? "invalid"
              : "failed",
      );
    } finally {
      setSending(false);
    }
  }

  async function cancelActive() {
    await axios.post("drishti/analysis/cancel");
    await updateJob();
  }

  return (
    <main className="mx-auto flex size-full max-w-6xl flex-col gap-5 overflow-y-auto px-4 py-6 md:px-8">
      <h1 className="text-2xl font-semibold">{t("analysis.title")}</h1>
      <section
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (busy) return;
          const dropped = e.dataTransfer.files?.[0];
          if (dropped) {
            setIssue("");
            setFile(dropped);
          }
        }}
        className="space-y-4 rounded-xl border border-dashed bg-card p-5"
      >
        <label htmlFor="clip" className="block text-sm">
          {t("analysis.instructions")} (Drag &amp; drop MP4 or browse)
        </label>
        <div className="flex flex-wrap items-center gap-3">
          <input
            id="clip"
            type="file"
            accept="video/mp4,.mp4"
            ref={input}
            disabled={busy}
            onChange={(e) => {
              setIssue("");
              setFile(e.target.files?.[0]);
            }}
          />
          <select
            aria-label="Detection model"
            value={modelId}
            disabled={busy}
            onChange={(e) => setModelId(e.target.value)}
            className="rounded-md border bg-background px-2.5 py-1.5 text-sm"
          >
            {(modelResp?.models ?? []).map((m) => (
              <option
                key={m.id}
                value={m.id}
                disabled={!m.availability.available}
              >
                {m.name} ({m.input_width}px)
                {m.availability.available ? "" : " — Unavailable"}
              </option>
            ))}
          </select>
          <select
            aria-label="Sampling FPS"
            value={sampleFps}
            disabled={busy}
            onChange={(e) => setSampleFps(Number(e.target.value))}
            className="rounded-md border bg-background px-2.5 py-1.5 text-sm"
          >
            <option value={2}>2 FPS (Fast)</option>
            <option value={5}>5 FPS (Temporal Incidents)</option>
            <option value={10}>10 FPS (High Detail)</option>
          </select>
          <Button disabled={busy || !file} onClick={() => void submit()}>
            {t(busy ? "analysis.busy" : "analysis.run")}
          </Button>
          {job?.status === "running" && (
            <Button variant="destructive" onClick={() => void cancelActive()}>
              Cancel
            </Button>
          )}
        </div>
        {job?.status === "running" && (
          <p role="status">
            {t("analysis.processing")} ({job.progress ?? 0}% ·{" "}
            {job.stage ?? "running"})
          </p>
        )}
        {job &&
          ["invalid", "failed", "interrupted", "cancelled"].includes(
            job.status,
          ) && (
            <p role="alert">
              {job.status === "cancelled"
                ? "Analysis cancelled."
                : t(`analysis.${job.status}`)}
            </p>
          )}
        {issue && (
          <p role="alert" className="text-destructive">
            {t(`analysis.${issue}`)}
          </p>
        )}
      </section>

      {(error || jobError) && <p role="alert">{t("analysis.failed")}</p>}
      {!error && !data && <p>{t("analysis.loading")}</p>}
      {data?.length === 0 && <p>{t("analysis.empty")}</p>}

      {item && url && (
        <div className="grid gap-5 lg:grid-cols-[15rem_1fr]">
          <section className="space-y-2">
            <h2 className="font-semibold">{t("analysis.history")}</h2>
            {data?.map((row) => (
              <button
                key={row.id}
                onClick={() => select(row.id)}
                aria-pressed={row.id === item.id}
                className={`block w-full rounded-lg border p-3 text-start ${row.id === item.id ? "bg-secondary" : ""}`}
              >
                <span className="block truncate">{row.name}</span>
                <small>
                  {new Date(row.completed_at * 1000).toLocaleString(
                    i18n.language,
                  )}
                </small>
              </button>
            ))}
          </section>

          <section className="min-w-0 space-y-3">
            <video
              controls
              playsInline
              preload="metadata"
              className="w-full rounded-xl bg-black"
              src={`${url}/review.mp4`}
            />
            <p className="text-sm">
              <strong>Model:</strong> {item.model ?? "YOLOX-Nano"} ·{" "}
              <strong>Frames:</strong> {item.sample_count} (
              {item.sample_fps ?? 2} FPS) · <strong>Unique Tracks:</strong>{" "}
              {item.unique_tracks_total ?? "N/A"} ·{" "}
              <strong>Per-Frame Boxes:</strong> {item.box_count} ·{" "}
              <strong>Median Inference:</strong> {item.median_inference_ms} ms
            </p>
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="rounded bg-secondary px-2 py-1">
                Incidents: {item.incidents?.length ?? 0}
              </span>
              <span className="rounded bg-secondary px-2 py-1">
                Risk Events: {item.risk_events?.length ?? 0}
              </span>
              <span className="rounded bg-secondary px-2 py-1">
                Safety Alerts: {item.safety_alerts?.length ?? 0}
              </span>
            </div>
            {(item.incidents?.length ?? 0) > 0 && (
              <div className="space-y-2 rounded-xl border border-destructive/40 bg-destructive/10 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <strong className="text-sm text-destructive">
                    Accident / Collision Incidents Detected (
                    {item.incidents?.length})
                  </strong>
                  {item.local_test_sos && (
                    <span className="rounded bg-amber-500/20 px-2 py-0.5 text-xs text-amber-400">
                      {item.local_test_sos.status} ·{" "}
                      {item.local_test_sos.notification_status}
                    </span>
                  )}
                </div>
                {item.incidents?.map((inc) => (
                  <div
                    key={inc.event_id}
                    className="space-y-1.5 rounded-lg border bg-card p-3 text-xs"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2 font-semibold">
                      <span>
                        {inc.state} ({inc.category}) @ {inc.timestamp_s}s
                      </span>
                      <span>
                        Severity: {inc.severity} · Evidence Score:{" "}
                        {inc.evidence_score}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {inc.contributing_evidence.map((ev) => (
                        <span
                          key={ev}
                          className="rounded bg-secondary px-1.5 py-0.5"
                        >
                          {ev}
                        </span>
                      ))}
                    </div>
                    <div className="flex flex-wrap gap-3 pt-1 underline">
                      {inc.snapshot_file && (
                        <a
                          href={`${url}/${inc.snapshot_file}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View Incident Snapshot
                        </a>
                      )}
                      {inc.clip_file && (
                        <a
                          href={`${url}/${inc.clip_file}`}
                          download={inc.clip_file}
                        >
                          Download Evidence Clip
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              {Object.entries(
                item.unique_tracks_by_label ?? item.boxes_by_label,
              ).map(([label, count]) => (
                <span
                  key={label}
                  className="rounded-md bg-secondary px-2 py-1 text-sm"
                >
                  {t(label.replaceAll(" ", "_"), {
                    ns: "objects",
                    defaultValue: label,
                  })}{" "}
                  · {count}
                </span>
              ))}
            </div>
            <div className="flex flex-wrap gap-4 text-sm underline">
              <a href={`${url}/review.mp4`} download="drishti-review.mp4">
                {t("analysis.video")}
              </a>
              <a href={`${url}/report.json`} download="drishti-report.json">
                {t("analysis.report")}
              </a>
              <a href={`${url}/report.csv`} download="drishti-report.csv">
                Download CSV
              </a>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
