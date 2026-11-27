import axios from "axios";
import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { useIsAdmin } from "@/hooks/use-is-admin";

type TabId =
  | "cameras"
  | "incidents"
  | "risks"
  | "safety"
  | "calibration"
  | "models";

const ROUTE_TO_TAB: Record<string, TabId> = {
  "/intelligence": "cameras",
  "/accident-intelligence": "incidents",
  "/incident-review": "incidents",
  "/collision-risk": "risks",
  "/traffic-analytics": "cameras",
  "/safety-violations": "safety",
  "/camera-calibration": "calibration",
  "/models": "models",
  "/diagnostics": "models",
};

type CameraHealthRow = {
  camera_id: string;
  source_type: string;
  enabled: boolean;
  inputs: string[];
  detect_resolution: [number, number];
  target_fps: number;
  captured_fps: number;
  processed_fps: number;
  skipped_fps: number;
  effective_detection_fps: number;
  offline_alert: boolean;
  calibrated: boolean;
};

type IncidentRow = {
  event_id: string;
  camera_id: string;
  source_type: string;
  timestamp_s: number;
  category: string;
  state: string;
  severity: string;
  evidence_score: number;
  evidence_score_note: string;
  involved_track_ids: string[];
  contributing_evidence: string[];
  review_status: string;
  analysis_name?: string;
};

type RiskRow = {
  risk_id: string;
  camera_id: string;
  timestamp_s: number;
  track_a: string;
  track_b: string;
  calibrated: boolean;
  unit_space: string;
  relative_speed: number;
  relative_distance: number;
  ttc_s: number | null;
  pet_s: number | null;
  risk_level: string;
  near_miss: boolean;
  uncertainty_note: string;
};

type SafetyResponse = {
  safety_alerts: Array<{
    alert_id: string;
    camera_id: string;
    timestamp_s: number;
    category: string;
    severity: string;
    track_ids: string[];
    evidence: string[];
    note: string;
  }>;
  capabilities: Record<string, { enabled: boolean; reason: string }>;
};

type ModelRow = {
  id: string;
  name: string;
  version: string;
  architecture: string;
  license: string;
  input_width: number;
  input_height: number;
  supported_labels: string[];
  unavailable_labels: string[];
  runtime_provider: string;
  notes: string;
  availability: { status: string; available: boolean; checksum_verified: boolean };
};

export default function TrafficIntelligence() {
  const location = useLocation();
  const isAdmin = useIsAdmin();
  const [tab, setTab] = useState<TabId>(
    ROUTE_TO_TAB[location.pathname] ?? "cameras",
  );

  useEffect(() => {
    const mapped = ROUTE_TO_TAB[location.pathname];
    if (mapped) setTab(mapped);
  }, [location.pathname]);

  useEffect(() => {
    document.title = "Traffic Intelligence | DRISHTI AI V2";
  }, []);

  const [severityFilter, setSeverityFilter] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [streamResult, setStreamResult] = useState<Record<string, unknown> | null>(null);
  const [testingStream, setTestingStream] = useState(false);

  const [calCam, setCalCam] = useState("hyderabad_replay");
  const [laneDeg, setLaneDeg] = useState("");
  const [speedLimit, setSpeedLimit] = useState("");
  const [calSavedMsg, setCalSavedMsg] = useState("");

  const { data: camData, mutate: refreshCams } = useSWR<{
    cameras: CameraHealthRow[];
  }>("drishti/v2/cameras", { refreshInterval: 5000 });

  const incQuery = new URLSearchParams();
  if (severityFilter) incQuery.set("severity", severityFilter);
  if (statusFilter) incQuery.set("review_status", statusFilter);
  const { data: incData, mutate: refreshInc } = useSWR<{
    incidents: IncidentRow[];
  }>(`drishti/v2/incidents?${incQuery.toString()}`);

  const { data: riskData } = useSWR<{ risk_events: RiskRow[] }>(
    "drishti/v2/risks",
  );
  const { data: safetyData } = useSWR<SafetyResponse>("drishti/v2/safety");
  const { data: modelData } = useSWR<{
    runtime_providers: string[];
    models: ModelRow[];
  }>("drishti/v2/models");
  const { data: diagData } = useSWR<{
    system: Record<string, unknown>;
    validation: Record<string, unknown>;
    completed_analyses: number;
  }>("drishti/v2/diagnostics", { refreshInterval: 10000 });

  async function testStream() {
    if (!streamUrl.trim()) return;
    setTestingStream(true);
    setStreamResult(null);
    try {
      const res = await axios.post("drishti/v2/cameras/test", {
        url: streamUrl.trim(),
      });
      setStreamResult(res.data);
    } catch (e) {
      setStreamResult({ reachable: false, error: String(e) });
    } finally {
      setTestingStream(false);
    }
  }

  async function reviewIncident(
    eventId: string,
    decision: "Human-confirmed accident" | "Dismissed incident" | "acknowledged",
  ) {
    await axios.post(`drishti/v2/incidents/${eventId}/review`, {
      decision,
      note: "",
    });
    await refreshInc();
  }

  async function saveCalibration() {
    setCalSavedMsg("");
    await axios.put(`drishti/v2/calibration/${calCam}`, {
      calibrated: false,
      lane_direction_deg: laneDeg ? Number(laneDeg) : null,
      speed_limit_kmh: speedLimit ? Number(speedLimit) : null,
    });
    setCalSavedMsg(`Saved calibration for ${calCam}`);
    await refreshCams();
  }

  const tabs: Array<{ id: TabId; label: string }> = [
    { id: "cameras", label: "Live Traffic & CCTV" },
    { id: "incidents", label: "Accident Intelligence" },
    { id: "risks", label: "Collision Risk" },
    { id: "safety", label: "Safety Analytics" },
    { id: "calibration", label: "Calibration" },
    { id: "models", label: "Models & Diagnostics" },
  ];

  return (
    <main className="mx-auto flex size-full max-w-7xl flex-col gap-5 overflow-y-auto px-4 py-6 md:px-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">
            DRISHTI AI V2 — Traffic Intelligence
          </h1>
          <p className="text-sm text-muted-foreground">
            Real-time telemetry, temporal incident review, collision risk, and
            ARM64 ONNX diagnostics.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {tabs.map((item) => (
            <Button
              key={item.id}
              variant={tab === item.id ? "default" : "outline"}
              size="sm"
              onClick={() => setTab(item.id)}
            >
              {item.label}
            </Button>
          ))}
        </div>
      </div>

      {tab === "cameras" && (
        <section className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            {(camData?.cameras ?? []).map((cam) => (
              <div
                key={cam.camera_id}
                className="space-y-2 rounded-xl border bg-card p-4"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold">{cam.camera_id}</span>
                  <span
                    className={`rounded px-2 py-0.5 text-xs font-medium ${
                      cam.source_type === "LIVE CCTV"
                        ? "bg-emerald-500/20 text-emerald-400"
                        : cam.source_type === "CAMERA OFFLINE"
                          ? "bg-destructive/20 text-destructive"
                          : "bg-amber-500/20 text-amber-400"
                    }`}
                  >
                    {cam.source_type}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Resolution: {cam.detect_resolution[0]}x
                  {cam.detect_resolution[1]} · Target {cam.target_fps} FPS ·{" "}
                  {cam.calibrated ? "Calibrated" : "Uncalibrated (image-space)"}
                </p>
                <div className="grid grid-cols-4 gap-2 text-center text-xs">
                  <div className="rounded bg-secondary p-2">
                    <div className="font-semibold">{cam.captured_fps}</div>
                    <div>Captured FPS</div>
                  </div>
                  <div className="rounded bg-secondary p-2">
                    <div className="font-semibold">{cam.processed_fps}</div>
                    <div>Processed FPS</div>
                  </div>
                  <div className="rounded bg-secondary p-2">
                    <div className="font-semibold">{cam.skipped_fps}</div>
                    <div>Skipped FPS</div>
                  </div>
                  <div className="rounded bg-secondary p-2">
                    <div className="font-semibold">
                      {cam.effective_detection_fps}
                    </div>
                    <div>Detection FPS</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {isAdmin && (
            <div className="space-y-3 rounded-xl border bg-card p-4">
              <h2 className="font-semibold">
                RTSP / IP CCTV Stream Connection Test
              </h2>
              <div className="flex flex-wrap gap-2">
                <input
                  type="text"
                  value={streamUrl}
                  onChange={(e) => setStreamUrl(e.target.value)}
                  placeholder="rtsp://user:pass@192.168.1.50:554/stream1"
                  className="min-w-[280px] flex-1 rounded-md border bg-background px-3 py-1.5 text-sm"
                />
                <Button
                  disabled={testingStream || !streamUrl.trim()}
                  onClick={() => void testStream()}
                >
                  {testingStream ? "Testing..." : "Test Connection"}
                </Button>
              </div>
              {streamResult && (
                <pre className="overflow-x-auto rounded bg-secondary p-3 text-xs">
                  {JSON.stringify(streamResult, null, 2)}
                </pre>
              )}
            </div>
          )}
        </section>
      )}

      {tab === "incidents" && (
        <section className="space-y-4">
          <div className="flex flex-wrap gap-3">
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            >
              <option value="">All severities</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            >
              <option value="">All review statuses</option>
              <option value="unreviewed">Unreviewed</option>
              <option value="Human-confirmed accident">Human-confirmed</option>
              <option value="Dismissed incident">Dismissed</option>
            </select>
          </div>
          {(incData?.incidents ?? []).length === 0 ? (
            <div className="rounded-xl border bg-card p-6 text-sm text-muted-foreground">
              No accident or collision incidents recorded yet. Normal traffic
              does not generate synthetic alerts.
            </div>
          ) : (
            <div className="space-y-3">
              {(incData?.incidents ?? []).map((inc) => (
                <div
                  key={inc.event_id}
                  className="space-y-2 rounded-xl border bg-card p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-semibold">
                      {inc.state} ({inc.severity}) — {inc.camera_id} @{" "}
                      {inc.timestamp_s}s
                    </span>
                    <span className="text-xs text-muted-foreground">
                      Evidence score: {inc.evidence_score} · {inc.review_status}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Tracks: {inc.involved_track_ids.join(", ")} ·{" "}
                    {inc.evidence_score_note}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {inc.contributing_evidence.map((ev) => (
                      <span
                        key={ev}
                        className="rounded bg-secondary px-2 py-0.5 text-xs"
                      >
                        {ev}
                      </span>
                    ))}
                  </div>
                  {isAdmin && (
                    <div className="flex flex-wrap gap-2 pt-1">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() =>
                          void reviewIncident(
                            inc.event_id,
                            "Human-confirmed accident",
                          )
                        }
                      >
                        Confirm Accident
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() =>
                          void reviewIncident(inc.event_id, "Dismissed incident")
                        }
                      >
                        Dismiss False Alarm
                      </Button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {tab === "risks" && (
        <section className="space-y-3">
          {(riskData?.risk_events ?? []).length === 0 ? (
            <div className="rounded-xl border bg-card p-6 text-sm text-muted-foreground">
              No near-miss or predictive collision-risk events recorded yet.
            </div>
          ) : (
            (riskData?.risk_events ?? []).map((rk) => (
              <div
                key={rk.risk_id}
                className="space-y-1 rounded-xl border bg-card p-4 text-sm"
              >
                <div className="flex items-center justify-between font-semibold">
                  <span>
                    {rk.near_miss ? "Near-Miss" : "Collision Risk"} (
                    {rk.risk_level}) — {rk.track_a} &amp; {rk.track_b}
                  </span>
                  <span>
                    TTC: {rk.ttc_s ?? "N/A"}s · {rk.unit_space}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  {rk.uncertainty_note}
                </p>
              </div>
            ))
          )}
        </section>
      )}

      {tab === "safety" && (
        <section className="space-y-4">
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {Object.entries(safetyData?.capabilities ?? {}).map(
              ([name, cap]) => (
                <div key={name} className="rounded-xl border bg-card p-4">
                  <div className="flex items-center justify-between">
                    <span className="font-medium">
                      {name.replaceAll("_", " ")}
                    </span>
                    <span
                      className={`rounded px-2 py-0.5 text-xs ${
                        cap.enabled
                          ? "bg-emerald-500/20 text-emerald-400"
                          : "bg-secondary text-muted-foreground"
                      }`}
                    >
                      {cap.enabled ? "Enabled" : "Disabled"}
                    </span>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {cap.reason}
                  </p>
                </div>
              ),
            )}
          </div>
          {(safetyData?.safety_alerts ?? []).length === 0 ? (
            <div className="rounded-xl border bg-card p-6 text-sm text-muted-foreground">
              No safety alerts recorded yet.
            </div>
          ) : (
            (safetyData?.safety_alerts ?? []).map((al) => (
              <div
                key={al.alert_id}
                className="rounded-xl border bg-card p-4 text-sm"
              >
                <div className="font-semibold">
                  {al.category} ({al.severity}) @ {al.timestamp_s}s
                </div>
                <p className="text-xs text-muted-foreground">{al.note}</p>
              </div>
            ))
          )}
        </section>
      )}

      {tab === "calibration" && (
        <section className="space-y-4 rounded-xl border bg-card p-5">
          <h2 className="font-semibold">
            Camera Calibration &amp; Lane Configuration
          </h2>
          <p className="text-xs text-muted-foreground">
            Without a 4-point ground-plane homography, speed and separation
            remain explicitly labeled as uncalibrated image-space pixels.
          </p>
          <div className="grid gap-3 sm:grid-cols-3">
            <input
              value={calCam}
              onChange={(e) => setCalCam(e.target.value)}
              placeholder="Camera ID"
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            />
            <input
              value={laneDeg}
              onChange={(e) => setLaneDeg(e.target.value)}
              placeholder="Lane direction (deg, e.g. 90)"
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            />
            <input
              value={speedLimit}
              onChange={(e) => setSpeedLimit(e.target.value)}
              placeholder="Speed limit (km/h)"
              className="rounded-md border bg-background px-3 py-1.5 text-sm"
            />
          </div>
          {isAdmin && (
            <Button onClick={() => void saveCalibration()}>
              Save Calibration
            </Button>
          )}
          {calSavedMsg && <p className="text-xs text-emerald-400">{calSavedMsg}</p>}
        </section>
      )}

      {tab === "models" && (
        <section className="space-y-4">
          {isAdmin && (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-card p-4">
              <div>
                <div className="font-semibold">
                  Historical Recording Reanalysis
                </div>
                <p className="text-xs text-muted-foreground">
                  Re-run the bundled traffic clip with YOLOX-S (640x640) to
                  compare recall and unique tracks against legacy YOLOX-Nano.
                </p>
              </div>
              <Button
                size="sm"
                onClick={() =>
                  void axios.post("drishti/v2/historical/reanalyze", {
                    model_id: "yolox-s-640",
                    sample_fps: 5,
                  })
                }
              >
                Reanalyze with YOLOX-S (640px)
              </Button>
            </div>
          )}
          {diagData && (
            <div className="rounded-xl border bg-card p-4">
              <h2 className="mb-2 font-semibold">
                Runtime &amp; Hardware Diagnostics
              </h2>
              <pre className="overflow-x-auto text-xs">
                {JSON.stringify(diagData.system, null, 2)}
              </pre>
            </div>
          )}
          <div className="grid gap-3 md:grid-cols-2">
            {(modelData?.models ?? []).map((m) => (
              <div key={m.id} className="space-y-1.5 rounded-xl border bg-card p-4">
                <div className="flex items-center justify-between">
                  <span className="font-semibold">
                    {m.name} ({m.architecture})
                  </span>
                  <span
                    className={`rounded px-2 py-0.5 text-xs ${
                      m.availability.available
                        ? "bg-emerald-500/20 text-emerald-400"
                        : "bg-amber-500/20 text-amber-400"
                    }`}
                  >
                    {m.availability.status}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Input: {m.input_width}x{m.input_height} · License: {m.license}{" "}
                  · Provider: {m.runtime_provider}
                </p>
                <p className="text-xs text-muted-foreground">{m.notes}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </main>
  );
}

