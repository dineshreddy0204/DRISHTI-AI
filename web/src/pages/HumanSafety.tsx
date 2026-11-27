import axios from "axios";
import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";

type Keypoint = {
  name: string;
  x: number;
  y: number;
  confidence: number;
  visible: boolean;
};

type PersonResult = {
  track_id: string;
  xyxy: [number, number, number, number];
  score: number;
  is_stationary: boolean;
  pose: {
    keypoints: Keypoint[];
    skeleton_edges: Array<[number, number]>;
    reliability: string;
    posture: string;
    torso_angle_deg: number | null;
    latency_ms: number;
  };
  fall: {
    fall_state: string;
    inactivity_s: number;
    evidence: string[];
  };
};

type AnalyzeFrameResponse = {
  camera_id: string;
  camera_number: number;
  source_type: string;
  test_mode: boolean;
  risk_level: string;
  fall_state: string;
  persons: PersonResult[];
  detections: Array<{ label: string; score: number; xyxy: [number, number, number, number] }>;
  visible_objects: number;
  unique_tracks_total: number;
  tamper_alerts: Array<{ event_type: string; risk_level: string }>;
  active_sos: {
    sos_id: string;
    state: string;
    remaining_s?: number;
    notification_status: string;
    test_mode: number;
  } | null;
  latency: {
    detection_ms: number;
    pose_ms: number;
    end_to_end_ms: number;
  };
};

function playLocalAlarmBeep() {
  try {
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new AudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    osc.frequency.setValueAtTime(660, ctx.currentTime + 0.18);
    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.45);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.45);
  } catch {
    // AudioContext blocked before user interaction
  }
}

export default function HumanSafety() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const inFlightRef = useRef(false);

  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDevice, setSelectedDevice] = useState("");
  const [cameraActive, setCameraActive] = useState(false);
  const [detectionEnabled, setDetectionEnabled] = useState(true);
  const [poseEnabled, setPoseEnabled] = useState(true);
  const [alarmMuted, setAlarmMuted] = useState(false);
  const [camError, setCamError] = useState("");
  const [lastResult, setLastResult] = useState<AnalyzeFrameResponse | null>(null);
  const [measuredFps, setMeasuredFps] = useState(0);

  const { data: sosState, mutate: refreshSos } = useSWR<{
    test_mode: boolean;
    active_sos: AnalyzeFrameResponse["active_sos"];
  }>("drishti/v2/sos", { refreshInterval: 1500 });

  useEffect(() => {
    document.title = "Human Safety Live Test | DRISHTI AI V2";
    navigator.mediaDevices
      ?.enumerateDevices()
      .then((list) => setDevices(list.filter((d) => d.kind === "videoinput")))
      .catch(() => {});
  }, []);

  const stopCamera = useCallback(async () => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    setMeasuredFps(0);
    await axios.post("drishti/v2/human-safety/reset").catch(() => {});
  }, []);

  useEffect(() => {
    return () => {
      void stopCamera();
    };
  }, [stopCamera]);

  const drawOverlay = useCallback((res: AnalyzeFrameResponse) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (const det of res.detections) {
      const [x1, y1, x2, y2] = det.xyxy;
      ctx.strokeStyle = "#31c7c9";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = "#31c7c9";
      ctx.font = "12px sans-serif";
      ctx.fillText(`${det.label} ${det.score.toFixed(2)}`, x1, Math.max(14, y1 - 4));
    }

    for (const p of res.persons) {
      const [x1, y1, x2, y2] = p.xyxy;
      ctx.strokeStyle =
        p.fall.fall_state === "NORMAL" ? "#22c55e" : "#ef4444";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = "#ffffff";
      ctx.font = "12px sans-serif";
      ctx.fillText(
        `${p.track_id} · ${p.pose.posture} · ${p.fall.fall_state}`,
        x1,
        Math.max(28, y1 - 18),
      );

      const kps = p.pose.keypoints;
      ctx.strokeStyle = "#facc15";
      ctx.lineWidth = 2;
      for (const [u, v] of p.pose.skeleton_edges) {
        if (kps[u]?.visible && kps[v]?.visible) {
          ctx.beginPath();
          ctx.moveTo(kps[u].x, kps[u].y);
          ctx.lineTo(kps[v].x, kps[v].y);
          ctx.stroke();
        }
      }
      ctx.fillStyle = "#facc15";
      for (const kp of kps) {
        if (kp.visible) {
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    if (!detectionEnabled || inFlightRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;

    inFlightRef.current = true;
    const offscreen = document.createElement("canvas");
    offscreen.width = video.videoWidth || 640;
    offscreen.height = video.videoHeight || 480;
    const ctx = offscreen.getContext("2d");
    if (!ctx) {
      inFlightRef.current = false;
      return;
    }
    ctx.drawImage(video, 0, 0, offscreen.width, offscreen.height);

    const tStart = performance.now();
    offscreen.toBlob(
      async (blob) => {
        if (!blob) {
          inFlightRef.current = false;
          return;
        }
        try {
          const form = new FormData();
          form.append("frame", blob, "frame.jpg");
          const res = await axios.post<AnalyzeFrameResponse>(
            "drishti/v2/human-safety/analyze-frame",
            form,
            {
              params: {
                camera_id: "CAM-001",
                enable_pose: poseEnabled,
              },
            },
          );
          setLastResult(res.data);
          drawOverlay(res.data);
          const dtMs = performance.now() - tStart;
          setMeasuredFps(Number((1000 / Math.max(dtMs, 1)).toFixed(1)));
          if (
            res.data.active_sos &&
            ["COUNTDOWN", "ACTIVATED"].includes(res.data.active_sos.state) &&
            !alarmMuted
          ) {
            playLocalAlarmBeep();
          }
          if (res.data.active_sos) {
            void refreshSos();
          }
        } catch {
          // transient error
        } finally {
          inFlightRef.current = false;
        }
      },
      "image/jpeg",
      0.8,
    );
  }, [alarmMuted, detectionEnabled, drawOverlay, poseEnabled, refreshSos]);

  async function startCamera() {
    setCamError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: selectedDevice
          ? { deviceId: { exact: selectedDevice }, width: 640, height: 480 }
          : { width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
      const list = await navigator.mediaDevices.enumerateDevices();
      setDevices(list.filter((d) => d.kind === "videoinput"));
      timerRef.current = window.setInterval(() => {
        void captureAndAnalyze();
      }, 250);
    } catch (e) {
      setCamError(
        `Cannot access laptop webcam: ${String(e)}. Grant browser camera permission or run scripts/start_webcam_bridge.bat.`,
      );
    }
  }

  useEffect(() => {
    if (!cameraActive) return;
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      void captureAndAnalyze();
    }, 250);
    return () => {
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
    };
  }, [cameraActive, captureAndAnalyze]);

  async function triggerManualSos() {
    if (!alarmMuted) playLocalAlarmBeep();
    await axios.post("drishti/v2/sos/manual", {
      camera_id: "CAM-001",
      note: "Manual Test SOS from Human Safety Dashboard",
    });
    await refreshSos();
  }

  async function respondSos(action: "i_am_okay" | "cancel" | "acknowledge" | "confirm_emergency") {
    const sos = sosState?.active_sos;
    if (!sos) return;
    await axios.post(`drishti/v2/sos/${sos.sos_id}/respond`, { action });
    await refreshSos();
  }

  const primaryPerson = lastResult?.persons?.[0];
  const activeSos = sosState?.active_sos;

  return (
    <main className="mx-auto flex size-full max-w-7xl flex-col gap-4 overflow-y-auto px-4 py-6 md:px-8">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2.5 text-sm">
        <div>
          <strong>SAFE TEST MODE ACTIVE</strong> — Real webcam frames &amp; ONNX
          AI inference enabled; external emergency dispatch is disabled in backend.
        </div>
        <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-xs font-medium text-emerald-400">
          CAM-001 · LIVE WEBCAM
        </span>
      </div>

      {activeSos && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-destructive bg-destructive/15 p-4">
          <div>
            <div className="text-lg font-bold text-destructive">
              ARE YOU OKAY? — {activeSos.state} ({activeSos.sos_id})
            </div>
            <div className="text-sm">
              Camera: CAM-001 (Laptop Webcam) · Countdown:{" "}
              <strong>{activeSos.remaining_s ?? 0}s</strong> · Notification:{" "}
              <strong>{activeSos.notification_status}</strong>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button onClick={() => void respondSos("i_am_okay")}>
              I Am Okay
            </Button>
            <Button variant="outline" onClick={() => void respondSos("cancel")}>
              Cancel Alert
            </Button>
            <Button
              variant="destructive"
              onClick={() => void respondSos("confirm_emergency")}
            >
              Confirm Emergency
            </Button>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <select
          value={selectedDevice}
          onChange={(e) => setSelectedDevice(e.target.value)}
          disabled={cameraActive}
          className="rounded-md border bg-background px-3 py-1.5 text-sm"
        >
          <option value="">Default Laptop Webcam</option>
          {devices.map((d, idx) => (
            <option key={d.deviceId || idx} value={d.deviceId}>
              {d.label || `Webcam ${idx + 1}`}
            </option>
          ))}
        </select>
        {!cameraActive ? (
          <Button onClick={() => void startCamera()}>Start Camera</Button>
        ) : (
          <Button variant="destructive" onClick={() => void stopCamera()}>
            Stop Camera
          </Button>
        )}
        <Button
          variant={detectionEnabled ? "default" : "outline"}
          onClick={() => setDetectionEnabled((v) => !v)}
        >
          {detectionEnabled ? "Disable Detection" : "Enable Detection"}
        </Button>
        <Button
          variant={poseEnabled ? "default" : "outline"}
          onClick={() => setPoseEnabled((v) => !v)}
        >
          {poseEnabled ? "Disable Pose" : "Enable Pose"}
        </Button>
        <Button variant="outline" onClick={() => playLocalAlarmBeep()}>
          Test Alarm
        </Button>
        <Button variant="outline" onClick={() => setAlarmMuted((m) => !m)}>
          {alarmMuted ? "Unmute Alarm" : "Mute Alarm"}
        </Button>
        <Button variant="destructive" onClick={() => void triggerManualSos()}>
          Manual Test SOS
        </Button>
      </div>

      {camError && (
        <p role="alert" className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {camError}
        </p>
      )}

      <div className="grid gap-5 lg:grid-cols-[1fr_22rem]">
        <div className="relative overflow-hidden rounded-xl border bg-black">
          <video
            ref={videoRef}
            playsInline
            muted
            className="block w-full object-contain"
          />
          <canvas
            ref={canvasRef}
            className="pointer-events-none absolute inset-0 size-full object-contain"
          />
        </div>

        <aside className="space-y-3 rounded-xl border bg-card p-4 text-sm">
          <h2 className="text-base font-semibold">Live Telemetry (CAM-001)</h2>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Risk Level</div>
              <div className="font-bold">{lastResult?.risk_level ?? "NORMAL"}</div>
            </div>
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Fall State</div>
              <div className="font-bold">{lastResult?.fall_state ?? "NORMAL"}</div>
            </div>
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Posture</div>
              <div className="font-bold">{primaryPerson?.pose.posture ?? "N/A"}</div>
            </div>
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Torso Angle</div>
              <div className="font-bold">
                {primaryPerson?.pose.torso_angle_deg != null
                  ? `${primaryPerson.pose.torso_angle_deg}°`
                  : "N/A"}
              </div>
            </div>
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Track ID</div>
              <div className="truncate font-bold">
                {primaryPerson?.track_id ?? "None"}
              </div>
            </div>
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Inactivity</div>
              <div className="font-bold">
                {primaryPerson ? `${primaryPerson.fall.inactivity_s}s` : "0s"}
              </div>
            </div>
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Pose Reliability</div>
              <div className="font-bold">
                {primaryPerson?.pose.reliability ?? "N/A"}
              </div>
            </div>
            <div className="rounded bg-secondary p-2">
              <div className="text-muted-foreground">Processed FPS</div>
              <div className="font-bold">{measuredFps} FPS</div>
            </div>
          </div>
          {lastResult && (
            <div className="rounded bg-secondary p-2.5 text-xs">
              Detection: {lastResult.latency.detection_ms} ms · Pose:{" "}
              {lastResult.latency.pose_ms} ms · Total:{" "}
              {lastResult.latency.end_to_end_ms} ms
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}

