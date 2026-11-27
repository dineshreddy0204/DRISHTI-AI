import axios from "axios";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import useSWR from "swr";
import { baseUrl } from "@/api/baseUrl";
import { Button } from "@/components/ui/button";
import { useIsAdmin } from "@/hooks/use-is-admin";

type CommandCenterData = {
  test_mode: boolean;
  cameras_v2: Array<{
    camera_id: string;
    camera_number: number;
    camera_name: string;
    location: string;
    source_type: string;
    enabled_modules: string[];
  }>;
  active_sos: {
    sos_id: string;
    incident_id: string;
    camera_id: string;
    state: string;
    remaining_s?: number;
    notification_status: string;
  } | null;
  incidents: Array<{
    incident_id: string;
    camera_id: string;
    camera_number: number;
    camera_name: string;
    camera_location: string;
    source_type: string;
    timestamp: number;
    event_type: string;
    risk_level: string;
    contributing_evidence: string[];
    model_version: string;
    rule_version: string;
    evidence_reliability: string;
    snapshot_path: string | null;
    review_status: string;
    sos_status: string;
    notification_status: string;
    is_test_mode: number;
    is_historical: number;
  }>;
  public_safety_capabilities: Record<
    string,
    { enabled: boolean; reason: string }
  >;
  contacts: Array<{
    contact_id: string;
    name: string;
    channel: string;
    target_masked: string;
    enabled: number;
  }>;
  system: {
    arch: string;
    cpu_percent: number;
    ram_percent: number;
    ram_used_gb: number;
    ram_total_gb: number;
  };
};

export default function CommandCenter() {
  const isAdmin = useIsAdmin();
  const { data, mutate } = useSWR<CommandCenterData>(
    "drishti/v2/command-center",
    { refreshInterval: 3000 },
  );

  const [contactName, setContactName] = useState("");
  const [contactChannel, setContactChannel] = useState<"sms" | "email" | "whatsapp">("sms");
  const [contactTarget, setContactTarget] = useState("");

  useEffect(() => {
    document.title = "Emergency Command Center | DRISHTI AI V2";
  }, []);

  async function respondSos(
    action: "i_am_okay" | "cancel" | "acknowledge" | "confirm_emergency",
  ) {
    if (!data?.active_sos) return;
    await axios.post(`drishti/v2/sos/${data.active_sos.sos_id}/respond`, {
      action,
    });
    await mutate();
  }

  async function triggerManualSos() {
    await axios.post("drishti/v2/sos/manual", {
      camera_id: "CAM-001",
      note: "Manual Test SOS from Emergency Command Center",
    });
    await mutate();
  }

  async function reviewIncident(
    incidentId: string,
    review_status: "HUMAN_CONFIRMED_EMERGENCY" | "DISMISSED" | "ACKNOWLEDGED",
  ) {
    await axios.post(
      `drishti/v2/command-center/incidents/${incidentId}/review`,
      { review_status },
    );
    await mutate();
  }

  async function addContact() {
    if (!contactName.trim() || !contactTarget.trim()) return;
    await axios.post("drishti/v2/contacts", {
      name: contactName.trim(),
      channel: contactChannel,
      target: contactTarget.trim(),
    });
    setContactName("");
    setContactTarget("");
    await mutate();
  }

  const activeSos = data?.active_sos;

  return (
    <main className="mx-auto flex size-full max-w-7xl flex-col gap-5 overflow-y-auto px-4 py-6 md:px-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">
            Emergency Command Center — DRISHTI AI V2
          </h1>
          <p className="text-sm text-muted-foreground">
            Multi-camera incident feed, SOS workflow, public safety capabilities
            &amp; ARM64 health.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded bg-amber-500/20 px-2.5 py-1 text-xs font-semibold text-amber-400">
            {data?.test_mode ? "TEST MODE ENABLED" : "PRODUCTION MODE"}
          </span>
          <Link to="/human-safety">
            <Button size="sm" variant="outline">
              Open Webcam Safety Test
            </Button>
          </Link>
          <Button size="sm" variant="destructive" onClick={() => void triggerManualSos()}>
            Manual SOS
          </Button>
        </div>
      </div>

      {activeSos && (
        <section className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-destructive bg-destructive/15 p-4">
          <div>
            <div className="text-lg font-bold text-destructive">
              ACTIVE SOS: {activeSos.state} · Camera {activeSos.camera_id}
            </div>
            <div className="text-sm">
              Incident: {activeSos.incident_id} · Countdown:{" "}
              {activeSos.remaining_s ?? 0}s · Dispatch:{" "}
              <strong>{activeSos.notification_status}</strong>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={() => void respondSos("i_am_okay")}>
              I Am Okay
            </Button>
            <Button size="sm" variant="outline" onClick={() => void respondSos("acknowledge")}>
              Acknowledge
            </Button>
            <Button size="sm" variant="outline" onClick={() => void respondSos("cancel")}>
              Cancel
            </Button>
            <Button size="sm" variant="destructive" onClick={() => void respondSos("confirm_emergency")}>
              Confirm Emergency
            </Button>
          </div>
        </section>
      )}

      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {(data?.cameras_v2 ?? []).map((cam) => {
          const highlighted = activeSos?.camera_id === cam.camera_id;
          return (
            <div
              key={cam.camera_id}
              className={`space-y-1.5 rounded-xl border bg-card p-4 ${highlighted ? "border-destructive ring-2 ring-destructive/40" : ""}`}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold">
                  #{cam.camera_number} {cam.camera_name} ({cam.camera_id})
                </span>
                <span className="rounded bg-secondary px-2 py-0.5 text-xs">
                  {cam.source_type}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Location: {cam.location}
              </p>
              <p className="text-xs text-muted-foreground">
                Modules: {cam.enabled_modules.join(", ")}
              </p>
            </div>
          );
        })}
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">
          Persisted Incident Feed ({data?.incidents.length ?? 0})
        </h2>
        {(data?.incidents ?? []).length === 0 ? (
          <div className="rounded-xl border bg-card p-6 text-sm text-muted-foreground">
            No safety or emergency incidents persisted yet.
          </div>
        ) : (
          <div className="space-y-3">
            {(data?.incidents ?? []).map((inc) => (
              <div
                key={inc.incident_id}
                className="grid gap-3 rounded-xl border bg-card p-4 md:grid-cols-[1fr_auto]"
              >
                <div className="space-y-1">
                  <div className="flex flex-wrap items-center gap-2 font-semibold">
                    <span
                      className={`rounded px-2 py-0.5 text-xs ${
                        inc.risk_level === "CRITICAL"
                          ? "bg-destructive/20 text-destructive"
                          : inc.risk_level === "HIGH"
                            ? "bg-amber-500/20 text-amber-400"
                            : "bg-secondary"
                      }`}
                    >
                      {inc.risk_level}
                    </span>
                    <span>
                      {inc.event_type} — #{inc.camera_number} {inc.camera_name}
                    </span>
                    {inc.is_test_mode === 1 && (
                      <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-xs text-amber-400">
                        TEST MODE
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {new Date(inc.timestamp * 1000).toLocaleString()} ·{" "}
                    {inc.source_type} · Model: {inc.model_version} · Review:{" "}
                    {inc.review_status} · Notification:{" "}
                    {inc.notification_status}
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
                  <div className="flex flex-wrap gap-2 pt-1">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        void reviewIncident(inc.incident_id, "ACKNOWLEDGED")
                      }
                    >
                      Acknowledge
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        void reviewIncident(
                          inc.incident_id,
                          "HUMAN_CONFIRMED_EMERGENCY",
                        )
                      }
                    >
                      Confirm Emergency
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        void reviewIncident(inc.incident_id, "DISMISSED")
                      }
                    >
                      Dismiss
                    </Button>
                  </div>
                </div>
                {inc.snapshot_path && (
                  <img
                    src={`${baseUrl}api/drishti/v2/evidence/${inc.incident_id}.jpg`}
                    alt={inc.event_type}
                    className="h-28 w-44 rounded-lg border object-cover"
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="grid gap-5 md:grid-cols-2">
        <div className="space-y-3 rounded-xl border bg-card p-4">
          <h2 className="font-semibold">
            Emergency Contacts (Blocked in Test Mode)
          </h2>
          {isAdmin && (
            <div className="flex flex-wrap gap-2">
              <input
                value={contactName}
                onChange={(e) => setContactName(e.target.value)}
                placeholder="Contact name"
                className="rounded-md border bg-background px-2.5 py-1 text-sm"
              />
              <select
                value={contactChannel}
                onChange={(e) =>
                  setContactChannel(e.target.value as "sms" | "email" | "whatsapp")
                }
                className="rounded-md border bg-background px-2.5 py-1 text-sm"
              >
                <option value="sms">SMS</option>
                <option value="email">Email</option>
                <option value="whatsapp">WhatsApp</option>
              </select>
              <input
                value={contactTarget}
                onChange={(e) => setContactTarget(e.target.value)}
                placeholder="Phone or email"
                className="rounded-md border bg-background px-2.5 py-1 text-sm"
              />
              <Button size="sm" onClick={() => void addContact()}>
                Add
              </Button>
            </div>
          )}
          <div className="space-y-1 text-xs text-muted-foreground">
            {(data?.contacts ?? []).length === 0
              ? "No emergency contacts configured."
              : (data?.contacts ?? []).map((c) => (
                  <div key={c.contact_id}>
                    {c.name} ({c.channel}: {c.target_masked})
                  </div>
                ))}
          </div>
        </div>

        <div className="space-y-2 rounded-xl border bg-card p-4 text-xs">
          <h2 className="text-sm font-semibold">
            System Health &amp; Public Safety Modules
          </h2>
          {data?.system && (
            <div>
              Arch: {data.system.arch} · CPU: {data.system.cpu_percent}% · RAM:{" "}
              {data.system.ram_used_gb}/{data.system.ram_total_gb} GB (
              {data.system.ram_percent}%)
            </div>
          )}
          <div className="grid gap-1.5 pt-1">
            {Object.entries(data?.public_safety_capabilities ?? {}).map(
              ([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span>{k.replaceAll("_", " ")}</span>
                  <span
                    className={
                      v.enabled ? "text-emerald-400" : "text-muted-foreground"
                    }
                  >
                    {v.enabled ? "Active" : "Disabled"}
                  </span>
                </div>
              ),
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

