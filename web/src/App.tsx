import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import Hls from "hls.js";

const API_BASE = "/api/v1";

interface StatusResponse {
  worker_running: boolean;
  camera_uri: string;
  hls_playlist: string;
}

interface UnknownFaceResponse {
  id: number;
  image_url: string;
  created_at: string;
  confidence: number;
}

interface PersonResponse {
  id: number;
  label: string;
  created_at: string;
  last_seen: string | null;
  total_detections: number;
}

interface DetectionEventResponse {
  id: number;
  label: string;
  confidence: number;
  is_known: boolean;
  created_at: string;
  image_url?: string | null;
}

interface ActivityActionResponse {
  id: number;
  activity_id: number;
  action_type: string;
  confidence: number;
  start_offset: number | null;
  end_offset: number | null;
  created_at: string;
  updated_at: string;
}

interface PersonActivityResponse {
  id: number;
  label: string;
  visit_id: number | null;
  activity: string | null;
  confidence: number;
  needs_review: boolean;
  clip_url: string | null;
  window_start: string;
  window_end: string;
  created_at: string;
  updated_at: string;
  actions: ActivityActionResponse[];
}

const COMMON_ACTIONS = ["cooking", "doing dishes", "idle", "other"];
const DETECTION_WINDOW_SECONDS = 10;

type ActivitySocketPayload = PersonActivityResponse & { type: "activity" };

function isActivityPayload(payload: unknown): payload is ActivitySocketPayload {
  return typeof payload === "object" && payload !== null && (payload as ActivitySocketPayload).type === "activity";
}

function getClipMimeType(clipUrl?: string | null): string {
  if (!clipUrl) return "video/mp4";
  const lower = clipUrl.toLowerCase();
  if (lower.endsWith(".webm")) return "video/webm";
  if (lower.endsWith(".mkv")) return "video/x-matroska";
  if (lower.endsWith(".mov")) return "video/quicktime";
  return "video/mp4";
}

function useHls(src: string | null) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;

    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = src;
      return;
    }

    if (Hls.isSupported()) {
      const hls = new Hls();
      hls.loadSource(src);
      hls.attachMedia(video);
      return () => hls.destroy();
    }
  }, [src]);

  return videoRef;
}

function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [unknownFaces, setUnknownFaces] = useState<UnknownFaceResponse[]>([]);
  const [people, setPeople] = useState<PersonResponse[]>([]);
  const [events, setEvents] = useState<DetectionEventResponse[]>([]);
  const [pendingActivities, setPendingActivities] = useState<PersonActivityResponse[]>([]);
  const [activityTimeline, setActivityTimeline] = useState<PersonActivityResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [timezone, setTimezone] = useState<string>(() => {
    try {
      return localStorage.getItem("user.timezone") || "America/Chicago";
    } catch {
      return "America/Chicago";
    }
  });
  const [mjpegAvailable, setMjpegAvailable] = useState(true);
  const [snapshotTick, setSnapshotTick] = useState(0);
  const [snapshotReady, setSnapshotReady] = useState(false);
  const [activeClip, setActiveClip] = useState<PersonActivityResponse | null>(null);

  const playlistUrl = useMemo(() => status?.hls_playlist ?? null, [status]);
  const videoRef = useHls(playlistUrl);
  const fallbackStreamUrl = `${API_BASE}/stream/mjpeg`;
  const snapshotUrl = useMemo(() => `${API_BASE}/stream/frame.jpg?ts=${snapshotTick}`, [snapshotTick]);

  const refreshAll = async () => {
    setLoading(true);
    try {
      const [
        statusRes,
        unknownRes,
        peopleRes,
        eventsRes,
        activitiesRes,
        timelineRes
      ] = await Promise.all([
        axios.get<StatusResponse>(`${API_BASE}/status`),
        axios.get<UnknownFaceResponse[]>(`${API_BASE}/faces/unknown`),
        axios.get<PersonResponse[]>(`${API_BASE}/faces/people`),
        axios.get<DetectionEventResponse[]>(`${API_BASE}/detections`),
        axios.get<PersonActivityResponse[]>(`${API_BASE}/activities/unknown`),
        axios.get<PersonActivityResponse[]>(`${API_BASE}/activities`, { params: { limit: 20 } })
      ]);
      setStatus(statusRes.data);
      setUnknownFaces(unknownRes.data);
      setPeople(peopleRes.data);
      setEvents(eventsRes.data);
      setPendingActivities(activitiesRes.data);
      setActivityTimeline(timelineRes.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    try {
      localStorage.setItem("user.timezone", timezone);
    } catch {
      /* ignore */
    }
  }, [timezone]);

  function formatDateTime(iso: string | Date | null | undefined, opts?: Intl.DateTimeFormatOptions) {
    if (!iso) return "—";
    const d = typeof iso === "string" ? new Date(iso) : iso;
    try {
      return new Intl.DateTimeFormat(undefined, { timeZone: timezone, ...(opts || {}) }).format(d);
    } catch (err) {
      return d.toLocaleString();
    }
  }

  function formatTime(iso: string | Date | null | undefined) {
    return formatDateTime(iso, { hour: "numeric", minute: "numeric", second: "numeric" });
  }

  function formatDetectionWindow(iso: string | Date) {
    const target = typeof iso === "string" ? new Date(iso) : iso;
    const halfWindowMs = (DETECTION_WINDOW_SECONDS * 1000) / 2;
    const start = new Date(target.getTime() - halfWindowMs);
    const end = new Date(target.getTime() + halfWindowMs);
    return `${formatTime(start)} – ${formatTime(end)}`;
  }

  function summarizeActions(activity: PersonActivityResponse) {
    if (!activity.actions || activity.actions.length === 0) {
      return activity.activity ?? "unknown";
    }
    return activity.actions.map((action) => action.action_type).join(", ");
  }

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}${API_BASE}/detections/ws`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (isActivityPayload(data)) {
        const payload = { ...data, actions: data.actions ?? [] } as PersonActivityResponse;
        setActivityTimeline((prev) => [payload, ...prev].slice(0, 20));
        if (payload.needs_review) {
          setPendingActivities((prev) => {
            const exists = prev.find((a) => a.id === payload.id);
            if (exists) {
              return prev.map((a) => (a.id === payload.id ? payload : a));
            }
            return [payload, ...prev];
          });
        } else {
          setPendingActivities((prev) => prev.filter((a) => a.id !== payload.id));
        }
        return;
      }

      const payload = data as DetectionEventResponse;
      setEvents((prev) => {
        const existingIndex = prev.findIndex((evt) => evt.id === payload.id);
        if (existingIndex !== -1) {
          const updated = [...prev];
          updated[existingIndex] = payload;
          return updated;
        }
        return [payload, ...prev].slice(0, 50);
      });
      if (!payload.is_known) {
        refreshUnknownFaces();
      }
    };
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (playlistUrl || mjpegAvailable) {
      return;
    }
    const id = setInterval(() => setSnapshotTick((prev) => prev + 1), 1000);
    return () => clearInterval(id);
  }, [playlistUrl, mjpegAvailable]);

  useEffect(() => {
    if (mjpegAvailable) {
      setSnapshotReady(false);
    }
  }, [mjpegAvailable]);

  useEffect(() => {
    if (mjpegAvailable || playlistUrl) {
      return;
    }
    const retry = setTimeout(() => setMjpegAvailable(true), 5000);
    return () => clearTimeout(retry);
  }, [mjpegAvailable, playlistUrl]);

  const refreshUnknownFaces = async () => {
    const res = await axios.get<UnknownFaceResponse[]>(`${API_BASE}/faces/unknown`);
    setUnknownFaces(res.data);
  };

  const refreshActivities = async () => {
    const [pendingRes, timelineRes] = await Promise.all([
      axios.get<PersonActivityResponse[]>(`${API_BASE}/activities/unknown`),
      axios.get<PersonActivityResponse[]>(`${API_BASE}/activities`, { params: { limit: 20 } })
    ]);
    setPendingActivities(pendingRes.data);
    setActivityTimeline(timelineRes.data);
  };

  const refreshPeople = async () => {
    const res = await axios.get<PersonResponse[]>(`${API_BASE}/faces/people`);
    setPeople(res.data);
  };

  const refreshEvents = async () => {
    const res = await axios.get<DetectionEventResponse[]>(`${API_BASE}/detections`);
    setEvents(res.data);
  };

  const labelUnknown = async (faceId: number) => {
    const label = window.prompt("Enter label for this person:");
    if (!label) return;
    await axios.post(`${API_BASE}/faces/unknown/${faceId}/label`, { label });
    await Promise.all([refreshUnknownFaces(), refreshPeople(), refreshEvents()]);
  };

  const deleteUnknown = async (faceId: number) => {
    await axios.delete(`${API_BASE}/faces/unknown/${faceId}`);
    await refreshUnknownFaces();
  };

  const updateActivity = async (
    activityId: number,
    update: Partial<Pick<PersonActivityResponse, "activity" | "confidence" | "needs_review">>
  ) => {
    await axios.patch<PersonActivityResponse>(`${API_BASE}/activities/${activityId}`, update);
    await refreshActivities();
  };

  const createAction = async (activityId: number, actionType: string, confidence = 0.99) => {
    await axios.post(`${API_BASE}/actions`, {
      activity_id: activityId,
      action_type: actionType,
      confidence,
    });
    await refreshActivities();
  };

  const deleteAction = async (actionId: number) => {
    await axios.delete(`${API_BASE}/actions/${actionId}`);
    await refreshActivities();
  };

  const handleActionLabel = async (activityId: number, suggestion?: string) => {
    let value = suggestion ?? "";
    if (!value) {
      const input = window.prompt("Enter action label");
      if (!input) {
        return;
      }
      value = input;
    }
    await createAction(activityId, value);
  };

  const handleActionDelete = async (actionId: number) => {
    if (!window.confirm("Remove this action?")) {
      return;
    }
    await deleteAction(actionId);
  };

  const markActivityUnknown = async (activityId: number) => {
    await updateActivity(activityId, { activity: "unknown", confidence: 0.0, needs_review: true });
  };

  const markActivityReviewed = async (activityId: number) => {
    await updateActivity(activityId, { needs_review: false });
  };

  const openClip = (activity: PersonActivityResponse) => {
    if (!activity.clip_url) return;
    setActiveClip(activity);
  };

  const closeClip = () => setActiveClip(null);

  return (
    <div className="app">
      <header>
        <div>
          <h1>Jetson Nano Monitor</h1>
          <p>{status?.worker_running ? "Worker running" : "Worker stopped"}</p>
          <div style={{ marginTop: 6 }}>
            <label style={{ marginRight: 8 }}>
              Timezone:
            </label>
            <select value={timezone} onChange={(e) => setTimezone(e.target.value)}>
              <option value="America/Chicago">America/Chicago (Central)</option>
              <option value="UTC">UTC</option>
              <option value="America/New_York">America/New_York (Eastern)</option>
              <option value="Europe/London">Europe/London</option>
              <option value="Asia/Kolkata">Asia/Kolkata</option>
            </select>
          </div>
        </div>
        <button onClick={refreshAll} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </header>

      <section className="grid">
        <div className="card">
          <h2>Live Stream</h2>
          {playlistUrl ? (
            <video ref={videoRef} controls autoPlay muted playsInline />
          ) : mjpegAvailable ? (
            <img
              src={fallbackStreamUrl}
              alt="Live stream"
              className="live-frame"
              onError={() => setMjpegAvailable(false)}
            />
          ) : (
            <div className="live-fallback">
              {!snapshotReady && <p>Waiting for live frames…</p>}
              <img
                src={snapshotUrl}
                alt="Latest frame"
                className="live-frame"
                onLoad={() => setSnapshotReady(true)}
                onError={() => setSnapshotReady(false)}
              />
            </div>
          )}
          {!playlistUrl && <small>Fallback stream served directly from the detection worker.</small>}
        </div>

        <div className="card">
          <h2>Recent Detections</h2>
          <ul className="event-list">
            {events.map((event) => (
              <li key={event.id}>
                <strong>{event.label}</strong>
                <span>{formatTime(event.created_at)}</span>
                <small>{(event.confidence * 100).toFixed(1)}%</small>
                <small>Window: {formatDetectionWindow(event.created_at)}</small>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="grid">
        <div className="card">
          <h2>Activity Review</h2>
          {pendingActivities.length === 0 ? (
            <p>No activities need review.</p>
          ) : (
            <div className="activity-grid">
              {pendingActivities.map((activity) => (
                <article key={activity.id}>
                  {activity.clip_url ? (
                    <div className="activity-media">
                      <video
                        controls
                        preload="metadata"
                        playsInline
                        onClick={() => openClip(activity)}
                      >
                        <source src={activity.clip_url} type={getClipMimeType(activity.clip_url)} />
                        Your browser does not support MP4 playback.
                      </video>
                      <div className="activity-media-links">
                        <button type="button" onClick={() => openClip(activity)}>
                          Play inline
                        </button>
                        <a href={activity.clip_url} target="_blank" rel="noreferrer">
                          Open tab
                        </a>
                        <a href={activity.clip_url} download>
                          Download
                        </a>
                      </div>
                    </div>
                  ) : (
                    <div className="activity-placeholder">No clip available</div>
                  )}
                  <header>
                    <strong>{activity.label}</strong>
                    <span>
                      {formatTime(activity.window_start)} – {" "}
                      {formatTime(activity.window_end)}
                    </span>
                  </header>
                  <div className="action-list">
                    {activity.actions.length === 0 ? (
                      <p className="action-list__empty">No actions recorded yet.</p>
                    ) : (
                      activity.actions.map((action) => (
                        <div key={action.id} className="action-list__item">
                          <div>
                            <strong>{action.action_type}</strong>
                            <small>{(action.confidence * 100).toFixed(1)}%</small>
                          </div>
                          <button type="button" onClick={() => handleActionDelete(action.id)}>
                            Remove
                          </button>
                        </div>
                      ))
                    )}
                  </div>
                  <div className="activity-actions">
                    {COMMON_ACTIONS.map((entry) => (
                      <button key={entry} onClick={() => handleActionLabel(activity.id, entry)}>
                        {entry}
                      </button>
                    ))}
                    <button onClick={() => handleActionLabel(activity.id)}>Custom action</button>
                    <button onClick={() => markActivityReviewed(activity.id)}>Mark reviewed</button>
                    <button onClick={() => markActivityUnknown(activity.id)}>Flag unknown</button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>

        <div className="card">
          <h2>Unknown Faces</h2>
          <div className="unknown-grid">
            {unknownFaces.map((face) => (
              <article key={face.id}>
                <img src={face.image_url} alt="Unknown face" />
                <footer>
                  <span>{formatDateTime(face.created_at)}</span>
                  <div className="actions">
                    <button onClick={() => labelUnknown(face.id)}>Label</button>
                    <button onClick={() => deleteUnknown(face.id)}>Dismiss</button>
                  </div>
                </footer>
              </article>
            ))}
            {unknownFaces.length === 0 && <p>No pending unknown faces.</p>}
          </div>
        </div>

        <div className="card">
          <h2>People Directory</h2>
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Total</th>
                <th>Last seen</th>
              </tr>
            </thead>
            <tbody>
              {people.map((person) => (
                <tr key={person.id}>
                  <td>{person.label}</td>
                  <td>{person.total_detections}</td>
                  <td>{person.last_seen ? formatDateTime(person.last_seen) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>Recent Activities</h2>
          <table>
            <thead>
              <tr>
                <th>Person</th>
                <th>Activity</th>
                <th>Confidence</th>
                <th>Window</th>
              </tr>
            </thead>
            <tbody>
              {activityTimeline.map((activity) => (
                <tr key={activity.id}>
                  <td>{activity.label}</td>
                  <td>{summarizeActions(activity)}</td>
                  <td>{(activity.confidence * 100).toFixed(1)}%</td>
                  <td>
                    {formatTime(activity.window_start)} – {" "}
                    {formatTime(activity.window_end)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {activeClip && (
        <div className="clip-overlay" onClick={closeClip}>
          <div className="clip-overlay__dialog" onClick={(event) => event.stopPropagation()}>
            <header>
              <div>
                <strong>{activeClip.label}</strong>
                <p>
                  {formatTime(activeClip.window_start)} – {" "}
                  {formatTime(activeClip.window_end)}
                </p>
              </div>
              <button type="button" onClick={closeClip}>
                Close
              </button>
            </header>
            <video controls autoPlay playsInline>
              <source src={activeClip.clip_url ?? undefined} type={getClipMimeType(activeClip.clip_url)} />
              Your browser does not support MP4 playback.
            </video>
          </div>
        </div>
      )}
    </div>
  );
}
export default App;
