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
  const [loading, setLoading] = useState(true);
  const [mjpegAvailable, setMjpegAvailable] = useState(true);
  const [snapshotTick, setSnapshotTick] = useState(0);
  const [snapshotReady, setSnapshotReady] = useState(false);

  const playlistUrl = useMemo(() => status?.hls_playlist ?? null, [status]);
  const videoRef = useHls(playlistUrl);
  const fallbackStreamUrl = `${API_BASE}/stream/mjpeg`;
  const snapshotUrl = useMemo(() => `${API_BASE}/stream/frame.jpg?ts=${snapshotTick}`, [snapshotTick]);

  const refreshAll = async () => {
    setLoading(true);
    try {
      const [statusRes, unknownRes, peopleRes, eventsRes] = await Promise.all([
        axios.get<StatusResponse>(`${API_BASE}/status`),
        axios.get<UnknownFaceResponse[]>(`${API_BASE}/faces/unknown`),
        axios.get<PersonResponse[]>(`${API_BASE}/faces/people`),
        axios.get<DetectionEventResponse[]>(`${API_BASE}/detections`)
      ]);
      setStatus(statusRes.data);
      setUnknownFaces(unknownRes.data);
      setPeople(peopleRes.data);
      setEvents(eventsRes.data);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refreshAll();
  }, []);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${window.location.host}${API_BASE}/detections/ws`);
    ws.onmessage = (event) => {
      const payload: DetectionEventResponse = JSON.parse(event.data);
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

  return (
    <div className="app">
      <header>
        <div>
          <h1>Jetson Nano Monitor</h1>
          <p>{status?.worker_running ? "Worker running" : "Worker stopped"}</p>
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
                <span>{new Date(event.created_at).toLocaleTimeString()}</span>
                <small>{(event.confidence * 100).toFixed(1)}%</small>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="grid">
        <div className="card">
          <h2>Unknown Faces</h2>
          <div className="unknown-grid">
            {unknownFaces.map((face) => (
              <article key={face.id}>
                <img src={face.image_url} alt="Unknown face" />
                <footer>
                  <span>{new Date(face.created_at).toLocaleString()}</span>
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
                  <td>{person.last_seen ? new Date(person.last_seen).toLocaleString() : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default App;
