"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type AgentMode = "idle" | "running" | "stopped" | "finished" | "error";
type ActionType = "collect_data" | "start_training" | "get_training_status" | "finish";

type AgentState = {
  data_count: number;
  target_data: number;
  loss: number;
  target_loss: number;
  iteration: number;
  max_iteration: number;
};

type LogEntry = {
  step: number;
  actor: string;
  message: string;
  action?: ActionType | null;
};

type PipelineStatus = {
  state: AgentState;
  mode: AgentMode;
  done: boolean;
  current_action?: ActionType | null;
  logs: LogEntry[];
  last_tool_result: Record<string, unknown>;
  session_name: string;
  error?: string | null;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8001";
const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");

const defaultState: AgentState = {
  data_count: 0,
  target_data: 100,
  loss: 1.0,
  target_loss: 0.1,
  iteration: 0,
  max_iteration: 10,
};

const emptyStatus: PipelineStatus = {
  state: defaultState,
  mode: "idle",
  done: false,
  current_action: null,
  logs: [],
  last_tool_result: {},
  session_name: "robot-training-session",
  error: null,
};

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers ?? {}),
    },
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data?.detail ?? "Request failed");
  }
  return data as T;
}

export default function HomePage() {
  const [status, setStatus] = useState<PipelineStatus>(emptyStatus);
  const [sessionName, setSessionName] = useState("robot-training-session");
  const [agentState, setAgentState] = useState<AgentState>(defaultState);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [socketState, setSocketState] = useState<"connecting" | "open" | "closed">("connecting");
  const logRef = useRef<HTMLDivElement | null>(null);

  const isRunning = status.mode === "running";
  const summary = useMemo(
    () => [
      { label: "Data", value: `${status.state.data_count} / ${status.state.target_data}` },
      { label: "Loss", value: `${status.state.loss.toFixed(4)} / ${status.state.target_loss}` },
      { label: "Iteration", value: `${status.state.iteration} / ${status.state.max_iteration}` },
      { label: "Mode", value: status.mode },
    ],
    [status],
  );

  useEffect(() => {
    void refreshStatus();
  }, []);

  useEffect(() => {
    const socket = new WebSocket(`${WS_BASE_URL}/agent/ws`);
    setSocketState("connecting");

    socket.onopen = () => {
      setSocketState("open");
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as { type: string; status: PipelineStatus };
        if (payload.type === "status") {
          setStatus(payload.status);
          setSessionName(payload.status.session_name);
          setAgentState(payload.status.state);
          setError(payload.status.error ?? null);
        }
      } catch {
        setError("Unable to parse WebSocket payload");
      }
    };

    socket.onerror = () => {
      setSocketState("closed");
    };

    socket.onclose = () => {
      setSocketState("closed");
    };

    return () => socket.close();
  }, []);

  useEffect(() => {
    if (socketState === "open" || !isRunning) return;
    const timer = window.setInterval(() => {
      void refreshStatus();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [isRunning, socketState]);

  useEffect(() => {
    if (!logRef.current) return;
    logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [status.logs]);

  async function refreshStatus() {
    try {
      const data = await fetchJSON<{ status: PipelineStatus }>(`${API_BASE_URL}/agent/status`);
      setStatus(data.status);
      setSessionName(data.status.session_name);
      setAgentState(data.status.state);
      setError(data.status.error ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to load status");
    }
  }

  async function startAgent() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchJSON<{ status: PipelineStatus }>(`${API_BASE_URL}/agent/start`, {
        method: "POST",
        body: JSON.stringify({
          session_name: sessionName,
          state: agentState,
          training_config: {
            epochs: 3,
            batch_size: 16,
            learning_rate: 0.0005,
          },
        }),
      });
      setStatus(data.status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start agent");
    } finally {
      setIsLoading(false);
    }
  }

  async function stopAgent() {
    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchJSON<{ status: PipelineStatus }>(`${API_BASE_URL}/agent/stop`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      setStatus(data.status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to stop agent");
    } finally {
      setIsLoading(false);
    }
  }

  function updateStateField<K extends keyof AgentState>(key: K, value: AgentState[K]) {
    setAgentState((current) => ({ ...current, [key]: value }));
  }

  return (
    <main className="page-shell">
      <section className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">Robot Training Control Center</p>
          <h1>AI agent that decides when to collect data and when to retrain</h1>
          <p className="lead">
            Start the loop, watch the reasoning, and inspect each tool execution in real time.
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary-button" onClick={startAgent} disabled={isLoading || isRunning}>
            Start Agent
          </button>
          <button className="secondary-button" onClick={stopAgent} disabled={isLoading || !isRunning}>
            Stop Agent
          </button>
        </div>
      </section>

      <section className="dashboard-grid">
        <article className="panel control-panel">
          <div className="panel-head">
            <h2>Session Setup</h2>
            <span className={`status-pill status-${status.mode}`}>{status.mode}</span>
          </div>
          <label className="field">
            <span>Session name</span>
            <input value={sessionName} onChange={(e) => setSessionName(e.target.value)} />
          </label>

          <div className="field-grid">
            <label className="field">
              <span>Current data</span>
              <input
                type="number"
                value={agentState.data_count}
                onChange={(e) => updateStateField("data_count", Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>Target data</span>
              <input
                type="number"
                value={agentState.target_data}
                onChange={(e) => updateStateField("target_data", Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>Current loss</span>
              <input
                type="number"
                step="0.01"
                value={agentState.loss}
                onChange={(e) => updateStateField("loss", Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>Target loss</span>
              <input
                type="number"
                step="0.01"
                value={agentState.target_loss}
                onChange={(e) => updateStateField("target_loss", Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>Iteration</span>
              <input
                type="number"
                value={agentState.iteration}
                onChange={(e) => updateStateField("iteration", Number(e.target.value))}
              />
            </label>
            <label className="field">
              <span>Max iteration</span>
              <input
                type="number"
                value={agentState.max_iteration}
                onChange={(e) => updateStateField("max_iteration", Number(e.target.value))}
              />
            </label>
          </div>
        </article>

        <article className="panel summary-panel">
          <div className="panel-head">
            <h2>Pipeline Status</h2>
            <div className="panel-head-actions">
              <span className={`status-pill socket-${socketState}`}>ws {socketState}</span>
              <button className="ghost-button" onClick={() => void refreshStatus()}>
                Refresh
              </button>
            </div>
          </div>
          <div className="summary-grid">
            {summary.map((item) => (
              <div key={item.label} className="metric-card">
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>
          <div className="tool-result">
            <p className="tool-label">Last tool result</p>
            <pre>{JSON.stringify(status.last_tool_result, null, 2)}</pre>
          </div>
          {error ? <p className="error-box">{error}</p> : null}
        </article>
      </section>

      <section className="panel log-panel">
        <div className="panel-head">
          <h2>Agent Log</h2>
          <span className="log-hint">Chat-style reasoning and tool execution</span>
        </div>
        <div className="log-stream" ref={logRef}>
          {status.logs.length === 0 ? (
            <div className="empty-log">No events yet. Start the agent to see live decisions.</div>
          ) : (
            status.logs.map((entry) => (
              <article key={entry.step} className={`log-bubble actor-${entry.actor}`}>
                <div className="log-meta">
                  <span>{entry.actor}</span>
                  <span>step {entry.step}</span>
                </div>
                <p>{entry.message}</p>
              </article>
            ))
          )}
        </div>
      </section>
    </main>
  );
}
