import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as speechCommands from "@tensorflow-models/speech-commands";
import * as poseDetection from "@tensorflow-models/pose-detection";

const STORAGE_KEY = "model-studio-project-v2";
const IMAGE_MODEL_KEY = "indexeddb://model-studio-image-classifier";
const POSE_MODEL_KEY = "indexeddb://model-studio-pose-classifier";
const BRIDGE_URL_KEY = "lerobot-bridge-url";
const AGENT_API_BASE_URL =
  import.meta.env.VITE_AGENT_API_BASE_URL || "https://robot-agent-backend.onrender.com";
const AGENT_WS_URL = AGENT_API_BASE_URL.replace(/^http/, "ws");

const projectModes = {
  image: {
    title: "LeRobot 카메라 학습",
    badge: "비전",
    subtitle: "로봇에 연결된 카메라 화면으로 태스크별 에피소드를 기록합니다.",
    description:
      "로봇 카메라 피드를 연결하고 태스크별 episode를 기록한 뒤, 브라우저에서 바로 학습하고 예측 결과를 테스트합니다.",
    helper: "로봇 카메라를 켠 뒤 태스크별로 몇 초 동안 몇 회 기록할지 정하고 에피소드를 모아보세요.",
    previewLabel: "로봇 카메라 피드",
    inputAction: "카메라 연결",
    trainingBadge: "LeRobot 비전 학습",
  },
  audio: {
    title: "LeRobot 학습 세션",
    badge: "세션",
    subtitle: "수집, 학습, 기록을 하나의 실험 세션으로 관리합니다.",
    description:
      "로봇 실험 세션을 기록하고 나중에 같은 설정으로 다시 테스트할 수 있는 관리 영역입니다.",
    helper: "실험 세션 이름과 로봇 연결 상태를 정리한 뒤 학습 파이프라인을 시작하세요.",
    previewLabel: "세션 개요",
    inputAction: "세션 준비",
    trainingBadge: "실험 세션 기록",
  },
  pose: {
    title: "LeRobot 테스트",
    badge: "테스트",
    subtitle: "학습 결과와 서보 반응을 실시간으로 검증합니다.",
    description:
      "서보 상태를 확인하고 테스트 동작을 보내면서 학습 결과가 실제 로봇 행동과 맞는지 점검합니다.",
    helper: "로봇 상태를 읽은 뒤 테스트 동작을 보내고, 카메라 예측과 함께 비교해보세요.",
    previewLabel: "테스트 미리보기",
    inputAction: "테스트 준비",
    trainingBadge: "실시간 테스트",
  },
};

const classTones = ["green", "purple", "orange"];
const minimumClassCount = 0;
const defaultClassNames = {
  image: [],
  audio: [],
  pose: [],
};

const defaultTaskConfigs = {
  image: [],
  audio: [],
  pose: [],
};

const modeIds = ["image", "audio", "pose"];

const trainingSettings = {
  image: { epochs: 12, batchSize: 8, denseUnits: 48, dropout: 0.12, size: 48 },
  pose: { epochs: 24, batchSize: 8, denseUnits: 96, dropout: 0.15 },
  minSamples: 9,
  captureIntervalMs: 180,
  previewIntervalMs: 240,
};

const defaultAgentState = {
  data_count: 0,
  target_data: 100,
  loss: 1.0,
  target_loss: 0.1,
  iteration: 0,
  max_iteration: 10,
};

const emptyAgentStatus = {
  state: defaultAgentState,
  mode: "idle",
  done: false,
  current_action: null,
  logs: [],
  last_tool_result: {},
  session_name: "default-session",
  error: null,
};

const agentModeLabels = {
  idle: "대기",
  running: "실행 중",
  stopped: "중지됨",
  finished: "완료",
  error: "오류",
};

const agentActorLabels = {
  system: "시스템",
  agent: "에이전트",
  tool: "도구",
  error: "오류",
};

const gpuJobStateLabels = {
  queued: "대기 중",
  running: "학습 중",
  completed: "완료",
  error: "오류",
};

export default function App() {
  const [mode, setMode] = useState("image");
  const [studioView, setStudioView] = useState("simple");
  const [isStudioOpen, setIsStudioOpen] = useState(false);
  const [projectName, setProjectName] = useState("AI Robot Studio 데모");
  const [classNamesByMode, setClassNamesByMode] = useState(defaultClassNames);
  const [sampleCountsByMode, setSampleCountsByMode] = useState({
    image: [],
    audio: [],
    pose: [],
  });
  const [predictionsByMode, setPredictionsByMode] = useState({
    image: [],
    audio: [],
    pose: [],
  });
  const [statusByMode, setStatusByMode] = useState({
    image: "LeRobot 카메라를 연결하면 학습을 시작할 수 있습니다.",
    audio: "세션 관리 기능을 준비하면 시작할 수 있습니다.",
    pose: "테스트 기능을 준비하면 시작할 수 있습니다.",
  });
  const [progressByMode, setProgressByMode] = useState({
    image: 0,
    audio: 0,
    pose: 0,
  });
  const [trainedByMode, setTrainedByMode] = useState({
    image: false,
    audio: false,
    pose: false,
  });
  const [readyByMode, setReadyByMode] = useState({
    image: false,
    audio: false,
    pose: false,
  });
  const [isTraining, setIsTraining] = useState(false);
  const [isPreviewRunning, setIsPreviewRunning] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isImageCameraOn, setIsImageCameraOn] = useState(false);
  const [isPoseCameraOn, setIsPoseCameraOn] = useState(false);
  const [saveMessage, setSaveMessage] = useState("아직 저장된 프로젝트가 없습니다.");
  const [pendingUploadClassIndex, setPendingUploadClassIndex] = useState(null);
  const [bridgeUrl, setBridgeUrl] = useState(() => {
    const saved = window.localStorage.getItem(BRIDGE_URL_KEY);
    if (saved) {
      return saved;
    }
    return "http://127.0.0.1:5057";
  });
  const [serialPorts, setSerialPorts] = useState([]);
  const [detectedServos, setDetectedServos] = useState([]);
  const [selectedPort, setSelectedPort] = useState("");
  const [selectedServoId, setSelectedServoId] = useState("");
  const [baudRate, setBaudRate] = useState(1000000);
  const [timeoutMs, setTimeoutMs] = useState(50);
  const [robotConnection, setRobotConnection] = useState(null);
  const [robotMetrics, setRobotMetrics] = useState(null);
  const [robotHistory, setRobotHistory] = useState([]);
  const [taskConfigsByMode, setTaskConfigsByMode] = useState(defaultTaskConfigs);
  const [clipLibraryByMode, setClipLibraryByMode] = useState({
    image: [],
    audio: [],
    pose: [],
  });
  const [robotGoal, setRobotGoal] = useState(2048);
  const [robotSpeed, setRobotSpeed] = useState(300);
  const [robotAcceleration, setRobotAcceleration] = useState(10);
  const [controlMode, setControlMode] = useState("write");
  const [torqueEnabled, setTorqueEnabled] = useState(true);
  const [autoStart, setAutoStart] = useState(0);
  const [autoEnd, setAutoEnd] = useState(4095);
  const [autoSweepDelay, setAutoSweepDelay] = useState(2500);
  const [autoStep, setAutoStep] = useState(10);
  const [autoStepDelay, setAutoStepDelay] = useState(10);
  const [isSweepRunning, setIsSweepRunning] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(30);
  const [recordFilename, setRecordFilename] = useState("lerobot-record.csv");
  const [recordedFrames, setRecordedFrames] = useState([]);
  const [servoToolState, setServoToolState] = useState({
    bridgeOnline: false,
    toolRootExists: false,
    appBundleExists: false,
    pythonEntryExists: false,
    isRunning: false,
    message: "로컬 로봇 제어 브리지를 확인하는 중입니다.",
    toolRoot: "/Users/almond/feetech-servo-tool",
  });
  const [agentSessionName, setAgentSessionName] = useState("default-session");
  const [agentState, setAgentState] = useState(defaultAgentState);
  const [agentStatus, setAgentStatus] = useState(emptyAgentStatus);
  const [agentError, setAgentError] = useState("");
  const [agentBusy, setAgentBusy] = useState(false);
  const [agentSocketState, setAgentSocketState] = useState("connecting");
  const [gpuJobs, setGpuJobs] = useState([]);
  const [gpuBusy, setGpuBusy] = useState(false);
  const [gpuError, setGpuError] = useState("");

  const imageVideoRef = useRef(null);
  const poseVideoRef = useRef(null);
  const poseCanvasRef = useRef(null);
  const uploadInputRef = useRef(null);
  const imageStreamRef = useRef(null);
  const poseStreamRef = useRef(null);
  const imageCaptureTimerRef = useRef(null);
  const poseCaptureTimerRef = useRef(null);
  const imageHoldTimeoutRef = useRef(null);
  const poseHoldTimeoutRef = useRef(null);
  const imagePreviewTimerRef = useRef(null);
  const poseLoopFrameRef = useRef(null);
  const posePredictionAtRef = useRef(0);
  const currentPoseVectorRef = useRef(null);
  const sweepTimerRef = useRef(null);
  const recordTimerRef = useRef(null);

  const imageClassifierRef = useRef(null);
  const imageSamplesRef = useRef([]);

  const audioBaseRecognizerRef = useRef(null);
  const audioTransferRecognizerRef = useRef(null);

  const poseDetectorRef = useRef(null);
  const poseClassifierRef = useRef(null);
  const poseSamplesRef = useRef([]);

  const currentClassNames = classNamesByMode[mode];
  const currentSampleCounts = sampleCountsByMode[mode];
  const currentPredictions = predictionsByMode[mode];
  const currentStatus = statusByMode[mode];
  const currentProgress = progressByMode[mode];
  const isCurrentModeReady = readyByMode[mode];
  const isCurrentModeTrained = trainedByMode[mode];
  const totalSamples = useMemo(
    () => currentSampleCounts.reduce((sum, count) => sum + count, 0),
    [currentSampleCounts],
  );
  const readinessText =
    currentClassNames.length < 2
      ? "태스크를 2개 이상 추가하세요"
      : totalSamples >= trainingSettings.minSamples
      ? "학습 가능"
      : `${trainingSettings.minSamples - totalSamples}개 샘플 더 필요`;
  const isAgentRunning = agentStatus.mode === "running";
  const latestGpuJob = gpuJobs[0] || null;
  const agentSummary = [
    { label: "데이터", value: `${agentStatus.state.data_count} / ${agentStatus.state.target_data}` },
    { label: "손실값", value: `${agentStatus.state.loss.toFixed(4)} / ${agentStatus.state.target_loss}` },
    { label: "반복", value: `${agentStatus.state.iteration} / ${agentStatus.state.max_iteration}` },
    { label: "상태", value: agentModeLabels[agentStatus.mode] || agentStatus.mode },
  ];
  const currentClipCount = clipLibraryByMode[mode].length;
  const currentTaskCount = currentClassNames.length;

  useEffect(() => {
    return () => {
      stopAllMediaAndLoops();
      stopSweep();
      stopRecording();
      clearImageSamples();
      clearPoseSamples();
      imageClassifierRef.current?.dispose();
      poseClassifierRef.current?.dispose();
      poseDetectorRef.current?.dispose?.();
    };
  }, []);

  useEffect(() => {
    const socket = new window.WebSocket(`${AGENT_WS_URL}/agent/ws`);
    setAgentSocketState("connecting");

    socket.onopen = () => setAgentSocketState("open");
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === "status") {
          setAgentStatus(payload.status);
          setAgentSessionName(payload.status.session_name);
          setAgentState(payload.status.state);
          setAgentError(payload.status.error || "");
        }
      } catch (error) {
        setAgentError("AI 에이전트 상태를 읽는 중 문제가 생겼습니다.");
      }
    };
    socket.onerror = () => setAgentSocketState("closed");
    socket.onclose = () => setAgentSocketState("closed");

    return () => socket.close();
  }, []);

  useEffect(() => {
    void refreshGpuJobs();

    const interval = window.setInterval(() => {
      void refreshGpuJobs();
    }, 5000);

    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (agentSocketState === "open") {
      return undefined;
    }

    const interval = window.setInterval(() => {
      void refreshAgentStatus();
    }, 4000);

    return () => window.clearInterval(interval);
  }, [agentSocketState]);

  useEffect(() => {
    stopPreview();
    stopImageCaptureLoop();
    stopPoseCaptureLoop();

    if (mode !== "image") {
      stopImageCamera();
    }

    if (mode !== "pose") {
      stopPoseCamera();
    }
  }, [mode]);

  async function fetchAgentJson(path, options = {}) {
    const response = await fetch(`${AGENT_API_BASE_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail || "AI 에이전트 요청에 실패했습니다.");
    }
    return data;
  }

  async function refreshAgentStatus() {
    try {
      const data = await fetchAgentJson("/agent/status");
      setAgentStatus(data.status);
      setAgentSessionName(data.status.session_name);
      setAgentState(data.status.state);
      setAgentError(data.status.error || "");
    } catch (error) {
      setAgentError(error.message);
      setAgentSocketState("closed");
    }
  }

  async function refreshGpuJobs() {
    try {
      const data = await fetchAgentJson("/gpu/jobs");
      setGpuJobs(data.jobs || []);
      setGpuError("");
    } catch (error) {
      setGpuError(error.message || "GPU 작업 상태를 읽지 못했습니다.");
    }
  }

  function updateAgentState(key, value) {
    setAgentState((current) => ({ ...current, [key]: value }));
  }

  function syncAgentFromProject() {
    setAgentSessionName(projectName.trim() || "lerobot-session");
    setAgentState((current) => ({
      ...current,
      data_count: totalSamples,
      target_data: Math.max(current.target_data, totalSamples || 100),
    }));
    setAgentError("");
  }

  async function startAgentLoop() {
    setAgentBusy(true);
    setAgentError("");
    try {
      const data = await fetchAgentJson("/agent/start", {
        method: "POST",
        body: JSON.stringify({
          session_name: agentSessionName,
          state: agentState,
          training_config: {
            epochs: 3,
            batch_size: 16,
            learning_rate: 0.0005,
          },
        }),
      });
      setAgentStatus(data.status);
    } catch (error) {
      setAgentError(error.message || "AI 에이전트 요청 중 문제가 생겼습니다.");
    } finally {
      setAgentBusy(false);
    }
  }

  async function stopAgentLoop() {
    setAgentBusy(true);
    setAgentError("");
    try {
      const data = await fetchAgentJson("/agent/stop", {
        method: "POST",
        body: JSON.stringify({}),
      });
      setAgentStatus(data.status);
    } catch (error) {
      setAgentError(error.message || "AI 에이전트 요청 중 문제가 생겼습니다.");
    } finally {
      setAgentBusy(false);
    }
  }

  function buildDatasetSummary() {
    return {
      mode,
      project_name: projectName,
      task_names: currentClassNames,
      sample_counts: currentSampleCounts,
      clip_count: currentClipCount,
      has_raw_episodes: currentClipCount > 0,
      notes:
        currentClipCount > 0
          ? "현재는 브라우저에 저장된 클립 메타데이터 기준입니다. 이후 원본 episode 업로드 파이프라인을 연결할 수 있습니다."
          : "아직 수집된 클립이 없습니다.",
    };
  }

  async function submitGpuTrainingJob() {
    setGpuBusy(true);
    setGpuError("");
    try {
      const data = await fetchAgentJson("/gpu/train", {
        method: "POST",
        body: JSON.stringify({
          session_name: projectName.trim() || "lerobot-gpu-session",
          dataset: buildDatasetSummary(),
          training_config: {
            provider: "vessl",
            project: projectName.trim() || "lerobot-gpu-session",
            epochs: 10,
          },
        }),
      });
      setGpuJobs((current) => [data.job, ...current.filter((item) => item.job_id !== data.job.job_id)]);
    } catch (error) {
      setGpuError(error.message || "GPU 학습 요청에 실패했습니다.");
    } finally {
      setGpuBusy(false);
    }
  }

  useEffect(() => {
    if (mode !== "image" || !isPreviewRunning || !trainedByMode.image) {
      if (imagePreviewTimerRef.current) {
        window.clearInterval(imagePreviewTimerRef.current);
        imagePreviewTimerRef.current = null;
      }
      return undefined;
    }

    imagePreviewTimerRef.current = window.setInterval(() => {
      void runImagePrediction();
    }, trainingSettings.previewIntervalMs);

    return () => {
      if (imagePreviewTimerRef.current) {
        window.clearInterval(imagePreviewTimerRef.current);
        imagePreviewTimerRef.current = null;
      }
    };
  }, [isPreviewRunning, mode, trainedByMode.image]);

  useEffect(() => {
    void refreshServoToolStatus();

    const interval = window.setInterval(() => {
      void refreshServoToolStatus({ silent: true });
    }, 5000);

    return () => {
      window.clearInterval(interval);
    };
  }, [bridgeUrl]);

  useEffect(() => {
    window.localStorage.setItem(BRIDGE_URL_KEY, bridgeUrl);
  }, [bridgeUrl]);

  useEffect(() => {
    if (!servoToolState.bridgeOnline) {
      return undefined;
    }

    void loadRobotPorts();
    void refreshRobotConnection();

    const interval = window.setInterval(() => {
      void refreshRobotConnection({ silent: true });
      void refreshRobotMetrics({ silent: true });
    }, 3000);

    return () => {
      window.clearInterval(interval);
    };
  }, [bridgeUrl, servoToolState.bridgeOnline, robotConnection]);

  useEffect(() => {
    if (!robotMetrics) {
      return;
    }

    setRobotHistory((current) => {
      const next = [
        ...current,
        {
          position: Number(robotMetrics.position ?? 0),
          load: Number(robotMetrics.load ?? 0),
          speed: Number(robotMetrics.speed ?? 0),
          current: Number(robotMetrics.current ?? 0),
          temperature: Number(robotMetrics.temperature ?? 0),
          voltage: Number(robotMetrics.voltage ?? 0),
        },
      ];
      return next.slice(-40);
    });
  }, [robotMetrics]);

  async function loadSavedProject() {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return;
    }

    try {
      const saved = JSON.parse(raw);
      if (saved.projectName) {
        setProjectName(saved.projectName);
      }
      if (saved.classNamesByMode) {
        setClassNamesByMode((current) => ({ ...current, ...saved.classNamesByMode }));
      }
      if (saved.sampleCountsByMode) {
        setSampleCountsByMode((current) => ({ ...current, ...saved.sampleCountsByMode }));
      }
      if (saved.taskConfigsByMode) {
        setTaskConfigsByMode((current) => ({ ...current, ...saved.taskConfigsByMode }));
      }
      if (saved.clipLibraryByMode) {
        setClipLibraryByMode((current) => ({ ...current, ...saved.clipLibraryByMode }));
      }
      setMode("image");

      try {
        const imageModel = await tf.loadLayersModel(IMAGE_MODEL_KEY);
        imageClassifierRef.current?.dispose();
        imageClassifierRef.current = imageModel;
        updateReady("image", true);
        updateTrained("image", true);
      } catch (error) {
        console.info("No saved image model found.", error);
      }

      try {
        const poseModel = await tf.loadLayersModel(POSE_MODEL_KEY);
        poseClassifierRef.current?.dispose();
        poseClassifierRef.current = poseModel;
        updateReady("pose", true);
        updateTrained("pose", true);
      } catch (error) {
        console.info("No saved pose model found.", error);
      }

      setSaveMessage("마지막으로 저장한 프로젝트를 불러왔습니다.");
    } catch (error) {
      console.error(error);
      setSaveMessage("저장된 프로젝트를 복원하지 못했습니다.");
    }
  }

  function setModeStatus(targetMode, message) {
    setStatusByMode((current) => ({ ...current, [targetMode]: message }));
  }

  function setModeProgress(targetMode, value) {
    setProgressByMode((current) => ({ ...current, [targetMode]: value }));
  }

  function setModePredictions(targetMode, values) {
    setPredictionsByMode((current) => ({ ...current, [targetMode]: values }));
  }

  function addClipRecord(targetMode, clip) {
    setClipLibraryByMode((current) => ({
      ...current,
      [targetMode]: [clip, ...current[targetMode]].slice(0, 18),
    }));
  }

  function getTaskConfig(targetMode, index) {
    return (
      taskConfigsByMode[targetMode][index] ?? {
        durationSeconds: 3,
        repeatCount: 3,
      }
    );
  }

  function buildGraphPath(values, maxValue) {
    if (!values.length) {
      return "";
    }

    return values
      .map((value, index) => {
        const x = values.length === 1 ? 0 : (index / (values.length - 1)) * 100;
        const y = 100 - (Math.min(Math.max(value, 0), maxValue) / Math.max(maxValue, 1)) * 100;
        return `${index === 0 ? "M" : "L"} ${x} ${y}`;
      })
      .join(" ");
  }

  function wait(ms) {
    return new Promise((resolve) => {
      window.setTimeout(resolve, ms);
    });
  }

  function getBridgeEndpoint(path) {
    return `${bridgeUrl}${path}`;
  }

  async function refreshServoToolStatus(options = {}) {
    const { silent = false } = options;

    try {
      const response = await fetch(getBridgeEndpoint("/servo-tool/status"));
      if (!response.ok) {
        throw new Error("bridge unavailable");
      }
      const payload = await response.json();
      setServoToolState({
        bridgeOnline: true,
        toolRootExists: payload.toolRootExists,
        appBundleExists: payload.appBundleExists,
        pythonEntryExists: payload.pythonEntryExists,
        isRunning: payload.isRunning,
        message: payload.isRunning
          ? "LeRobot 제어 도구가 실행 중입니다."
          : payload.toolRootExists
            ? "LeRobot 제어 도구를 실행할 수 있습니다."
            : "feetech-servo-tool 폴더를 찾지 못했습니다.",
        toolRoot: payload.toolRoot ?? "/Users/almond/feetech-servo-tool",
      });
      if (payload.connection) {
        setRobotConnection(payload.connection);
        if (payload.connection.port) {
          setSelectedPort(payload.connection.port);
        }
        if (payload.connection.servoId != null) {
          setSelectedServoId(String(payload.connection.servoId));
        }
      }
    } catch (error) {
      if (!silent) {
        console.info("servo bridge offline", error);
      }
      setServoToolState((current) => ({
        ...current,
        bridgeOnline: false,
        message:
          "로컬 서보 브리지가 꺼져 있습니다. 로봇이 연결된 컴퓨터에서 `npm run servo-bridge`로 실행하세요.",
      }));
    }
  }

  async function launchServoTool() {
    try {
      const response = await fetch(getBridgeEndpoint("/servo-tool/launch"), {
        method: "POST",
      });
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.message || "서보 툴 실행에 실패했습니다.");
      }
      setServoToolState((current) => ({
        ...current,
        bridgeOnline: true,
        toolRootExists: payload.toolRootExists ?? current.toolRootExists,
        appBundleExists: payload.appBundleExists ?? current.appBundleExists,
        pythonEntryExists: payload.pythonEntryExists ?? current.pythonEntryExists,
        isRunning: payload.isRunning ?? true,
        message: payload.message || "LeRobot 제어 도구를 실행했습니다.",
      }));
    } catch (error) {
      setServoToolState((current) => ({
        ...current,
        message:
          error instanceof Error
            ? error.message
            : "서보 툴 실행에 실패했습니다. 브리지 상태를 확인하세요.",
      }));
    }
  }

  async function loadRobotPorts() {
    try {
      const response = await fetch(getBridgeEndpoint("/robot/ports"));
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.message || "포트 목록을 불러오지 못했습니다.");
      }
      setSerialPorts(payload.ports ?? []);
      if (!selectedPort && payload.ports?.length) {
        setSelectedPort(payload.ports[0].device);
      }
    } catch (error) {
      setServoToolState((current) => ({
        ...current,
        message:
          error instanceof Error ? error.message : "포트 목록을 불러오지 못했습니다.",
      }));
    }
  }

  async function refreshRobotConnection(options = {}) {
    const { silent = false } = options;

    try {
      const response = await fetch(getBridgeEndpoint("/robot/connection"));
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.message || "연결 상태를 읽지 못했습니다.");
      }
      setRobotConnection(payload.connection ?? null);
      if (payload.connection?.port) {
        setSelectedPort(payload.connection.port);
      }
      if (payload.connection?.servoId != null) {
        setSelectedServoId(String(payload.connection.servoId));
      }
    } catch (error) {
      if (!silent) {
        console.info("robot connection unavailable", error);
      }
    }
  }

  async function refreshRobotMetrics(options = {}) {
    const { silent = false } = options;
    if (!robotConnection) {
      return;
    }

    try {
      const response = await fetch(getBridgeEndpoint("/robot/servo-status"));
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.message || "서보 상태를 읽지 못했습니다.");
      }
      setRobotMetrics(payload);
    } catch (error) {
      if (!silent) {
        setServoToolState((current) => ({
          ...current,
          message:
            error instanceof Error ? error.message : "서보 상태를 읽지 못했습니다.",
        }));
      }
    }
  }

  async function autoConnectRobot() {
    if (!selectedPort) {
      setServoToolState((current) => ({
        ...current,
        message: "먼저 연결할 시리얼 포트를 선택하세요.",
      }));
      return;
    }

    try {
      const response = await fetch(getBridgeEndpoint("/robot/connect"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          port: selectedPort,
          baudRate,
          timeoutMs,
          startId: 0,
          endId: 12,
          servoId: selectedServoId || undefined,
        }),
      });
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.message || "로봇 자동 연결에 실패했습니다.");
      }
      setRobotConnection(payload.connection);
      setDetectedServos(payload.servos ?? []);
      if (payload.connection?.servoId != null) {
        setSelectedServoId(String(payload.connection.servoId));
      }
      setServoToolState((current) => ({
        ...current,
        message: payload.message || "로봇 연결이 완료되었습니다.",
      }));
      await refreshRobotMetrics();
    } catch (error) {
      setServoToolState((current) => ({
        ...current,
        message:
          error instanceof Error ? error.message : "로봇 자동 연결에 실패했습니다.",
      }));
    }
  }

  async function sendRobotMove(goal) {
    if (!robotConnection) {
      setServoToolState((current) => ({
        ...current,
        message: "먼저 로봇을 연결하세요.",
      }));
      return;
    }

    try {
      const response = await fetch(getBridgeEndpoint("/robot/move"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          goal,
          speed: robotSpeed,
          acc: robotAcceleration,
          mode: controlMode,
          torqueEnabled,
        }),
      });
      const payload = await response.json();
      if (!response.ok || payload.ok === false) {
        throw new Error(payload.message || "동작 명령 전송에 실패했습니다.");
      }
      setRobotGoal(goal);
      setServoToolState((current) => ({
        ...current,
        message: `서보 목표 위치 ${goal} 명령을 전송했습니다.`,
      }));
      await refreshRobotMetrics();
    } catch (error) {
      setServoToolState((current) => ({
        ...current,
        message:
          error instanceof Error ? error.message : "동작 명령 전송에 실패했습니다.",
      }));
    }
  }

  function stopSweep() {
    if (sweepTimerRef.current) {
      window.clearTimeout(sweepTimerRef.current);
      sweepTimerRef.current = null;
    }
    setIsSweepRunning(false);
  }

  async function runStepSequence() {
    if (!robotConnection) {
      setServoToolState((current) => ({ ...current, message: "먼저 로봇을 연결하세요." }));
      return;
    }

    stopSweep();
    const nextGoal = Math.min(autoEnd, robotGoal + autoStep);
    await sendRobotMove(nextGoal);
  }

  function startSweep() {
    if (!robotConnection) {
      setServoToolState((current) => ({ ...current, message: "먼저 로봇을 연결하세요." }));
      return;
    }

    stopSweep();
    setIsSweepRunning(true);
    let currentGoal = autoStart;
    let direction = 1;

    const tick = async () => {
      await sendRobotMove(currentGoal);
      if (currentGoal >= autoEnd) {
        direction = -1;
      } else if (currentGoal <= autoStart) {
        direction = 1;
      }
      currentGoal += autoStep * direction;
      currentGoal = Math.max(autoStart, Math.min(autoEnd, currentGoal));
      sweepTimerRef.current = window.setTimeout(tick, autoSweepDelay);
    };

    void tick();
  }

  function stopRecording() {
    if (recordTimerRef.current) {
      window.clearInterval(recordTimerRef.current);
      recordTimerRef.current = null;
    }
    setIsRecording(false);
  }

  function exportRecordedFrames() {
    if (!recordedFrames.length) {
      setServoToolState((current) => ({ ...current, message: "내보낼 기록이 아직 없습니다." }));
      return;
    }

    const rows = [
      ["No", "Time", "Position", "Goal", "Torque", "Speed", "Current", "Temp", "Voltage"].join(","),
      ...recordedFrames.map((row, index) =>
        [
          index + 1,
          row.time,
          row.position,
          row.goal,
          row.load,
          row.speed,
          row.current,
          row.temperature,
          row.voltage,
        ].join(","),
      ),
    ];
    downloadBlob(new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" }), recordFilename);
  }

  function startRecording() {
    if (!robotConnection) {
      setServoToolState((current) => ({ ...current, message: "먼저 로봇을 연결하세요." }));
      return;
    }

    stopRecording();
    setRecordedFrames([]);
    setIsRecording(true);
    const startedAt = Date.now();

    recordTimerRef.current = window.setInterval(async () => {
      try {
        const response = await fetch(getBridgeEndpoint("/robot/servo-status"));
        const payload = await response.json();
        if (payload.ok) {
          setRobotMetrics(payload);
          setRecordedFrames((current) => [
            ...current,
            {
              time: ((Date.now() - startedAt) / 1000).toFixed(2),
              ...payload,
            },
          ]);
        }
      } catch (error) {
        console.error(error);
      }

      if (Date.now() - startedAt >= recordSeconds * 1000) {
        stopRecording();
      }
    }, 1000);
  }

  function setModeCounts(targetMode, values) {
    setSampleCountsByMode((current) => ({ ...current, [targetMode]: values }));
  }

  function incrementModeCount(targetMode, classIndex) {
    setSampleCountsByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode].map((count, index) =>
        index === classIndex ? count + 1 : count,
      ),
    }));
  }

  function getAudioClassKeys() {
    return classNamesByMode.audio.map((_, index) => `audio-class-${index}`);
  }

  function updateReady(targetMode, value) {
    setReadyByMode((current) => ({ ...current, [targetMode]: value }));
  }

  function updateTrained(targetMode, value) {
    setTrainedByMode((current) => ({ ...current, [targetMode]: value }));
  }

  function updateClassName(targetMode, index, value) {
    setClassNamesByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode].map((name, itemIndex) =>
        itemIndex === index ? value : name,
      ),
    }));
  }

  function updateTaskConfig(targetMode, index, key, value) {
    setTaskConfigsByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode].map((config, itemIndex) =>
        itemIndex === index
          ? {
              ...config,
              [key]: Math.max(1, Number(value) || 1),
            }
          : config,
      ),
    }));
  }

  function addClass(targetMode) {
    const nextCount = classNamesByMode[targetMode].length + 1;
    const nextName = `태스크 ${nextCount}`;

    setClassNamesByMode((current) => ({
      ...current,
      [targetMode]: [...current[targetMode], nextName],
    }));
    setSampleCountsByMode((current) => ({
      ...current,
      [targetMode]: [...current[targetMode], 0],
    }));
    setPredictionsByMode((current) => ({
      ...current,
      [targetMode]: [...current[targetMode], 0],
    }));
    setTaskConfigsByMode((current) => ({
      ...current,
      [targetMode]: [
        ...current[targetMode],
        {
          durationSeconds: 3,
          repeatCount: 3,
        },
      ],
    }));
    updateTrained(targetMode, false);
    setModeProgress(targetMode, 0);
    setModeStatus(targetMode, "태스크를 추가했습니다. 길이와 반복 횟수를 설정한 뒤 에피소드를 기록하세요.");

    if (targetMode === "image") {
      clearImageSamples();
      imageClassifierRef.current?.dispose();
      imageClassifierRef.current = null;
    }

    if (targetMode === "audio") {
      audioTransferRecognizerRef.current?.clearExamples();
    }

    if (targetMode === "pose") {
      clearPoseSamples();
      poseClassifierRef.current?.dispose();
      poseClassifierRef.current = null;
    }
  }

  function removeClass(targetMode, classIndex) {
    if (classNamesByMode[targetMode].length <= minimumClassCount) {
      setModeStatus(targetMode, `태스크는 최소 ${minimumClassCount}개 이상 있어야 합니다.`);
      return;
    }

    setClassNamesByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode].filter((_, index) => index !== classIndex),
    }));
    setSampleCountsByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode].filter((_, index) => index !== classIndex).map(() => 0),
    }));
    setPredictionsByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode].filter((_, index) => index !== classIndex).map(() => 0),
    }));
    setTaskConfigsByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode].filter((_, index) => index !== classIndex),
    }));
    setClipLibraryByMode((current) => ({
      ...current,
      [targetMode]: current[targetMode]
        .filter((clip) => clip.taskIndex !== classIndex)
        .map((clip) => ({
          ...clip,
          taskIndex: clip.taskIndex > classIndex ? clip.taskIndex - 1 : clip.taskIndex,
        })),
    }));
    updateTrained(targetMode, false);
    setModeProgress(targetMode, 0);
    setModeStatus(targetMode, "태스크를 삭제했습니다. 샘플을 다시 수집한 뒤 학습하세요.");

    if (targetMode === "image") {
      clearImageSamples();
      imageClassifierRef.current?.dispose();
      imageClassifierRef.current = null;
    }

    if (targetMode === "audio") {
      audioTransferRecognizerRef.current?.clearExamples();
    }

    if (targetMode === "pose") {
      clearPoseSamples();
      poseClassifierRef.current?.dispose();
      poseClassifierRef.current = null;
    }
  }

  function stopAllMediaAndLoops() {
    stopPreview();
    stopImageCaptureLoop();
    stopPoseCaptureLoop();
    stopImageCamera();
    stopPoseCamera();
  }

  async function prepareCurrentMode() {
    if (mode === "image") {
      if (imageStreamRef.current) {
        stopImageCamera();
        return;
      }
      await ensureImageResources();
      await startImageCamera();
      return;
    }

    if (mode === "audio") {
      await ensureAudioResources();
      return;
    }

    if (poseStreamRef.current) {
      stopPoseCamera();
      return;
    }

    await ensurePoseResources();
    await startPoseCamera();
  }

  async function ensureImageResources() {
    if (readyByMode.image) {
      return;
    }

    setIsModelLoading(true);
    setModeStatus("image", "LeRobot 비전 학습 백엔드를 준비하는 중입니다...");

    try {
      await tf.setBackend("webgl").catch(() => tf.setBackend("cpu"));
      await tf.ready();
      updateReady("image", true);
      setModeStatus("image", `비전 학습 백엔드 준비 완료: ${tf.getBackend()}. 로봇 카메라를 켜고 샘플을 수집하세요.`);
    } catch (error) {
      console.error(error);
      setModeStatus("image", "비전 학습 준비에 실패했습니다. 새로고침 후 다시 시도하세요.");
    } finally {
      setIsModelLoading(false);
    }
  }

  async function startImageCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      imageStreamRef.current = stream;
      if (imageVideoRef.current) {
        imageVideoRef.current.srcObject = stream;
        await imageVideoRef.current.play();
      }
      setIsImageCameraOn(true);
      setModeStatus("image", "로봇 카메라가 켜졌습니다. 클래스 버튼을 누르면 샘플 1개가 수집됩니다.");
    } catch (error) {
      console.error(error);
      setModeStatus("image", "카메라 권한이 차단되었습니다. 권한을 허용한 뒤 다시 시도하세요.");
    }
  }

  function stopImageCamera() {
    imageStreamRef.current?.getTracks().forEach((track) => track.stop());
    imageStreamRef.current = null;
    if (imageVideoRef.current) {
      imageVideoRef.current.srcObject = null;
    }
    setIsImageCameraOn(false);
  }

  function clearImageSamples() {
    imageSamplesRef.current.forEach(({ embedding }) => embedding.dispose());
    imageSamplesRef.current = [];
  }

  async function captureImageSample(classIndex) {
    await ensureImageResources();

    if (!imageVideoRef.current || !imageStreamRef.current || imageVideoRef.current.readyState < 2) {
      setModeStatus("image", "에피소드를 기록하려면 먼저 카메라를 켜세요.");
      return;
    }

    const taskConfig = getTaskConfig("image", classIndex);
    const repeats = Math.max(1, Number(taskConfig.repeatCount) || 1);
    const clipDurationSeconds = Math.max(1, Number(taskConfig.durationSeconds) || 1);
    const clipDurationMs = Math.max(1000, clipDurationSeconds * 1000);
    const sampleIntervalMs = 250;

    for (let episodeIndex = 0; episodeIndex < repeats; episodeIndex += 1) {
      const startedAt = Date.now();
      const metricsLog = [];
      const controlLog = [];

      while (Date.now() - startedAt < clipDurationMs) {
        const imageTensor = tf.tidy(() =>
          tf.browser
            .fromPixels(imageVideoRef.current)
            .resizeBilinear([trainingSettings.image.size, trainingSettings.image.size])
            .toFloat()
            .div(255)
            .expandDims(0),
        );
        imageSamplesRef.current.push({ embedding: imageTensor, label: classIndex });
        incrementModeCount("image", classIndex);

        metricsLog.push({
          t: Date.now() - startedAt,
          position: robotMetrics?.position ?? 0,
          load: robotMetrics?.load ?? 0,
          speed: robotMetrics?.speed ?? 0,
          current: robotMetrics?.current ?? 0,
          temperature: robotMetrics?.temperature ?? 0,
          voltage: robotMetrics?.voltage ?? 0,
        });
        controlLog.push({
          t: Date.now() - startedAt,
          goal: robotGoal,
          speed: robotSpeed,
          acc: robotAcceleration,
          controlMode,
          torqueEnabled,
        });
        await wait(sampleIntervalMs);
      }

      const previewCanvas = document.createElement("canvas");
      previewCanvas.width = imageVideoRef.current.videoWidth || 320;
      previewCanvas.height = imageVideoRef.current.videoHeight || 180;
      const context = previewCanvas.getContext("2d");
      context.drawImage(imageVideoRef.current, 0, 0, previewCanvas.width, previewCanvas.height);
      addClipRecord("image", {
        taskIndex: classIndex,
        taskName: classNamesByMode.image[classIndex],
        duration: clipDurationSeconds,
        repeats: 1,
        episodeIndex: episodeIndex + 1,
        sampleCount: metricsLog.length,
        thumbnail: previewCanvas.toDataURL("image/jpeg", 0.72),
        createdAt: Date.now(),
        metricsLog,
        controlLog,
      });
    }

    updateTrained("image", false);
    setModePredictions("image", classNamesByMode.image.map(() => 0));
    setModeProgress("image", 0);
    setModeStatus(
      "image",
      `${classNamesByMode.image[classIndex]} 에피소드 ${repeats}개를 기록했습니다. 상태값과 제어 로그도 함께 저장했습니다.`,
    );
  }

  function stopImageCaptureLoop() {
    if (imageHoldTimeoutRef.current) {
      window.clearTimeout(imageHoldTimeoutRef.current);
      imageHoldTimeoutRef.current = null;
    }
    if (imageCaptureTimerRef.current) {
      window.clearInterval(imageCaptureTimerRef.current);
      imageCaptureTimerRef.current = null;
    }
  }

  async function trainImageModel() {
    await ensureImageResources();

    if (classNamesByMode.image.length < 2) {
      setModeStatus("image", "LeRobot 학습을 시작하려면 태스크를 2개 이상 추가하세요.");
      return;
    }

    if (imageSamplesRef.current.length < trainingSettings.minSamples) {
      setModeStatus("image", "학습용 장면 샘플을 최소 9개 이상 수집하세요.");
      return;
    }

    setIsTraining(true);
    updateTrained("image", false);
    setIsPreviewRunning(false);
    setModeProgress("image", 0);
      setModeStatus("image", "로봇 카메라 텐서를 준비하는 중입니다...");

    try {
      imageClassifierRef.current?.dispose();

      setModeStatus("image", `샘플 ${imageSamplesRef.current.length}개를 학습 텐서로 묶는 중입니다...`);
      await tf.nextFrame();
      const xs = tf.concat(imageSamplesRef.current.map((sample) => sample.embedding));
      setModeStatus("image", "라벨 텐서를 준비하는 중입니다...");
      await tf.nextFrame();
      const labels = tf.tensor1d(
        imageSamplesRef.current.map((sample) => sample.label),
        "int32",
      );
      const classCount = classNamesByMode.image.length;
      const ys = tf.oneHot(labels, classCount);
      const inputShape = imageSamplesRef.current[0].embedding.shape.slice(1);
      const batchSize = Math.min(trainingSettings.image.batchSize, imageSamplesRef.current.length);
      const batchesPerEpoch = Math.max(1, Math.ceil(imageSamplesRef.current.length / batchSize));
      const totalBatches = trainingSettings.image.epochs * batchesPerEpoch;
      let completedBatches = 0;

      setModeStatus("image", "텐서 준비가 끝났습니다. LeRobot 비전 모델을 구성하는 중입니다...");
      await tf.nextFrame();

      const classifier = tf.sequential({
        layers: [
          tf.layers.conv2d({
            inputShape,
            filters: 8,
            kernelSize: 3,
            activation: "relu",
            padding: "same",
          }),
          tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
          tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            activation: "relu",
            padding: "same",
          }),
          tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
          tf.layers.flatten(),
          tf.layers.dense({ units: trainingSettings.image.denseUnits, activation: "relu" }),
          tf.layers.dropout({ rate: trainingSettings.image.dropout }),
          tf.layers.dense({ units: classCount, activation: "softmax" }),
        ],
      });

      classifier.compile({
        optimizer: tf.train.adam(0.0008),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
      });

      imageClassifierRef.current = classifier;
      setModeStatus("image", "비전 모델 구성이 끝났습니다. 학습을 시작합니다...");

      await classifier.fit(xs, ys, {
        epochs: trainingSettings.image.epochs,
        batchSize,
        shuffle: true,
        yieldEvery: "batch",
        callbacks: {
          onBatchEnd: async (_batch, logs) => {
            completedBatches += 1;
            const progress = Math.min(99, Math.round((completedBatches / totalBatches) * 100));
            setModeProgress("image", progress);
            setModeStatus(
              "image",
              `LeRobot 학습 중... ${progress}%${logs?.loss ? `, 손실 ${logs.loss.toFixed(3)}` : ""}`,
            );
            await tf.nextFrame();
          },
          onEpochEnd: async (epoch, logs) => {
            const progress = Math.round(((epoch + 1) / trainingSettings.image.epochs) * 100);
            setModeProgress("image", progress);
            setModeStatus(
              "image",
              `학습 ${progress}% 완료${logs?.accuracy ? `, 정확도 ${(logs.accuracy * 100).toFixed(0)}%` : ""}`,
            );
            await tf.nextFrame();
          },
        },
      });

      updateTrained("image", true);
      setModeStatus("image", "학습이 끝났습니다. 실시간 테스트를 시작해보세요.");

      xs.dispose();
      labels.dispose();
      ys.dispose();
    } catch (error) {
      console.error(error);
      setModeStatus(
        "image",
        `비전 학습에 실패했습니다. ${error?.message ? `오류: ${error.message}` : "초기화 후 다시 시도하세요."}`,
      );
    } finally {
      setIsTraining(false);
    }
  }

  async function runImagePrediction() {
    if (
      !imageClassifierRef.current ||
      !imageVideoRef.current ||
      !imageStreamRef.current ||
      imageVideoRef.current.readyState < 2
    ) {
      return;
    }

    const input = tf.tidy(() =>
      tf.browser
        .fromPixels(imageVideoRef.current)
        .resizeBilinear([trainingSettings.image.size, trainingSettings.image.size])
        .toFloat()
        .div(255)
        .expandDims(0),
    );
    const output = imageClassifierRef.current.predict(input);
    const values = Array.from(await output.data());
    input.dispose();
    output.dispose();
    setModePredictions("image", values);
  }

  async function ensureAudioResources() {
    if (audioTransferRecognizerRef.current) {
      updateReady("audio", true);
      setModeStatus("audio", "오디오 인식기 준비가 끝났습니다. 버튼을 눌러 마이크 샘플을 녹음하세요.");
      return;
    }

    setIsModelLoading(true);
    setModeStatus("audio", "Speech Commands 인식기를 불러오는 중입니다...");

    try {
      await tf.ready();
      const baseRecognizer = speechCommands.create("BROWSER_FFT");
      await baseRecognizer.ensureModelLoaded();
      audioBaseRecognizerRef.current = baseRecognizer;
      audioTransferRecognizerRef.current = baseRecognizer.createTransfer("model-studio-audio");
      updateReady("audio", true);
      setModeStatus("audio", "오디오 인식기 준비가 끝났습니다. 버튼을 눌러 마이크 샘플을 녹음하세요.");
    } catch (error) {
      console.error(error);
      setModeStatus("audio", "오디오 인식기 로딩에 실패했습니다. 새로고침 후 다시 시도하세요.");
    } finally {
      setIsModelLoading(false);
    }
  }

  async function captureAudioSample(classIndex) {
    await ensureAudioResources();
    const transferRecognizer = audioTransferRecognizerRef.current;

    if (!transferRecognizer) {
      return;
    }

    const label = getAudioClassKeys()[classIndex];
    setModeStatus("audio", `${classNamesByMode.audio[classIndex]} 샘플을 녹음하는 중입니다...`);

    try {
      await transferRecognizer.collectExample(label);
      const counts = transferRecognizer.countExamples();
      setModeCounts("audio", getAudioClassKeys().map((key) => counts[key] ?? 0));
      updateTrained("audio", false);
      setModePredictions("audio", classNamesByMode.audio.map(() => 0));
      setModeProgress("audio", 0);
      setModeStatus("audio", `${classNamesByMode.audio[classIndex]} 샘플 1개를 수집했습니다.`);
    } catch (error) {
      console.error(error);
      setModeStatus("audio", "오디오 수집에 실패했습니다. 마이크 권한을 확인하세요.");
    }
  }

  async function trainAudioModel() {
    await ensureAudioResources();
    const transferRecognizer = audioTransferRecognizerRef.current;
    const counts = sampleCountsByMode.audio;

    if (!transferRecognizer || counts.reduce((sum, count) => sum + count, 0) < trainingSettings.minSamples) {
      setModeStatus("audio", "오디오 샘플을 최소 9개 이상 수집하세요.");
      return;
    }

    setIsTraining(true);
    updateTrained("audio", false);
    setIsPreviewRunning(false);
    setModeProgress("audio", 0);
    setModeStatus("audio", "오디오 전이학습 모델을 학습하는 중입니다...");

    try {
      await transferRecognizer.train({
        epochs: 25,
        callback: {
          onEpochEnd: async (epoch, logs) => {
            const progress = Math.round(((epoch + 1) / 25) * 100);
            setModeProgress("audio", progress);
            setModeStatus(
              "audio",
              `학습 ${progress}% 완료${logs?.acc ? `, 정확도 ${(logs.acc * 100).toFixed(0)}%` : ""}`,
            );
            await tf.nextFrame();
          },
        },
      });

      updateTrained("audio", true);
      setModeStatus("audio", "오디오 학습이 끝났습니다. 실시간 미리보기를 켜고 소리를 내보세요.");
    } catch (error) {
      console.error(error);
      setModeStatus("audio", "오디오 학습에 실패했습니다. 더 선명한 예제로 다시 시도하세요.");
    } finally {
      setIsTraining(false);
    }
  }

  async function startAudioPreview() {
    await ensureAudioResources();
    const transferRecognizer = audioTransferRecognizerRef.current;

    if (!transferRecognizer || !trainedByMode.audio) {
      setModeStatus("audio", "오디오 미리보기를 시작하기 전에 먼저 학습하세요.");
      return;
    }

    try {
      const labelOrder = transferRecognizer.wordLabels();
      const audioClassKeys = getAudioClassKeys();
      await transferRecognizer.listen(
        ({ scores }) => {
          const mapped = audioClassKeys.map((key) => {
            const index = labelOrder.indexOf(key);
            return index >= 0 ? scores[index] : 0;
          });
          setModePredictions("audio", mapped);
        },
        { includeSpectrogram: false, probabilityThreshold: 0, overlapFactor: 0.5 },
      );
      setModeStatus("audio", "오디오 실시간 예측을 듣는 중입니다...");
      setIsPreviewRunning(true);
    } catch (error) {
      console.error(error);
      setModeStatus("audio", "오디오 실시간 예측에 실패했습니다. 마이크 권한을 확인하세요.");
    }
  }

  async function stopAudioPreview() {
    try {
      await audioTransferRecognizerRef.current?.stopListening();
    } catch (error) {
      console.info("Audio preview already stopped.", error);
    }
  }

  async function ensurePoseResources() {
    if (poseDetectorRef.current) {
      updateReady("pose", true);
      setModeStatus("pose", "포즈 모델 준비가 끝났습니다. 카메라를 켜고 클래스 버튼을 클릭하세요.");
      return;
    }

    setIsModelLoading(true);
    setModeStatus("pose", "MoveNet 포즈 모델을 불러오는 중입니다...");

    try {
      await tf.ready();
      poseDetectorRef.current = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
        },
      );
      updateReady("pose", true);
      setModeStatus("pose", "포즈 모델 준비가 끝났습니다. 카메라를 켜고 클래스 버튼을 클릭하세요.");
    } catch (error) {
      console.error(error);
      setModeStatus("pose", "포즈 모델 로딩에 실패했습니다. 새로고침 후 다시 시도하세요.");
    } finally {
      setIsModelLoading(false);
    }
  }

  async function startPoseCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      poseStreamRef.current = stream;
      if (poseVideoRef.current) {
        poseVideoRef.current.srcObject = stream;
        await poseVideoRef.current.play();
      }
      setIsPoseCameraOn(true);
      setModeStatus("pose", "포즈 카메라가 켜졌습니다. 자세를 취한 뒤 클래스 버튼을 클릭하세요.");
      startPoseLoop();
    } catch (error) {
      console.error(error);
      setModeStatus("pose", "포즈 카메라 권한이 차단되었습니다. 권한을 허용한 뒤 다시 시도하세요.");
    }
  }

  function stopPoseCamera() {
    poseStreamRef.current?.getTracks().forEach((track) => track.stop());
    poseStreamRef.current = null;
    if (poseVideoRef.current) {
      poseVideoRef.current.srcObject = null;
    }
    if (poseLoopFrameRef.current) {
      window.cancelAnimationFrame(poseLoopFrameRef.current);
      poseLoopFrameRef.current = null;
    }
    clearPoseCanvas();
    setIsPoseCameraOn(false);
  }

  function clearPoseCanvas() {
    const canvas = poseCanvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  function buildPoseVector(keypoints, width, height) {
    return keypoints.flatMap((keypoint) => {
      if (!keypoint || (keypoint.score ?? 0) < 0.2) {
        return [0, 0];
      }
      return [keypoint.x / width, keypoint.y / height];
    });
  }

  function drawPose(keypoints, width, height) {
    const canvas = poseCanvasRef.current;
    if (!canvas) {
      return;
    }

    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, width, height);
    context.fillStyle = "rgba(255,255,255,0.88)";

    keypoints.forEach((keypoint) => {
      if ((keypoint.score ?? 0) < 0.2) {
        return;
      }
      context.beginPath();
      context.arc(keypoint.x, keypoint.y, 5, 0, Math.PI * 2);
      context.fill();
    });
  }

  function startPoseLoop() {
    if (poseLoopFrameRef.current) {
      window.cancelAnimationFrame(poseLoopFrameRef.current);
    }

    const loop = async () => {
      if (
        !poseDetectorRef.current ||
        !poseVideoRef.current ||
        !poseStreamRef.current ||
        poseVideoRef.current.readyState < 2
      ) {
        poseLoopFrameRef.current = window.requestAnimationFrame(loop);
        return;
      }

      try {
        const poses = await poseDetectorRef.current.estimatePoses(poseVideoRef.current);
        const pose = poses[0];

        if (!pose || !pose.keypoints?.length) {
          currentPoseVectorRef.current = null;
          clearPoseCanvas();
          poseLoopFrameRef.current = window.requestAnimationFrame(loop);
          return;
        }

        drawPose(pose.keypoints, poseVideoRef.current.videoWidth, poseVideoRef.current.videoHeight);
        currentPoseVectorRef.current = buildPoseVector(
          pose.keypoints,
          poseVideoRef.current.videoWidth,
          poseVideoRef.current.videoHeight,
        );

        if (
          mode === "pose" &&
          isPreviewRunning &&
          trainedByMode.pose &&
          Date.now() - posePredictionAtRef.current > trainingSettings.previewIntervalMs
        ) {
          posePredictionAtRef.current = Date.now();
          await runPosePrediction();
        }
      } catch (error) {
        console.error(error);
      }

      poseLoopFrameRef.current = window.requestAnimationFrame(loop);
    };

    poseLoopFrameRef.current = window.requestAnimationFrame(loop);
  }

  function clearPoseSamples() {
    poseSamplesRef.current.forEach(({ vector }) => vector.dispose());
    poseSamplesRef.current = [];
  }

  async function capturePoseSample(classIndex) {
    await ensurePoseResources();

    if (!poseStreamRef.current) {
      setModeStatus("pose", "포즈 샘플을 수집하려면 먼저 카메라를 켜세요.");
      return;
    }

    const vector = currentPoseVectorRef.current;
    if (!vector) {
      setModeStatus("pose", "포즈가 잘 감지되지 않았습니다. 화면 안으로 들어와 다시 시도하세요.");
      return;
    }

    poseSamplesRef.current.push({ vector: tf.tensor2d([vector]), label: classIndex });
    incrementModeCount("pose", classIndex);
    updateTrained("pose", false);
    setModePredictions("pose", classNamesByMode.pose.map(() => 0));
    setModeProgress("pose", 0);
    setModeStatus("pose", `${classNamesByMode.pose[classIndex]} 샘플 1개를 수집했습니다.`);
  }

  function stopPoseCaptureLoop() {
    if (poseHoldTimeoutRef.current) {
      window.clearTimeout(poseHoldTimeoutRef.current);
      poseHoldTimeoutRef.current = null;
    }
    if (poseCaptureTimerRef.current) {
      window.clearInterval(poseCaptureTimerRef.current);
      poseCaptureTimerRef.current = null;
    }
  }

  function handleImageClassButtonClick(classIndex) {
    if (imageCaptureTimerRef.current) {
      return;
    }
    void captureImageSample(classIndex);
  }

  function handlePoseClassButtonClick(classIndex) {
    if (poseCaptureTimerRef.current) {
      return;
    }
    void capturePoseSample(classIndex);
  }

  async function trainPoseModel() {
    await ensurePoseResources();

    if (poseSamplesRef.current.length < trainingSettings.minSamples) {
      setModeStatus("pose", "포즈 샘플을 최소 9개 이상 수집하세요.");
      return;
    }

    setIsTraining(true);
    updateTrained("pose", false);
    setIsPreviewRunning(false);
    setModeProgress("pose", 0);
    setModeStatus("pose", "포즈 벡터를 준비하는 중입니다...");
    poseClassifierRef.current?.dispose();

    const xs = tf.concat(poseSamplesRef.current.map((sample) => sample.vector));
    const labels = tf.tensor1d(
      poseSamplesRef.current.map((sample) => sample.label),
      "int32",
    );
    const classCount = classNamesByMode.pose.length;
    const ys = tf.oneHot(labels, classCount);

    const classifier = tf.sequential({
      layers: [
        tf.layers.dense({
          inputShape: [poseSamplesRef.current[0].vector.shape[1]],
          units: trainingSettings.pose.denseUnits,
          activation: "relu",
        }),
        tf.layers.dropout({ rate: trainingSettings.pose.dropout }),
        tf.layers.dense({ units: classCount, activation: "softmax" }),
      ],
    });

    classifier.compile({
      optimizer: tf.train.adam(0.001),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    poseClassifierRef.current = classifier;

    try {
      await classifier.fit(xs, ys, {
        epochs: trainingSettings.pose.epochs,
        batchSize: Math.min(trainingSettings.pose.batchSize, poseSamplesRef.current.length),
        shuffle: true,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const progress = Math.round(((epoch + 1) / trainingSettings.pose.epochs) * 100);
            setModeProgress("pose", progress);
            setModeStatus(
              "pose",
              `학습 ${progress}% 완료${logs?.accuracy ? `, 정확도 ${(logs.accuracy * 100).toFixed(0)}%` : ""}`,
            );
            await tf.nextFrame();
          },
        },
      });

      updateTrained("pose", true);
      setModeStatus("pose", "포즈 학습이 끝났습니다. 실시간 미리보기로 자세를 테스트해보세요.");
    } catch (error) {
      console.error(error);
      setModeStatus("pose", "포즈 학습에 실패했습니다. 전신이 더 잘 보이게 다시 수집해보세요.");
    } finally {
      xs.dispose();
      labels.dispose();
      ys.dispose();
      setIsTraining(false);
    }
  }

  async function runPosePrediction() {
    if (!poseClassifierRef.current || !currentPoseVectorRef.current) {
      return;
    }

    const input = tf.tensor2d([currentPoseVectorRef.current]);
    const output = poseClassifierRef.current.predict(input);
    const values = Array.from(await output.data());
    input.dispose();
    output.dispose();
    setModePredictions("pose", values);
  }

  function stopPreview() {
    if (imagePreviewTimerRef.current) {
      window.clearInterval(imagePreviewTimerRef.current);
      imagePreviewTimerRef.current = null;
    }
    if (audioTransferRecognizerRef.current) {
      void stopAudioPreview();
    }
    setIsPreviewRunning(false);
  }

  async function togglePreview() {
    if (isPreviewRunning) {
      stopPreview();
      return;
    }

    if (mode === "image") {
      if (!trainedByMode.image || !isImageCameraOn) {
        setModeStatus("image", "실시간 테스트를 쓰려면 먼저 학습을 마치고 카메라를 켜세요.");
        return;
      }
      setIsPreviewRunning(true);
      setModeStatus("image", "이미지 실시간 예측을 실행 중입니다...");
      return;
    }

    if (mode === "audio") {
      await startAudioPreview();
      return;
    }

    if (!trainedByMode.pose || !isPoseCameraOn) {
      setModeStatus("pose", "실시간 미리보기를 쓰려면 먼저 포즈 학습을 하고 카메라를 켜세요.");
      return;
    }

    setIsPreviewRunning(true);
    setModeStatus("pose", "포즈 실시간 예측을 실행 중입니다...");
  }

  function resetCurrentMode() {
    stopPreview();

    if (mode === "image") {
      stopImageCaptureLoop();
      clearImageSamples();
      imageClassifierRef.current?.dispose();
      imageClassifierRef.current = null;
    }

    if (mode === "audio") {
      audioTransferRecognizerRef.current?.clearExamples();
    }

    if (mode === "pose") {
      stopPoseCaptureLoop();
      clearPoseSamples();
      poseClassifierRef.current?.dispose();
      poseClassifierRef.current = null;
    }

    setModeCounts(mode, classNamesByMode[mode].map(() => 0));
    setModePredictions(mode, classNamesByMode[mode].map(() => 0));
    setModeProgress(mode, 0);
    updateTrained(mode, false);
    setClipLibraryByMode((current) => ({ ...current, [mode]: [] }));
    setModeStatus(mode, `${projectModes[mode].badge} 작업공간을 초기화했습니다. 새 샘플을 수집하세요.`);
  }

  async function trainCurrentMode() {
    if (mode === "image") {
      await trainImageModel();
      return;
    }
    if (mode === "audio") {
      await trainAudioModel();
      return;
    }
    await trainPoseModel();
  }

  async function saveProject() {
    const payload = {
      projectName,
      classNamesByMode,
      sampleCountsByMode,
      taskConfigsByMode,
      clipLibraryByMode,
      mode,
      savedAt: new Date().toISOString(),
    };

    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    try {
      if (imageClassifierRef.current && trainedByMode.image) {
        await imageClassifierRef.current.save(IMAGE_MODEL_KEY);
      }
      if (poseClassifierRef.current && trainedByMode.pose) {
        await poseClassifierRef.current.save(POSE_MODEL_KEY);
      }
      setSaveMessage("프로젝트 정보를 로컬에 저장했습니다. 학습된 비전 모델도 함께 저장했습니다.");
    } catch (error) {
      console.error(error);
      setSaveMessage("프로젝트 정보는 저장했지만 일부 모델 저장에는 실패했습니다.");
    }
  }

  async function exportCurrentMode() {
    const slug = projectName.toLowerCase().replace(/[^a-z0-9]+/g, "-");

    if (mode === "image") {
      if (!imageClassifierRef.current || !trainedByMode.image) {
        setModeStatus("image", "비전 모델을 내보내려면 먼저 학습하세요.");
        return;
      }
      await imageClassifierRef.current.save(`downloads://${slug}-image-classifier`);
      setModeStatus("image", "비전 모델 내보내기를 시작했습니다.");
      return;
    }

    if (mode === "pose") {
      if (!poseClassifierRef.current || !trainedByMode.pose) {
        setModeStatus("pose", "포즈 모델을 내보내려면 먼저 학습하세요.");
        return;
      }
      await poseClassifierRef.current.save(`downloads://${slug}-pose-classifier`);
      setModeStatus("pose", "포즈 모델 내보내기를 시작했습니다.");
      return;
    }

    if (!audioTransferRecognizerRef.current) {
      setModeStatus("audio", "오디오를 내보내려면 먼저 인식기를 불러오고 학습하세요.");
      return;
    }

    try {
      if (typeof audioTransferRecognizerRef.current.save === "function" && trainedByMode.audio) {
        await audioTransferRecognizerRef.current.save(`downloads://${slug}-audio-classifier`);
        setModeStatus("audio", "오디오 모델 내보내기를 시작했습니다.");
        return;
      }

      if (typeof audioTransferRecognizerRef.current.serializeExamples === "function") {
        const buffer = await audioTransferRecognizerRef.current.serializeExamples();
        downloadBlob(
          new Blob([buffer], { type: "application/octet-stream" }),
          `${slug}-audio-examples.bin`,
        );
        downloadBlob(
          new Blob(
            [
              JSON.stringify(
                {
                  projectName,
                  labels: classNamesByMode.audio,
                  counts: sampleCountsByMode.audio,
                },
                null,
                2,
              ),
            ],
            { type: "application/json" },
          ),
          `${slug}-audio-project.json`,
        );
        setModeStatus("audio", "오디오 예제를 번들 파일로 내보냈습니다.");
      }
    } catch (error) {
      console.error(error);
      setModeStatus("audio", "오디오 내보내기에 실패했습니다. 다시 시도하세요.");
    }
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  function handleModeChange(nextMode) {
    startTransition(() => {
      setMode(nextMode);
    });
  }

  function openStudio(nextMode = mode, nextView = "simple") {
    startTransition(() => {
      setMode(nextMode);
      setStudioView(nextView);
      setIsStudioOpen(true);
    });
  }

  function triggerUpload(classIndex) {
    setPendingUploadClassIndex(classIndex);
    uploadInputRef.current?.click();
  }

  async function loadImageElementFromFile(file) {
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = dataUrl;
    });
  }

  async function handleFileUpload(event) {
    const files = Array.from(event.target.files ?? []);
    const classIndex = pendingUploadClassIndex;
    event.target.value = "";

    if (!files.length || classIndex == null) {
      return;
    }

    if (mode === "audio") {
      setModeStatus("audio", "오디오 파일 업로드는 아직 연결 중입니다. 현재는 마이크 녹음을 사용해주세요.");
      return;
    }

    try {
      if (mode === "image") {
        await ensureImageResources();
        let added = 0;

        for (const file of files) {
          const image = await loadImageElementFromFile(file);
          const embedding = tf.tidy(() =>
            tf.browser
              .fromPixels(image)
              .resizeBilinear([trainingSettings.image.size, trainingSettings.image.size])
              .toFloat()
              .div(255)
              .expandDims(0),
          );
          imageSamplesRef.current.push({ embedding, label: classIndex });
          added += 1;
        }

        setModeCounts(
          "image",
          sampleCountsByMode.image.map((count, index) =>
            index === classIndex ? count + added : count,
          ),
        );
        updateTrained("image", false);
        setModePredictions("image", classNamesByMode.image.map(() => 0));
        setModeStatus("image", `${added}개의 이미지 파일을 ${classNamesByMode.image[classIndex]} 클래스에 추가했습니다.`);
      }

      if (mode === "pose") {
        await ensurePoseResources();
        let added = 0;

        for (const file of files) {
          const image = await loadImageElementFromFile(file);
          const poses = await poseDetectorRef.current.estimatePoses(image);
          const pose = poses[0];
          if (!pose?.keypoints?.length) {
            continue;
          }

          const vector = buildPoseVector(
            pose.keypoints,
            image.width || 640,
            image.height || 480,
          );
          poseSamplesRef.current.push({ vector: tf.tensor2d([vector]), label: classIndex });
          added += 1;
        }

        setModeCounts(
          "pose",
          sampleCountsByMode.pose.map((count, index) =>
            index === classIndex ? count + added : count,
          ),
        );
        updateTrained("pose", false);
        setModePredictions("pose", classNamesByMode.pose.map(() => 0));
        setModeStatus("pose", `${added}개의 포즈 이미지 파일을 ${classNamesByMode.pose[classIndex]} 클래스에 추가했습니다.`);
      }
    } catch (error) {
      console.error(error);
      setModeStatus(mode, "파일 업로드 처리 중 오류가 발생했습니다.");
    } finally {
      setPendingUploadClassIndex(null);
    }
  }

  const inputButtonLabel =
    mode === "audio"
      ? isCurrentModeReady
        ? "세션 준비 완료"
        : projectModes.audio.inputAction
      : mode === "image"
        ? isImageCameraOn
          ? "카메라 끄기"
          : projectModes.image.inputAction
        : isPoseCameraOn
          ? "카메라 끄기"
          : projectModes.pose.inputAction;

  const canTrain =
    !isTraining && currentClassNames.length >= 2 && totalSamples >= trainingSettings.minSamples;
  const canPreview =
    (mode === "image" && trainedByMode.image && isImageCameraOn) ||
    (mode === "audio" && trainedByMode.audio) ||
    (mode === "pose" && trainedByMode.pose && isPoseCameraOn);

  return (
    <div className="page">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />
      <div className="grid-overlay" />

      <div className="shell">
        <section className="landing">
          <div className="landing-background" aria-hidden="true">
            <iframe
              src="https://my.spline.design/robotarm-tzv0cHF7ZNhS7EDMgBS2xgJe/"
              frameBorder="0"
              title="Robot arm background"
            />
            <div className="landing-backdrop" />
          </div>
          <div className="landing-mark" aria-hidden="true">
            LEROBOT
          </div>
          <div className="landing-copy">
            <span className="status-pill">LEROBOT PLATFORM</span>
            <h1>
              <span>Connect, Train, Test</span>
              <br />
              <span>LeRobot</span>
            </h1>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => openStudio("image", "simple")} type="button">
                바로 시작하기
              </button>
              <button className="secondary-button" onClick={() => openStudio("image", "developer")} type="button">
                개발자 모드
              </button>
            </div>
          </div>
        </section>

        {isStudioOpen && (
        <div className="studio-overlay" onClick={() => setIsStudioOpen(false)} role="presentation">
        <section className="editor-window" onClick={(event) => event.stopPropagation()}>
        <section className="editor" id="editor">
          <div className="editor-header">
            <div>
              <p className="eyebrow">플랫폼</p>
              <h2>{projectModes[mode].title}</h2>
              <p className="section-copy">{projectModes[mode].description}</p>
            </div>
            <div className="header-tools">
              <label className="project-name">
                  <span>프로젝트 이름</span>
                <input
                  onChange={(event) => setProjectName(event.target.value)}
                  type="text"
                  value={projectName}
                />
              </label>
              <div className="header-actions">
                <button
                  className={`toggle-button ${studioView === "simple" ? "active" : ""}`}
                  onClick={() => setStudioView("simple")}
                  type="button"
                >
                  메인 화면
                </button>
                <button
                  className={`toggle-button ${studioView === "developer" ? "active" : ""}`}
                  onClick={() => setStudioView("developer")}
                  type="button"
                >
                  개발자 화면
                </button>
                <button className="secondary-button" onClick={saveProject} type="button">
                  저장
                </button>
                <button className="secondary-button" onClick={() => void loadSavedProject()} type="button">
                  불러오기
                </button>
                <button
                  className="secondary-button"
                  disabled={!servoToolState.bridgeOnline}
                  onClick={() => void launchServoTool()}
                  type="button"
                >
                  LeRobot 실행
                </button>
                <button className="primary-button" onClick={() => void exportCurrentMode()} type="button">
                  내보내기
                </button>
                <button className="ghost-button" onClick={() => setIsStudioOpen(false)} type="button">
                  닫기
                </button>
              </div>
            </div>
          </div>

          <div className="save-message">{saveMessage}</div>

          {studioView === "simple" ? (
            <div className="simple-layout">
              <section className="panel simple-hero-panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">빠른 시작</p>
                    <h3>연결하고, 수집하고, GPU로 학습하기</h3>
                  </div>
                  <span className="status-pill">{projectModes[mode].badge}</span>
                </div>
                <div className="simple-intro-band">
                  <div>
                    <strong>초보자는 이 순서만 따라가면 됩니다.</strong>
                    <p>로봇 연결, 카메라 확인, 클립 수집, GPU 학습 시작까지 필요한 기능만 앞에 배치했습니다.</p>
                  </div>
                  <div className="simple-intro-actions">
                    <button className="secondary-button" onClick={() => setStudioView("developer")} type="button">
                      개발자 화면 열기
                    </button>
                  </div>
                </div>
                <div className="simple-summary-grid">
                  <div className="summary-card">
                    <span>연결 상태</span>
                    <strong>{robotConnection ? "로봇 연결됨" : "연결 전"}</strong>
                  </div>
                  <div className="summary-card">
                    <span>수집된 클립</span>
                    <strong>{currentClipCount}개</strong>
                  </div>
                  <div className="summary-card">
                    <span>태스크 수</span>
                    <strong>{currentTaskCount}개</strong>
                  </div>
                  <div className="summary-card">
                    <span>GPU 학습 상태</span>
                    <strong>{latestGpuJob ? gpuJobStateLabels[latestGpuJob.state] || latestGpuJob.state : "아직 없음"}</strong>
                  </div>
                </div>
                {mode === "audio" ? (
                  <section className="panel simple-camera-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">카메라 화면</p>
                        <h3>{projectModes[mode].previewLabel}</h3>
                      </div>
                      <button
                        className="secondary-button"
                        disabled={isModelLoading || (mode === "audio" && isCurrentModeReady)}
                        onClick={() => void prepareCurrentMode()}
                        type="button"
                      >
                        {isModelLoading ? "불러오는 중..." : inputButtonLabel}
                      </button>
                    </div>
                    <div className="audio-stage simple-audio-stage">
                      <div className="audio-bars">
                        <span />
                        <span />
                        <span />
                        <span />
                        <span />
                        <span />
                      </div>
                      <div className="audio-stage-copy">
                        <strong>{isPreviewRunning ? "실시간 마이크 미리보기" : "마이크 준비 완료"}</strong>
                        <p>{projectModes.audio.helper}</p>
                      </div>
                    </div>
                  </section>
                ) : (
                  <section className="panel simple-camera-panel">
                    <div className="camera-analytics-layout simple-camera-layout">
                      <section className="camera-stage-panel">
                        <div className="panel-header">
                          <div>
                            <p className="mini-label">카메라 화면</p>
                            <h3>{projectModes[mode].previewLabel}</h3>
                          </div>
                          <button
                            className="secondary-button"
                            disabled={isModelLoading}
                            onClick={() => void prepareCurrentMode()}
                            type="button"
                          >
                            {isModelLoading ? "불러오는 중..." : inputButtonLabel}
                          </button>
                        </div>
                        <div className={`camera-stage ${mode}`}>
                          {mode === "image" ? (
                            <video muted playsInline ref={imageVideoRef} />
                          ) : (
                            <>
                              <video muted playsInline ref={poseVideoRef} />
                              <canvas className="pose-overlay-canvas" ref={poseCanvasRef} />
                            </>
                          )}
                          {!((mode === "image" && isImageCameraOn) || (mode === "pose" && isPoseCameraOn)) && (
                            <div className="camera-placeholder">
                              <div className="camera-placeholder-copy">
                                <strong>{projectModes[mode].previewLabel}</strong>
                                <p>{projectModes[mode].helper}</p>
                              </div>
                            </div>
                          )}
                          <div className="camera-overlay">
                            <span>{projectModes[mode].badge}</span>
                            <span>
                              {mode === "image"
                                ? isImageCameraOn
                                  ? "실시간 카메라 연결"
                                  : "카메라 대기 중"
                                : isPoseCameraOn
                                  ? "테스트 추적 중"
                                  : "테스트 대기 중"}
                            </span>
                          </div>
                        </div>
                      </section>
                    </div>
                  </section>
                )}
                <div className="simple-flow-grid">
                  <article className="simple-step-card">
                    <div className="simple-step-number">1</div>
                    <strong>로봇 연결</strong>
                    <p>포트를 감지하고 자동 연결을 시도합니다. 초보자는 이 버튼만 누르면 됩니다.</p>
                    <div className="robot-toolbar">
                      <button className="secondary-button" onClick={() => void loadRobotPorts()} type="button">
                        포트 찾기
                      </button>
                      <button className="primary-button" onClick={() => void autoConnectRobot()} type="button">
                        연결하기
                      </button>
                    </div>
                  </article>

                  <article className="simple-step-card">
                    <div className="simple-step-number">2</div>
                    <strong>클립 수집</strong>
                    <p>위 카메라 화면을 확인한 뒤 태스크별로 에피소드 클립을 수집합니다.</p>
                    <div className="robot-toolbar">
                      <button className="secondary-button" onClick={() => addClass(mode)} type="button">
                        태스크 추가
                      </button>
                      <button className="secondary-button" onClick={() => triggerUpload(0)} type="button" disabled={!currentClassNames.length}>
                        파일 업로드
                      </button>
                    </div>
                    <div className="simple-task-list">
                      {currentClassNames.length ? (
                        currentClassNames.map((label, index) => (
                          <div className="simple-task-item" key={`simple-task-${mode}-${index}`}>
                            <div>
                              <strong>{label}</strong>
                              <span>{currentSampleCounts[index]}개 샘플 · {clipLibraryByMode[mode].filter((clip) => clip.taskIndex === index).length}개 클립</span>
                            </div>
                            {mode === "audio" ? (
                              <button className="secondary-button" disabled={isTraining} onClick={() => void captureAudioSample(index)} type="button">
                                샘플 녹음
                              </button>
                            ) : (
                              <button
                                className="primary-button"
                                disabled={(mode === "image" && !isImageCameraOn) || (mode === "pose" && !isPoseCameraOn)}
                                onClick={() => (mode === "image" ? handleImageClassButtonClick(index) : handlePoseClassButtonClick(index))}
                                type="button"
                              >
                                클립 기록
                              </button>
                            )}
                          </div>
                        ))
                      ) : (
                        <div className="workspace-empty compact">
                          <strong>아직 태스크가 없습니다.</strong>
                          <p>먼저 태스크를 하나 만들고 카메라를 켠 뒤 클립을 모아보세요.</p>
                        </div>
                      )}
                    </div>
                  </article>

                  <article className="simple-step-card">
                    <div className="simple-step-number">3</div>
                    <strong>외부 GPU 학습</strong>
                    <p>브라우저 대신 외부 GPU로 학습 작업을 보내 VESSL 같은 환경에서 훈련할 수 있게 준비합니다.</p>
                    <div className="robot-toolbar">
                      <button className="secondary-button" onClick={syncAgentFromProject} type="button">
                        프로젝트 상태 반영
                      </button>
                      <button className="primary-button" disabled={gpuBusy || currentClipCount === 0} onClick={() => void submitGpuTrainingJob()} type="button">
                        {gpuBusy ? "전송 중..." : "GPU 학습 시작"}
                      </button>
                    </div>
                    <div className="simple-job-box">
                      {latestGpuJob ? (
                        <>
                          <strong>{gpuJobStateLabels[latestGpuJob.state] || latestGpuJob.state}</strong>
                          <p>{latestGpuJob.message}</p>
                        </>
                      ) : (
                        <>
                          <strong>아직 GPU 작업이 없습니다.</strong>
                          <p>클립을 수집한 뒤 GPU 학습 시작 버튼을 눌러보세요.</p>
                        </>
                      )}
                    </div>
                    {gpuError && <div className="agent-error-box">{gpuError}</div>}
                  </article>

                  <article className="simple-step-card">
                    <div className="simple-step-number">4</div>
                    <strong>결과 확인</strong>
                    <p>AI 에이전트가 데이터가 충분한지, 더 수집해야 하는지 간단히 알려줍니다.</p>
                    <div className="simple-agent-status">
                      {agentSummary.map((item) => (
                        <div className="summary-card" key={`simple-agent-${item.label}`}>
                          <span>{item.label}</span>
                          <strong>{item.value}</strong>
                        </div>
                      ))}
                    </div>
                    <div className="simple-job-box">
                      <strong>AI 안내</strong>
                      <p>{agentStatus.logs[agentStatus.logs.length - 1]?.message || "아직 실행된 에이전트 로그가 없습니다."}</p>
                    </div>
                    <div className="robot-toolbar">
                      <button className="secondary-button" onClick={() => void refreshAgentStatus()} type="button">
                        안내 새로고침
                      </button>
                      <button className="primary-button" disabled={agentBusy || isAgentRunning} onClick={() => void startAgentLoop()} type="button">
                        AI 판단 실행
                      </button>
                    </div>
                  </article>
                </div>
              </section>
            </div>
          ) : (
          <div className="editor-layout">
            <div className="workspace">
              {mode === "audio" ? (
                <>
                  <div className="workspace-toolbar">
                    <div>
                      <p className="mini-label">카메라</p>
                      <h3>{projectModes[mode].previewLabel}</h3>
                    </div>
                    <button
                      className="secondary-button"
                      disabled={isModelLoading || (mode === "audio" && isCurrentModeReady)}
                      onClick={() => void prepareCurrentMode()}
                      type="button"
                    >
                      {isModelLoading ? "불러오는 중..." : inputButtonLabel}
                    </button>
                  </div>
                  <div className="audio-stage">
                    <div className="audio-bars">
                      <span />
                      <span />
                      <span />
                      <span />
                      <span />
                      <span />
                    </div>
                    <div className="audio-stage-copy">
                      <strong>{isPreviewRunning ? "실시간 마이크 미리보기" : "마이크 준비 완료"}</strong>
                      <p>{projectModes.audio.helper}</p>
                    </div>
                  </div>
                </>
              ) : (
                <div className="camera-analytics-layout">
                  <section className="camera-stage-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">카메라</p>
                        <h3>{projectModes[mode].previewLabel}</h3>
                      </div>
                      <button
                        className="secondary-button"
                        disabled={isModelLoading}
                        onClick={() => void prepareCurrentMode()}
                        type="button"
                      >
                        {isModelLoading ? "불러오는 중..." : inputButtonLabel}
                      </button>
                    </div>
                    <div className={`camera-stage ${mode}`}>
                      {mode === "image" ? (
                        <video muted playsInline ref={imageVideoRef} />
                      ) : (
                        <>
                          <video muted playsInline ref={poseVideoRef} />
                          <canvas className="pose-overlay-canvas" ref={poseCanvasRef} />
                        </>
                      )}
                      {!((mode === "image" && isImageCameraOn) || (mode === "pose" && isPoseCameraOn)) && (
                        <div className="camera-placeholder">
                          <div className="camera-placeholder-copy">
                            <strong>{projectModes[mode].previewLabel}</strong>
                            <p>{projectModes[mode].helper}</p>
                          </div>
                        </div>
                      )}
                      <div className="camera-overlay">
                        <span>{projectModes[mode].badge}</span>
                        <span>
                          {mode === "image"
                            ? isImageCameraOn
                              ? "실시간 카메라 연결"
                              : "카메라 대기 중"
                            : isPoseCameraOn
                              ? "테스트 추적 중"
                              : "테스트 대기 중"}
                        </span>
                      </div>
                    </div>
                  </section>

                  <section className="camera-graph-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">Debug Graph</p>
                        <h3>실시간 추이</h3>
                      </div>
                    </div>
                    <div className="servo-graph-card">
                      <svg className="servo-graph" preserveAspectRatio="none" viewBox="0 0 100 100">
                        {Array.from({ length: 5 }).map((_, index) => (
                          <line
                            key={`h-${index}`}
                            className="servo-grid-line"
                            x1="0"
                            x2="100"
                            y1={index * 25}
                            y2={index * 25}
                          />
                        ))}
                        {Array.from({ length: 6 }).map((_, index) => (
                          <line
                            key={`v-${index}`}
                            className="servo-grid-line"
                            x1={index * 20}
                            x2={index * 20}
                            y1="0"
                            y2="100"
                          />
                        ))}
                        <path className="servo-graph-line position" d={buildGraphPath(robotHistory.map((item) => item.position), 4095)} />
                        <path className="servo-graph-line torque" d={buildGraphPath(robotHistory.map((item) => item.load), 1000)} />
                        <path className="servo-graph-line speed" d={buildGraphPath(robotHistory.map((item) => item.speed), 1000)} />
                      </svg>
                      <div className="servo-graph-legend">
                        <span className="position">Position</span>
                        <span className="torque">Torque</span>
                        <span className="speed">Speed</span>
                      </div>
                    </div>
                  </section>
                </div>
              )}

              <div className="editor-summary">
                <div className="summary-card">
                  <span>카메라 준비</span>
                  <strong>{isCurrentModeReady ? "연결 완료" : "아직 연결 전"}</strong>
                </div>
                <div className="summary-card">
                  <span>수집 샘플</span>
                  <strong>{totalSamples}</strong>
                </div>
                <div className="summary-card">
                  <span>학습 상태</span>
                  <strong>{readinessText}</strong>
                </div>
              </div>

              <section className="panel workspace-class-panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">태스크 / 에피소드</p>
                    <h3>수집 태스크</h3>
                  </div>
                  <div className="class-panel-actions">
                    <button className="add-class-button" onClick={() => addClass(mode)} type="button">
                      + 태스크 추가
                    </button>
                    <button className="text-button" onClick={resetCurrentMode} type="button">
                      초기화
                    </button>
                  </div>
                </div>

                {currentClassNames.length ? (
                  <>
                    <div className="class-list">
                    {currentClassNames.map((label, index) => {
                      const taskConfig = getTaskConfig(mode, index);
                      const taskClips = clipLibraryByMode[mode].filter((clip) => clip.taskIndex === index);

                      return (
                        <article className={`class-card ${classTones[index % classTones.length]}`} key={`${mode}-${index}`}>
                          <div className="class-meta">
                            <div>
                              <strong>{`태스크 ${index + 1}`}</strong>
                              <input
                                className="class-name-input"
                                onChange={(event) => updateClassName(mode, index, event.target.value)}
                                type="text"
                                value={label}
                              />
                            </div>
                            <em>{currentSampleCounts[index]}개 샘플</em>
                          </div>

                          {mode !== "audio" && (
                            <div className="task-config-grid">
                              <label className="bridge-field">
                                <span>기록 길이(초)</span>
                                <input
                                  min="1"
                                  onChange={(event) =>
                                    updateTaskConfig(mode, index, "durationSeconds", event.target.value)
                                  }
                                  type="number"
                                  value={taskConfig.durationSeconds}
                                />
                              </label>
                              <label className="bridge-field">
                                <span>반복 횟수</span>
                                <input
                                  min="1"
                                  onChange={(event) =>
                                    updateTaskConfig(mode, index, "repeatCount", event.target.value)
                                  }
                                  type="number"
                                  value={taskConfig.repeatCount}
                                />
                              </label>
                            </div>
                          )}

                          <div className="class-card-actions">
                            {mode === "audio" ? (
                              <button
                                className={`record-button ${classTones[index % classTones.length]}`}
                                disabled={isTraining}
                                onClick={() => void captureAudioSample(index)}
                                type="button"
                              >
                                샘플 1개 녹음
                              </button>
                            ) : (
                              <>
                                <button
                                  className={`record-button ${classTones[index % classTones.length]}`}
                                  disabled={
                                    isTraining ||
                                    (mode === "image" && !isImageCameraOn) ||
                                    (mode === "pose" && !isPoseCameraOn)
                                  }
                                  onClick={() =>
                                    mode === "image"
                                      ? handleImageClassButtonClick(index)
                                      : handlePoseClassButtonClick(index)
                                  }
                                  type="button"
                                >
                                  에피소드 기록
                                </button>
                                <button className="upload-button" onClick={() => triggerUpload(index)} type="button">
                                  파일 업로드
                                </button>
                              </>
                            )}
                            {mode === "audio" && (
                              <button className="upload-button" disabled type="button">
                                파일 업로드 준비중
                              </button>
                            )}
                            <button className="remove-class-button" onClick={() => removeClass(mode, index)} type="button">
                              태스크 삭제
                            </button>
                          </div>

                          <div className="task-clip-group">
                            <div className="task-clip-group-header">
                              <strong>에피소드 클립</strong>
                              <span>{taskClips.length}개</span>
                            </div>
                            {taskClips.length ? (
                              <div className="task-clip-grid">
                                {taskClips.map((clip) => (
                                  <article className="clip-card compact" key={`${clip.createdAt}-${clip.taskIndex}-${clip.episodeIndex}`}>
                                    <img alt={clip.taskName} src={clip.thumbnail} />
                                    <div className="clip-card-body">
                                      <strong>{`Episode ${clip.episodeIndex}`}</strong>
                                      <span>{clip.duration}초 기록</span>
                                      <span>{clip.sampleCount ?? 0} 프레임</span>
                                      <span>{clip.metricsLog?.length ?? 0}개 상태 포인트</span>
                                      <span>{clip.controlLog?.length ?? 0}개 제어 로그</span>
                                    </div>
                                  </article>
                                ))}
                              </div>
                            ) : (
                              <div className="workspace-empty compact">
                                <strong>아직 기록된 에피소드가 없습니다.</strong>
                                <p>이 태스크의 길이와 반복 횟수를 정한 뒤 기록을 시작하세요.</p>
                              </div>
                            )}
                          </div>
                        </article>
                      );
                    })}
                    </div>

                    <button
                      className="primary-button wide class-train-button"
                      disabled={!canTrain}
                      onClick={() => void trainCurrentMode()}
                      type="button"
                    >
                      {isTraining ? "학습 중..." : "LeRobot 학습 시작"}
                    </button>
                  </>
                ) : (
                  <div className="workspace-empty">
                    <strong>아직 만든 태스크가 없습니다.</strong>
                    <p>`+ 태스크 추가` 버튼으로 원하는 만큼 수집 태스크를 만들어보세요.</p>
                  </div>
                )}
              </section>
            </div>

            <aside className="sidebar">
              <section className="panel ai-agent-panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">AI 에이전트</p>
                    <h3>수집·학습 진행 도우미</h3>
                  </div>
                  <div className={`agent-socket-badge ${agentSocketState}`}>
                    {agentSocketState === "open" ? "실시간 연결" : "상태 확인 중"}
                  </div>
                </div>

                <div className="agent-panel-copy">
                  <strong>지금 데이터를 더 모을지, 학습을 시작할지 AI가 판단합니다.</strong>
                  <p>이 에이전트는 로봇을 직접 가르치는 대신, 현재 프로젝트 상태를 보고 데이터 수집과 재학습 타이밍을 관리해 줍니다.</p>
                </div>

                <div className="agent-summary-grid">
                  {agentSummary.map((item) => (
                    <div className="summary-card" key={item.label}>
                      <span>{item.label}</span>
                      <strong>{item.value}</strong>
                    </div>
                  ))}
                </div>

                <div className="robot-panel-grid">
                  <label className="bridge-field">
                    <span>세션 이름</span>
                    <input onChange={(event) => setAgentSessionName(event.target.value)} type="text" value={agentSessionName} />
                  </label>
                  <label className="bridge-field">
                    <span>현재 데이터</span>
                    <input
                      onChange={(event) => updateAgentState("data_count", Number(event.target.value))}
                      type="number"
                      value={agentState.data_count}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>목표 데이터 수</span>
                    <input
                      onChange={(event) => updateAgentState("target_data", Number(event.target.value))}
                      type="number"
                      value={agentState.target_data}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>현재 손실값</span>
                    <input
                      onChange={(event) => updateAgentState("loss", Number(event.target.value))}
                      step="0.01"
                      type="number"
                      value={agentState.loss}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>목표 손실값</span>
                    <input
                      onChange={(event) => updateAgentState("target_loss", Number(event.target.value))}
                      step="0.01"
                      type="number"
                      value={agentState.target_loss}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>최대 반복</span>
                    <input
                      onChange={(event) => updateAgentState("max_iteration", Number(event.target.value))}
                      type="number"
                      value={agentState.max_iteration}
                    />
                  </label>
                </div>

                <div className="robot-toolbar agent-toolbar">
                  <button className="secondary-button" onClick={syncAgentFromProject} type="button">
                    현재 프로젝트 값 가져오기
                  </button>
                  <button className="secondary-button" onClick={() => void refreshAgentStatus()} type="button">
                    상태 새로고침
                  </button>
                  <button className="primary-button" disabled={agentBusy || isAgentRunning} onClick={() => void startAgentLoop()} type="button">
                    {agentBusy && !isAgentRunning ? "시작 중..." : "AI 루프 시작"}
                  </button>
                  <button className="ghost-button" disabled={agentBusy || !isAgentRunning} onClick={() => void stopAgentLoop()} type="button">
                    루프 중지
                  </button>
                </div>

                <div className="agent-result-box">
                  <p className="mini-label">최근 실행 결과</p>
                  <pre>{JSON.stringify(agentStatus.last_tool_result, null, 2)}</pre>
                </div>

                {(agentError || agentStatus.error) && (
                  <div className="agent-error-box">
                    {agentError || agentStatus.error}
                  </div>
                )}

                <div className="agent-log-list">
                  {agentStatus.logs.length ? (
                    agentStatus.logs.slice(-6).map((entry) => (
                      <article className={`agent-log-item ${entry.actor}`} key={`agent-log-${entry.step}`}>
                        <div className="agent-log-meta">
                          <span>{agentActorLabels[entry.actor] || entry.actor}</span>
                          <span>{entry.step}단계</span>
                        </div>
                        <p>{entry.message}</p>
                      </article>
                    ))
                  ) : (
                    <div className="workspace-empty compact">
                      <strong>아직 AI 에이전트 로그가 없습니다.</strong>
                      <p>루프를 시작하면 수집, 학습, 재학습 판단이 이 영역에 표시됩니다.</p>
                    </div>
                  )}
                </div>
              </section>

              <section className="panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">로봇 연결</p>
                    <h3>LeRobot 브리지</h3>
                  </div>
                  <button className="text-button" onClick={() => void refreshRobotConnection()} type="button">
                    새로고침
                  </button>
                </div>

                <div className="robot-panel-grid">
                  <label className="bridge-field">
                    <span>브리지 주소</span>
                    <input
                      onChange={(event) => setBridgeUrl(event.target.value)}
                      type="text"
                      value={bridgeUrl}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>시리얼 포트</span>
                    <select
                      className="bridge-select"
                      onChange={(event) => setSelectedPort(event.target.value)}
                      value={selectedPort}
                    >
                      <option value="">포트 선택</option>
                      {serialPorts.map((port) => (
                        <option key={port.device} value={port.device}>
                          {port.device}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="bridge-field">
                    <span>Baud Rate</span>
                    <input
                      onChange={(event) => setBaudRate(Number(event.target.value))}
                      type="number"
                      value={baudRate}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>TimeOut</span>
                    <input
                      onChange={(event) => setTimeoutMs(Number(event.target.value))}
                      type="number"
                      value={timeoutMs}
                    />
                  </label>
                </div>

                <div className="robot-toolbar bridge-toolbar">
                  <button className="secondary-button" onClick={() => void loadRobotPorts()} type="button">
                    포트 감지
                  </button>
                  <button className="primary-button" onClick={() => void autoConnectRobot()} type="button">
                    자동 연결
                  </button>
                  <button
                    className="secondary-button"
                    disabled={!servoToolState.bridgeOnline}
                    onClick={() => void launchServoTool()}
                    type="button"
                  >
                    제어 툴 열기
                  </button>
                </div>

                <div className="robot-status-list">
                  <div className="summary-card">
                    <span>연결 포트</span>
                    <strong>{robotConnection?.port ?? "미연결"}</strong>
                  </div>
                  <div className="summary-card">
                    <span>서보 모델</span>
                    <strong>{robotConnection?.modelName ?? "감지 전"}</strong>
                  </div>
                  <div className="summary-card">
                    <span>서보 ID</span>
                    <strong>{robotConnection?.servoId ?? "-"}</strong>
                  </div>
                </div>
              </section>

              <section className="panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">Servo Tool</p>
                    <h3>통합 제어 패널</h3>
                  </div>
                  <button className="secondary-button" onClick={() => void autoConnectRobot()} type="button">
                    Search
                  </button>
                </div>

                <div className="servo-tool-layout">
                  <section className="servo-subpanel servo-list-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">Servo List</p>
                        <h3>감지된 서보</h3>
                      </div>
                    </div>
                    <label className="bridge-field">
                      <span>선택 서보 ID</span>
                      <select
                        className="bridge-select"
                        onChange={(event) => setSelectedServoId(event.target.value)}
                        value={selectedServoId}
                      >
                        <option value="">자동 선택</option>
                        {detectedServos.map((servo) => (
                          <option key={`${servo.id}-${servo.modelNumber}`} value={String(servo.id)}>
                            {`ID ${servo.id} · ${servo.modelName}`}
                          </option>
                        ))}
                      </select>
                    </label>
                    <div className="servo-list-table">
                      <div className="servo-list-head">
                        <span>ID</span>
                        <span>Module</span>
                      </div>
                      {detectedServos.length ? (
                        detectedServos.map((servo) => (
                          <button
                            key={`servo-row-${servo.id}-${servo.modelNumber}`}
                            className={`servo-list-row ${String(servo.id) === selectedServoId ? "active" : ""}`}
                            onClick={() => setSelectedServoId(String(servo.id))}
                            type="button"
                          >
                            <span>{servo.id}</span>
                            <span>{servo.modelName}</span>
                          </button>
                        ))
                      ) : (
                        <div className="workspace-empty compact">
                          <strong>아직 감지된 서보가 없습니다.</strong>
                          <p>포트 감지 또는 Search를 눌러 목록을 불러오세요.</p>
                        </div>
                      )}
                    </div>
                  </section>

                  <section className="servo-subpanel servo-control-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">Servo Control</p>
                        <h3>제어 설정</h3>
                      </div>
                    </div>
                    <div className="control-mode-row">
                      <button
                        className={`toggle-button ${controlMode === "write" ? "active" : ""}`}
                        onClick={() => setControlMode("write")}
                        type="button"
                      >
                        Write
                      </button>
                      <button
                        className={`toggle-button ${controlMode === "sync" ? "active" : ""}`}
                        onClick={() => setControlMode("sync")}
                        type="button"
                      >
                        Sync Write
                      </button>
                      <button
                        className={`toggle-button ${controlMode === "reg" ? "active" : ""}`}
                        onClick={() => setControlMode("reg")}
                        type="button"
                      >
                        Reg Write
                      </button>
                      <label className="checkbox-row">
                        <input
                          checked={torqueEnabled}
                          onChange={(event) => setTorqueEnabled(event.target.checked)}
                          type="checkbox"
                        />
                        <span>Torque Enable</span>
                      </label>
                    </div>

                    <div className="robot-panel-grid movement-grid">
                      <label className="bridge-field">
                        <span>Acc</span>
                        <input
                          onChange={(event) => setRobotAcceleration(Number(event.target.value))}
                          type="number"
                          value={robotAcceleration}
                        />
                      </label>
                      <label className="bridge-field">
                        <span>Goal</span>
                        <input
                          onChange={(event) => setRobotGoal(Number(event.target.value))}
                          type="number"
                          value={robotGoal}
                        />
                      </label>
                      <label className="bridge-field">
                        <span>Speed</span>
                        <input
                          onChange={(event) => setRobotSpeed(Number(event.target.value))}
                          type="number"
                          value={robotSpeed}
                        />
                      </label>
                    </div>

                    <div className="robot-toolbar servo-actions-grid">
                      <button className="secondary-button" onClick={() => void refreshRobotMetrics()} type="button">
                        상태 읽기
                      </button>
                      <button className="secondary-button" onClick={() => void sendRobotMove(1024)} type="button">
                        Left
                      </button>
                      <button className="secondary-button" onClick={() => void sendRobotMove(2048)} type="button">
                        Center
                      </button>
                      <button className="primary-button" onClick={() => void sendRobotMove(robotGoal)} type="button">
                        Set
                      </button>
                    </div>
                  </section>

                  <section className="servo-subpanel servo-feedback-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">Servo Feedback</p>
                        <h3>실시간 피드백</h3>
                      </div>
                    </div>
                    <div className="feedback-grid">
                      <div className="summary-card"><span>Voltage</span><strong>{robotMetrics?.voltage ?? "0.0"}V</strong></div>
                      <div className="summary-card"><span>Torque</span><strong>{robotMetrics?.load ?? 0}</strong></div>
                      <div className="summary-card"><span>Current</span><strong>{robotMetrics?.current ?? 0}</strong></div>
                      <div className="summary-card"><span>Speed</span><strong>{robotMetrics?.speed ?? 0}</strong></div>
                      <div className="summary-card"><span>Temperature</span><strong>{robotMetrics?.temperature ?? 0}</strong></div>
                      <div className="summary-card"><span>Position</span><strong>{robotMetrics?.position ?? 0}</strong></div>
                      <div className="summary-card"><span>Moving</span><strong>{robotMetrics?.moving ?? 0}</strong></div>
                      <div className="summary-card"><span>Goal</span><strong>{robotMetrics?.goal ?? 0}</strong></div>
                    </div>
                  </section>

                  <section className="servo-subpanel servo-auto-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">Auto Debug</p>
                        <h3>자동 테스트</h3>
                      </div>
                      <button
                        className={`toggle-button ${isSweepRunning ? "active" : ""}`}
                        onClick={isSweepRunning ? stopSweep : startSweep}
                        type="button"
                      >
                        {isSweepRunning ? "Stop" : "Sweep"}
                      </button>
                    </div>
                    <div className="robot-panel-grid">
                      <label className="bridge-field">
                        <span>Start</span>
                        <input onChange={(event) => setAutoStart(Number(event.target.value))} type="number" value={autoStart} />
                      </label>
                      <label className="bridge-field">
                        <span>End</span>
                        <input onChange={(event) => setAutoEnd(Number(event.target.value))} type="number" value={autoEnd} />
                      </label>
                      <label className="bridge-field">
                        <span>Delay(Sweep)</span>
                        <input onChange={(event) => setAutoSweepDelay(Number(event.target.value))} type="number" value={autoSweepDelay} />
                      </label>
                      <label className="bridge-field">
                        <span>Step</span>
                        <input onChange={(event) => setAutoStep(Number(event.target.value))} type="number" value={autoStep} />
                      </label>
                      <label className="bridge-field">
                        <span>Step Delay</span>
                        <input onChange={(event) => setAutoStepDelay(Number(event.target.value))} type="number" value={autoStepDelay} />
                      </label>
                    </div>
                    <div className="robot-toolbar">
                      <button className="secondary-button" onClick={() => void runStepSequence()} type="button">
                        Step
                      </button>
                      <button className="secondary-button" onClick={() => void sendRobotMove(autoStart)} type="button">
                        Start 위치
                      </button>
                      <button className="secondary-button" onClick={() => void sendRobotMove(autoEnd)} type="button">
                        End 위치
                      </button>
                    </div>
                  </section>

                  <section className="servo-subpanel servo-data-panel">
                    <div className="panel-header">
                      <div>
                        <p className="mini-label">Data Analysis</p>
                        <h3>기록과 내보내기</h3>
                      </div>
                    </div>
                    <div className="robot-panel-grid">
                      <label className="bridge-field">
                        <span>time(s)</span>
                        <input onChange={(event) => setRecordSeconds(Number(event.target.value))} type="number" value={recordSeconds} />
                      </label>
                      <label className="bridge-field">
                        <span>file:rows</span>
                        <input onChange={(event) => setRecordFilename(event.target.value)} type="text" value={recordFilename} />
                      </label>
                    </div>
                    <div className="robot-toolbar">
                      <button
                        className={`toggle-button ${isRecording ? "active" : ""}`}
                        onClick={isRecording ? stopRecording : startRecording}
                        type="button"
                      >
                        {isRecording ? "Stop" : "Export"}
                      </button>
                      <button className="secondary-button" onClick={exportRecordedFrames} type="button">
                        CSV 저장
                      </button>
                      <button className="secondary-button" onClick={() => setRecordedFrames([])} type="button">
                        Empty
                      </button>
                    </div>
                    <div className="workspace-empty compact">
                      <strong>{recordedFrames.length} rows</strong>
                      <p>{isRecording ? "실시간으로 서보 데이터를 기록하는 중입니다." : "기록을 시작하면 CSV로 내보낼 수 있습니다."}</p>
                    </div>
                  </section>
                </div>
              </section>

              <section className="panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">테스트</p>
                    <h3>예측과 서보 상태</h3>
                  </div>
                  <button
                    className={`toggle-button ${isPreviewRunning ? "active" : ""}`}
                    disabled={!canPreview}
                    onClick={() => void togglePreview()}
                    type="button"
                  >
                    {isPreviewRunning ? "테스트 중지" : "실시간 테스트"}
                  </button>
                </div>

                <div className="prediction-list">
                  {currentClassNames.map((label, index) => (
                    <div className="prediction-row" key={`${mode}-prediction-${index}`}>
                      <span>{label}</span>
                      <div className="prediction-track">
                        <em
                          className={classTones[index % classTones.length]}
                          style={{ width: `${Math.round(currentPredictions[index] * 100)}%` }}
                        />
                      </div>
                      <strong>{Math.round(currentPredictions[index] * 100)}%</strong>
                    </div>
                  ))}
                </div>

                <div className="robot-status-list metrics">
                  <div className="summary-card">
                    <span>현재 위치</span>
                    <strong>{robotMetrics?.position ?? "-"}</strong>
                  </div>
                  <div className="summary-card">
                    <span>목표 위치</span>
                    <strong>{robotMetrics?.goal ?? "-"}</strong>
                  </div>
                  <div className="summary-card">
                    <span>온도</span>
                    <strong>{robotMetrics?.temperature != null ? `${robotMetrics.temperature}°C` : "-"}</strong>
                  </div>
                  <div className="summary-card">
                    <span>전압</span>
                    <strong>{robotMetrics?.voltage != null ? `${robotMetrics.voltage}V` : "-"}</strong>
                  </div>
                </div>

                <div className="robot-panel-grid movement-grid">
                  <label className="bridge-field">
                    <span>목표 위치</span>
                    <input
                      onChange={(event) => setRobotGoal(Number(event.target.value))}
                      type="number"
                      value={robotGoal}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>속도</span>
                    <input
                      onChange={(event) => setRobotSpeed(Number(event.target.value))}
                      type="number"
                      value={robotSpeed}
                    />
                  </label>
                  <label className="bridge-field">
                    <span>가속</span>
                    <input
                      onChange={(event) => setRobotAcceleration(Number(event.target.value))}
                      type="number"
                      value={robotAcceleration}
                    />
                  </label>
                </div>

                <div className="robot-toolbar">
                  <button className="secondary-button" onClick={() => void refreshRobotMetrics()} type="button">
                    상태 읽기
                  </button>
                  <button className="secondary-button" onClick={() => void sendRobotMove(1024)} type="button">
                    왼쪽 테스트
                  </button>
                  <button className="secondary-button" onClick={() => void sendRobotMove(2048)} type="button">
                    중앙 테스트
                  </button>
                  <button className="primary-button" onClick={() => void sendRobotMove(robotGoal)} type="button">
                    사용자 동작 전송
                  </button>
                </div>

                <div className="progress-block">
                  <div className="progress-track">
                    <span style={{ width: `${currentProgress}%` }} />
                  </div>
                  <p>{currentStatus}</p>
                </div>
              </section>
            </aside>
          </div>
          )}
        </section>
        </section>
        </div>
        )}
      </div>
      <input
        accept={mode === "image" || mode === "pose" ? "image/*" : ""}
        hidden
        multiple
        onChange={(event) => void handleFileUpload(event)}
        ref={uploadInputRef}
        type="file"
      />
    </div>
  );
}
