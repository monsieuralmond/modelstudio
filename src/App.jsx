import { startTransition, useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as speechCommands from "@tensorflow-models/speech-commands";
import * as poseDetection from "@tensorflow-models/pose-detection";

const STORAGE_KEY = "model-studio-project-v1";
const IMAGE_MODEL_KEY = "indexeddb://model-studio-image-classifier";
const POSE_MODEL_KEY = "indexeddb://model-studio-pose-classifier";

const projectModes = {
  image: {
    title: "이미지 프로젝트",
    badge: "이미지",
    subtitle: "웹캠 화면으로 시각 클래스를 학습합니다.",
    description:
      "웹캠 예제를 모으고 MobileNet 임베딩 위에서 분류기를 학습한 뒤, 브라우저에서 실시간 확률을 확인합니다.",
    helper: "이미지 모델을 불러온 뒤 카메라를 켜고 클래스 버튼을 클릭해 샘플을 수집하세요.",
    previewLabel: "웹캠 미리보기",
    inputAction: "카메라 켜기",
    trainingBadge: "실제 TensorFlow.js 학습",
  },
  audio: {
    title: "오디오 프로젝트",
    badge: "오디오",
    subtitle: "마이크 입력으로 소리 클래스를 학습합니다.",
    description:
      "클래스별 마이크 예제를 수집하고 speech-commands 전이학습 모델을 학습한 뒤 실시간 예측을 보여줍니다.",
    helper: "오디오 인식기를 불러온 뒤 각 클래스 버튼을 눌러 마이크 샘플을 녹음하세요.",
    previewLabel: "마이크 미리보기",
    inputAction: "오디오 준비",
    trainingBadge: "Speech Commands 전이학습",
  },
  pose: {
    title: "포즈 프로젝트",
    badge: "포즈",
    subtitle: "실시간 키포인트로 자세 클래스를 학습합니다.",
    description:
      "MoveNet으로 신체 키포인트를 추출하고 정규화된 포즈 벡터를 모아 브라우저에서 분류기를 학습합니다.",
    helper: "포즈 모델을 불러오고 카메라를 켠 뒤, 자세를 취한 상태에서 클래스 버튼을 클릭해 샘플을 수집하세요.",
    previewLabel: "포즈 미리보기",
    inputAction: "카메라 켜기",
    trainingBadge: "MoveNet + 사용자 분류기",
  },
};

const classTones = ["green", "purple", "orange"];
const minimumClassCount = 2;
const defaultClassNames = {
  image: ["손 흔들기", "브이 포즈", "중립 자세"],
  audio: ["박수", "휘파람", "조용함"],
  pose: ["손 들기", "옆 자세", "T자 자세"],
};

const modeIds = ["image", "audio", "pose"];

const trainingSettings = {
  image: { epochs: 18, batchSize: 8, denseUnits: 128, dropout: 0.2 },
  pose: { epochs: 24, batchSize: 8, denseUnits: 96, dropout: 0.15 },
  minSamples: 9,
  captureIntervalMs: 180,
  previewIntervalMs: 240,
};

export default function App() {
  const [mode, setMode] = useState("image");
  const [isStudioOpen, setIsStudioOpen] = useState(false);
  const [projectName, setProjectName] = useState("모델 스튜디오 데모");
  const [classNamesByMode, setClassNamesByMode] = useState(defaultClassNames);
  const [sampleCountsByMode, setSampleCountsByMode] = useState({
    image: [0, 0, 0],
    audio: [0, 0, 0],
    pose: [0, 0, 0],
  });
  const [predictionsByMode, setPredictionsByMode] = useState({
    image: [0, 0, 0],
    audio: [0, 0, 0],
    pose: [0, 0, 0],
  });
  const [statusByMode, setStatusByMode] = useState({
    image: "이미지 모델을 불러오면 시작할 수 있습니다.",
    audio: "오디오 인식기를 불러오면 시작할 수 있습니다.",
    pose: "포즈 모델을 불러오면 시작할 수 있습니다.",
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

  const imageBaseModelRef = useRef(null);
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
    totalSamples >= trainingSettings.minSamples
      ? "Ready to train"
      : `Collect ${trainingSettings.minSamples - totalSamples} samples`;

  useEffect(() => {
    void loadSavedProject();

    return () => {
      stopAllMediaAndLoops();
      clearImageSamples();
      clearPoseSamples();
      imageClassifierRef.current?.dispose();
      poseClassifierRef.current?.dispose();
      poseDetectorRef.current?.dispose?.();
    };
  }, []);

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
      if (saved.mode && modeIds.includes(saved.mode)) {
        setMode(saved.mode);
      }

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

  function addClass(targetMode) {
    const nextCount = classNamesByMode[targetMode].length + 1;
    const nextName = `${projectModes[targetMode].badge} 클래스 ${nextCount}`;

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
    updateTrained(targetMode, false);
    setModeProgress(targetMode, 0);
    setModeStatus(targetMode, "클래스를 추가했습니다. 샘플을 다시 수집한 뒤 학습하세요.");

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
      setModeStatus(targetMode, `클래스는 최소 ${minimumClassCount}개 이상 있어야 합니다.`);
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
    updateTrained(targetMode, false);
    setModeProgress(targetMode, 0);
    setModeStatus(targetMode, "클래스를 삭제했습니다. 샘플을 다시 수집한 뒤 학습하세요.");

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
    if (imageBaseModelRef.current) {
      return;
    }

    setIsModelLoading(true);
      setModeStatus("image", "MobileNet 기본 모델을 불러오는 중입니다...");

    try {
      await tf.ready();
      imageBaseModelRef.current = await mobilenet.load({ version: 2, alpha: 1 });
      updateReady("image", true);
      setModeStatus("image", "이미지 모델 준비가 끝났습니다. 카메라를 켜고 샘플을 수집하세요.");
    } catch (error) {
      console.error(error);
      setModeStatus("image", "이미지 모델 로딩에 실패했습니다. 새로고침 후 다시 시도하세요.");
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
      setModeStatus("image", "카메라가 켜졌습니다. 클래스 버튼을 클릭하면 샘플 1개가 수집됩니다.");
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
      setModeStatus("image", "이미지 샘플을 수집하려면 먼저 카메라를 켜세요.");
      return;
    }

    const embedding = tf.tidy(() => imageBaseModelRef.current.infer(imageVideoRef.current, true));
    imageSamplesRef.current.push({ embedding, label: classIndex });
    incrementModeCount("image", classIndex);
    updateTrained("image", false);
    setModePredictions("image", classNamesByMode.image.map(() => 0));
    setModeProgress("image", 0);
    setModeStatus("image", `${classNamesByMode.image[classIndex]} 샘플 1개를 수집했습니다.`);
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

    if (imageSamplesRef.current.length < trainingSettings.minSamples) {
      setModeStatus("image", "이미지 샘플을 최소 9개 이상 수집하세요.");
      return;
    }

    setIsTraining(true);
    updateTrained("image", false);
    setIsPreviewRunning(false);
    setModeProgress("image", 0);
    setModeStatus("image", "이미지 텐서를 준비하는 중입니다...");
    imageClassifierRef.current?.dispose();

    const xs = tf.concat(imageSamplesRef.current.map((sample) => sample.embedding));
    const labels = tf.tensor1d(
      imageSamplesRef.current.map((sample) => sample.label),
      "int32",
    );
    const classCount = classNamesByMode.image.length;
    const ys = tf.oneHot(labels, classCount);
    const inputShape = imageSamplesRef.current[0].embedding.shape.slice(1);

    const classifier = tf.sequential({
      layers: [
        tf.layers.flatten({ inputShape }),
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

    try {
      await classifier.fit(xs, ys, {
        epochs: trainingSettings.image.epochs,
        batchSize: Math.min(trainingSettings.image.batchSize, imageSamplesRef.current.length),
        shuffle: true,
        callbacks: {
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
      setModeStatus("image", "이미지 학습이 끝났습니다. 실시간 미리보기를 시작해보세요.");
    } catch (error) {
      console.error(error);
      setModeStatus("image", "이미지 학습에 실패했습니다. 초기화 후 다시 시도하세요.");
    } finally {
      xs.dispose();
      labels.dispose();
      ys.dispose();
      setIsTraining(false);
    }
  }

  async function runImagePrediction() {
    if (
      !imageClassifierRef.current ||
      !imageBaseModelRef.current ||
      !imageVideoRef.current ||
      !imageStreamRef.current ||
      imageVideoRef.current.readyState < 2
    ) {
      return;
    }

    const embedding = imageBaseModelRef.current.infer(imageVideoRef.current, true);
    const output = imageClassifierRef.current.predict(embedding);
    const values = Array.from(await output.data());
    embedding.dispose();
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
        setModeStatus("image", "실시간 미리보기를 쓰려면 먼저 이미지 학습을 하고 카메라를 켜세요.");
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
      setSaveMessage("프로젝트 정보를 로컬에 저장했습니다. 이미지/포즈 모델도 가능하면 함께 저장했습니다.");
    } catch (error) {
      console.error(error);
      setSaveMessage("프로젝트 정보는 저장했지만 일부 모델 저장에는 실패했습니다.");
    }
  }

  async function exportCurrentMode() {
    const slug = projectName.toLowerCase().replace(/[^a-z0-9]+/g, "-");

    if (mode === "image") {
      if (!imageClassifierRef.current || !trainedByMode.image) {
        setModeStatus("image", "이미지 모델을 내보내려면 먼저 학습하세요.");
        return;
      }
      await imageClassifierRef.current.save(`downloads://${slug}-image-classifier`);
      setModeStatus("image", "이미지 모델 내보내기를 시작했습니다.");
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

  function openStudio(nextMode = mode) {
    startTransition(() => {
      setMode(nextMode);
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
          const embedding = tf.tidy(() => imageBaseModelRef.current.infer(image, true));
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
        ? "Recognizer Ready"
        : projectModes.audio.inputAction
      : mode === "image"
        ? isImageCameraOn
          ? "Disable Camera"
          : projectModes.image.inputAction
        : isPoseCameraOn
          ? "Disable Camera"
          : projectModes.pose.inputAction;

  const canTrain = !isTraining && totalSamples >= trainingSettings.minSamples;
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
        <header className="topbar">
          <a className="brand" href="#top">
            <span className="brand-mark" />
            <span>모델 스튜디오</span>
          </a>
          <nav className="nav">
            <button className="nav-link" onClick={() => openStudio("image")} type="button">이미지</button>
            <button className="nav-link" onClick={() => openStudio("audio")} type="button">오디오</button>
            <button className="nav-link" onClick={() => openStudio("pose")} type="button">포즈</button>
          </nav>
          <button className="ghost-button" onClick={saveProject} type="button">
            프로젝트 저장
          </button>
        </header>

        <section className="landing">
          <div className="landing-copy">
            <p className="eyebrow">랜딩 페이지</p>
            <h1>이미지, 소리, 자세 데이터를 직접 모아 한국어 환경에서 바로 학습하세요.</h1>
            <p className="hero-body">
              모델 스튜디오는 웹 브라우저에서 샘플을 수집하고, 클래스를 직접 추가하고, 파일을 업로드하고, 학습과 예측까지 이어서 확인할 수 있는 실험용 인터페이스입니다.
            </p>
            <div className="hero-actions">
              <button className="primary-button" onClick={() => openStudio("image")} type="button">
                이미지 프로젝트 시작
              </button>
              <button className="secondary-button" onClick={() => openStudio("audio")} type="button">
                오디오 프로젝트 시작
              </button>
              <button className="secondary-button" onClick={() => openStudio("pose")} type="button">
                포즈 프로젝트 시작
              </button>
            </div>
          </div>

          <div className="landing-summary">
            <article className="landing-card">
              <strong>직접 샘플 수집</strong>
              <p>카메라, 마이크, 포즈 감지 또는 파일 업로드로 데이터셋을 구성할 수 있습니다.</p>
            </article>
            <article className="landing-card">
              <strong>클래스 자유 추가</strong>
              <p>처음부터 고정된 클래스가 아니라 `+ 클래스 추가` 버튼으로 원하는 만큼 늘릴 수 있습니다.</p>
            </article>
            <article className="landing-card">
              <strong>저장과 내보내기</strong>
              <p>프로젝트 정보를 로컬에 저장하고, 학습된 모델이나 예제 번들을 내려받을 수 있습니다.</p>
            </article>
          </div>
        </section>

        <section className="landing-features">
          <article className="feature-block">
            <p className="eyebrow">지원 입력</p>
            <h2>바로 시작할 수 있는 입력 방식</h2>
            <ul className="feature-list">
              <li>이미지: 웹캠 캡처 또는 이미지 파일 업로드</li>
              <li>오디오: 마이크 녹음 기반 샘플 수집</li>
              <li>포즈: 실시간 포즈 감지 또는 포즈 이미지 업로드</li>
            </ul>
          </article>
          <article className="feature-block">
            <p className="eyebrow">작업 방식</p>
            <h2>필요한 동작만 남긴 간단한 시작 흐름</h2>
            <ul className="feature-list">
              <li>프로젝트 카드를 누르면 같은 페이지 안에서 작업 창이 열립니다.</li>
              <li>클래스는 직접 추가하고 삭제할 수 있습니다.</li>
              <li>카메라 수집과 파일 업로드를 함께 사용할 수 있습니다.</li>
            </ul>
          </article>
        </section>

        <section className="project-switcher">
          {Object.entries(projectModes).map(([key, project]) => (
            <button
              key={key}
              className={`mode-card ${mode === key ? "active" : ""}`}
              onClick={() => openStudio(key)}
              type="button"
            >
              <span className={`mode-icon ${key}`} />
              <strong>{project.title}</strong>
              <span>{project.subtitle}</span>
            </button>
          ))}
        </section>

        {isStudioOpen && (
        <div className="studio-overlay" onClick={() => setIsStudioOpen(false)} role="presentation">
        <section className="editor-window" onClick={(event) => event.stopPropagation()}>
        <section className="editor" id="editor">
          <div className="editor-header">
            <div>
              <p className="eyebrow">에디터</p>
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
                <button className="secondary-button" onClick={saveProject} type="button">
                  저장
                </button>
                <button className="secondary-button" onClick={() => void loadSavedProject()} type="button">
                  불러오기
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

          <div className="editor-layout">
            <div className="workspace">
              <div className="workspace-toolbar">
                <div>
                  <p className="mini-label">입력</p>
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

              {mode === "audio" ? (
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
              ) : (
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
                      <strong>{projectModes[mode].previewLabel}</strong>
                      <p>{projectModes[mode].helper}</p>
                    </div>
                  )}
                  <div className="camera-overlay">
                    <span>{projectModes[mode].badge}</span>
                    <span>
                      {mode === "image"
                        ? isImageCameraOn
                          ? "Live session"
                          : "Idle preview"
                        : isPoseCameraOn
                          ? "Pose tracking live"
                          : "Idle preview"}
                    </span>
                  </div>
                </div>
              )}

              <div className="editor-summary">
                <div className="summary-card">
                  <span>상태</span>
                  <strong>{isCurrentModeReady ? "모델 준비 완료" : "아직 불러오지 않음"}</strong>
                </div>
                <div className="summary-card">
                  <span>샘플 수</span>
                  <strong>{totalSamples}</strong>
                </div>
                <div className="summary-card">
                  <span>학습 준비</span>
                  <strong>{readinessText}</strong>
                </div>
              </div>
            </div>

            <aside className="sidebar">
              <section className="panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">클래스</p>
                    <h3>학습 예제</h3>
                  </div>
                  <div className="class-panel-actions">
                    <button className="add-class-button" onClick={() => addClass(mode)} type="button">
                      + 클래스 추가
                    </button>
                    <button className="text-button" onClick={resetCurrentMode} type="button">
                      초기화
                    </button>
                  </div>
                </div>

                <div className="class-list">
                  {currentClassNames.map((label, index) => (
                    <article className={`class-card ${classTones[index % classTones.length]}`} key={`${mode}-${index}`}>
                      <div className="class-meta">
                        <div>
                          <strong>{`클래스 ${index + 1}`}</strong>
                          <input
                            className="class-name-input"
                            onChange={(event) => updateClassName(mode, index, event.target.value)}
                            type="text"
                            value={label}
                          />
                        </div>
                        <em>{currentSampleCounts[index]}개</em>
                      </div>
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
                              샘플 1개 수집
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
                          클래스 삭제
                        </button>
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <section className="panel">
                <div className="panel-header">
                  <div>
                    <p className="mini-label">학습</p>
                    <h3>모델 출력</h3>
                  </div>
                  <button
                    className={`toggle-button ${isPreviewRunning ? "active" : ""}`}
                    disabled={!canPreview}
                    onClick={() => void togglePreview()}
                    type="button"
                  >
                    {isPreviewRunning ? "미리보기 중지" : "실시간 미리보기"}
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

                <button
                  className="primary-button wide"
                  disabled={!canTrain}
                  onClick={() => void trainCurrentMode()}
                  type="button"
                >
                  {isTraining ? "학습 중..." : "모델 학습"}
                </button>

                <div className="progress-block">
                  <div className="progress-track">
                    <span style={{ width: `${currentProgress}%` }} />
                  </div>
                  <p>{currentStatus}</p>
                </div>
              </section>
            </aside>
          </div>
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
