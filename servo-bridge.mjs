import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { execFileSync, spawn } from "node:child_process";

const PORT = 5057;
const HOST = "0.0.0.0";
const TOOL_ROOT = process.env.FEETECH_TOOL_ROOT || "/Users/almond/feetech-servo-tool";
const HELPER_PATH = new URL("./servo-helper.py", import.meta.url).pathname;
const APP_BUNDLE_PATH = `${TOOL_ROOT}/dist/Servo Tool.app`;
const APP_BINARY_PATH = `${TOOL_ROOT}/dist/Servo Tool/Servo Tool`;
const PYTHON_ENTRY_PATH = `${TOOL_ROOT}/main.py`;
const VENV_PYTHON_PATH = `${TOOL_ROOT}/venv/bin/python`;

let activeRobotConnection = null;

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  response.end(JSON.stringify(payload));
}

function readJsonBody(request) {
  return new Promise((resolve, reject) => {
    let raw = "";
    request.on("data", (chunk) => {
      raw += chunk;
    });
    request.on("end", () => {
      if (!raw) {
        resolve({});
        return;
      }
      try {
        resolve(JSON.parse(raw));
      } catch (error) {
        reject(error);
      }
    });
    request.on("error", reject);
  });
}

function getPythonExecutable() {
  return existsSync(VENV_PYTHON_PATH) ? VENV_PYTHON_PATH : "python3";
}

function runHelper(args) {
  const output = execFileSync(getPythonExecutable(), [HELPER_PATH, ...args], {
    encoding: "utf8",
    env: {
      ...process.env,
      FEETECH_TOOL_ROOT: TOOL_ROOT,
    },
  });
  return JSON.parse(output);
}

function getRunningProcesses() {
  try {
    return execFileSync("ps", ["-ax", "-o", "command="], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    });
  } catch {
    return "";
  }
}

function getStatusPayload() {
  const processList = getRunningProcesses();
  const appBundleExists = existsSync(APP_BUNDLE_PATH);
  const appBinaryExists = existsSync(APP_BINARY_PATH);
  const pythonEntryExists = existsSync(PYTHON_ENTRY_PATH);
  const venvPythonExists = existsSync(VENV_PYTHON_PATH);
  const toolRootExists = existsSync(TOOL_ROOT);
  const isRunning =
    processList.includes(APP_BINARY_PATH) ||
    processList.includes(APP_BUNDLE_PATH) ||
    processList.includes(PYTHON_ENTRY_PATH);

  return {
    ok: true,
    toolRoot: TOOL_ROOT,
    toolRootExists,
    appBundlePath: APP_BUNDLE_PATH,
    appBundleExists,
    appBinaryPath: APP_BINARY_PATH,
    appBinaryExists,
    pythonEntryPath: PYTHON_ENTRY_PATH,
    pythonEntryExists,
    venvPythonPath: VENV_PYTHON_PATH,
    venvPythonExists,
    isRunning,
    launchTarget: appBundleExists ? "app" : pythonEntryExists ? "python" : null,
    connection: activeRobotConnection,
  };
}

function launchServoTool() {
  const status = getStatusPayload();

  if (!status.toolRootExists) {
    throw new Error("feetech-servo-tool 폴더를 찾지 못했습니다.");
  }

  if (status.appBundleExists) {
    const child = spawn("open", ["-a", APP_BUNDLE_PATH], {
      detached: true,
      stdio: "ignore",
    });
    child.unref();
    return {
      ...getStatusPayload(),
      launched: !status.isRunning,
      message: status.isRunning
        ? "Feetech Servo Tool 앱을 앞으로 가져왔습니다."
        : "Feetech Servo Tool 앱을 실행했습니다.",
    };
  }

  if (status.pythonEntryExists) {
    if (status.isRunning) {
      return {
        ...status,
        launched: false,
        message: "Feetech Servo Tool 파이썬 앱이 이미 실행 중입니다.",
      };
    }
    const child = spawn(getPythonExecutable(), [PYTHON_ENTRY_PATH], {
      cwd: TOOL_ROOT,
      detached: true,
      stdio: "ignore",
      env: {
        ...process.env,
        FEETECH_TOOL_ROOT: TOOL_ROOT,
      },
    });
    child.unref();
    return {
      ...getStatusPayload(),
      launched: true,
      message: "Feetech Servo Tool 파이썬 앱을 실행했습니다.",
    };
  }

  throw new Error("실행 가능한 feetech-servo-tool 진입점을 찾지 못했습니다.");
}

function handleHelperFailure(error) {
  return {
    ok: false,
    message: error instanceof Error ? error.message : "서보 브리지 처리 중 오류가 발생했습니다.",
  };
}

const server = createServer(async (request, response) => {
  if (request.method === "OPTIONS") {
    sendJson(response, 204, {});
    return;
  }

  if (request.url === "/health" && request.method === "GET") {
    sendJson(response, 200, { ok: true, service: "servo-bridge" });
    return;
  }

  if (request.url === "/servo-tool/status" && request.method === "GET") {
    sendJson(response, 200, getStatusPayload());
    return;
  }

  if (request.url === "/servo-tool/launch" && request.method === "POST") {
    try {
      sendJson(response, 200, launchServoTool());
    } catch (error) {
      sendJson(response, 500, handleHelperFailure(error));
    }
    return;
  }

  if (request.url === "/robot/ports" && request.method === "GET") {
    try {
      sendJson(response, 200, runHelper(["list-ports"]));
    } catch (error) {
      sendJson(response, 500, handleHelperFailure(error));
    }
    return;
  }

  if (request.url === "/robot/connection" && request.method === "GET") {
    sendJson(response, 200, { ok: true, connection: activeRobotConnection });
    return;
  }

  if (request.url === "/robot/disconnect" && request.method === "POST") {
    activeRobotConnection = null;
    sendJson(response, 200, { ok: true, connection: null, message: "로봇 연결 정보를 초기화했습니다." });
    return;
  }

  if (request.url === "/robot/connect" && request.method === "POST") {
    try {
      const body = await readJsonBody(request);
      const scanResult = runHelper([
        "scan",
        "--port",
        body.port,
        "--baud-rate",
        String(body.baudRate ?? 1000000),
        "--timeout-ms",
        String(body.timeoutMs ?? 50),
        "--start-id",
        String(body.startId ?? 0),
        "--end-id",
        String(body.endId ?? 12),
      ]);

      if (!scanResult.ok) {
        sendJson(response, 500, scanResult);
        return;
      }

      const requestedServoId =
        body.servoId == null || body.servoId === "" ? null : Number(body.servoId);
      const firstServo =
        requestedServoId == null
          ? scanResult.servos?.[0]
          : scanResult.servos?.find((servo) => servo.id === requestedServoId);
      if (!firstServo) {
        activeRobotConnection = null;
        sendJson(response, 200, {
          ok: false,
          message: "연결된 서보를 찾지 못했습니다. 포트와 전원을 확인하세요.",
          servos: [],
        });
        return;
      }

      activeRobotConnection = {
        port: body.port,
        baudRate: Number(body.baudRate ?? 1000000),
        timeoutMs: Number(body.timeoutMs ?? 50),
        servoId: firstServo.id,
        series: firstServo.series,
        modelName: firstServo.modelName,
        modelNumber: firstServo.modelNumber,
      };

      sendJson(response, 200, {
        ok: true,
        message: `${firstServo.modelName} 서보를 자동 감지했습니다.`,
        servos: scanResult.servos,
        connection: activeRobotConnection,
      });
    } catch (error) {
      sendJson(response, 500, handleHelperFailure(error));
    }
    return;
  }

  if (request.url === "/robot/servo-status" && request.method === "GET") {
    if (!activeRobotConnection) {
      sendJson(response, 400, { ok: false, message: "먼저 로봇을 연결하세요." });
      return;
    }

    try {
      const payload = runHelper([
        "read-status",
        "--port",
        activeRobotConnection.port,
        "--baud-rate",
        String(activeRobotConnection.baudRate),
        "--timeout-ms",
        String(activeRobotConnection.timeoutMs ?? 50),
        "--servo-id",
        String(activeRobotConnection.servoId),
        "--series",
        activeRobotConnection.series,
      ]);
      sendJson(response, 200, payload);
    } catch (error) {
      sendJson(response, 500, handleHelperFailure(error));
    }
    return;
  }

  if (request.url === "/robot/move" && request.method === "POST") {
    if (!activeRobotConnection) {
      sendJson(response, 400, { ok: false, message: "먼저 로봇을 연결하세요." });
      return;
    }

    try {
      const body = await readJsonBody(request);
      const payload = runHelper([
        "move",
        "--port",
        activeRobotConnection.port,
        "--baud-rate",
        String(activeRobotConnection.baudRate),
        "--timeout-ms",
        String(activeRobotConnection.timeoutMs ?? 50),
        "--servo-id",
        String(activeRobotConnection.servoId),
        "--series",
        activeRobotConnection.series,
        "--goal",
        String(body.goal ?? 2048),
        "--speed",
        String(body.speed ?? 300),
        "--acc",
        String(body.acc ?? 10),
      ]);
      sendJson(response, payload.ok ? 200 : 500, payload);
    } catch (error) {
      sendJson(response, 500, handleHelperFailure(error));
    }
    return;
  }

  sendJson(response, 404, { ok: false, message: "지원하지 않는 경로입니다." });
});

server.listen(PORT, HOST, () => {
  console.log(`servo-bridge listening on http://127.0.0.1:${PORT}`);
  console.log(`network access available on http://<robot-host-ip>:${PORT}`);
});
