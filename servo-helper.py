#!/usr/bin/env python3

import argparse
import json
import os
import sys

TOOL_ROOT = os.environ.get("FEETECH_TOOL_ROOT", "/Users/almond/feetech-servo-tool")
if TOOL_ROOT not in sys.path:
    sys.path.insert(0, TOOL_ROOT)

import serial.tools.list_ports  # type: ignore
import servo  # type: ignore
from servobus import ServoBus  # type: ignore


def output(payload):
    print(json.dumps(payload, ensure_ascii=False))


def open_bus(port, baud_rate, timeout_ms, series):
    bus = ServoBus()
    if not bus.open(port):
        raise RuntimeError(f"{port} 포트를 열지 못했습니다.")
    bus.set_baudrate(int(baud_rate))
    bus.set_timeout(int(timeout_ms))
    bus.set_end(1 if series == "SCS" else 0)
    return bus


def list_ports():
    ports = []
    for info in serial.tools.list_ports.comports():
        ports.append(
            {
                "device": info.device,
                "name": info.name,
                "description": info.description,
                "hwid": info.hwid,
            }
        )
    output({"ok": True, "ports": ports})


def scan(port, baud_rate, timeout_ms, start_id, end_id):
    found = []
    seen = set()

    for end in (0, 1):
        bus = ServoBus()
        if not bus.open(port):
            raise RuntimeError(f"{port} 포트를 열지 못했습니다.")
        bus.set_baudrate(int(baud_rate))
        bus.set_timeout(int(timeout_ms))
        bus.set_end(end)

        for servo_id in range(int(start_id), int(end_id) + 1):
            model_number = bus.read_model_number(servo_id)
            if not model_number or servo_id in seen:
                continue
            model_name = servo.getModelType(model_number)
            series = servo.getModelSeries(model_name)
            found.append(
                {
                    "id": servo_id,
                    "modelNumber": model_number,
                    "modelName": model_name,
                    "series": series,
                }
            )
            seen.add(servo_id)

        bus.close()

    output({"ok": True, "port": port, "baudRate": int(baud_rate), "servos": found})


def read_status(port, baud_rate, timeout_ms, servo_id, series):
    bus = open_bus(port, baud_rate, timeout_ms, series)
    proto = servo.Servo(bus)
    servo_id = int(servo_id)

    payload = {
        "ok": True,
        "port": port,
        "baudRate": int(baud_rate),
        "servoId": servo_id,
        "series": series,
        "position": proto.read_position(servo_id),
        "load": proto.read_load(servo_id),
        "speed": proto.read_speed(servo_id),
        "current": proto.read_current(servo_id),
        "temperature": proto.read_temperature(servo_id),
        "voltageRaw": proto.read_voltage(servo_id),
        "voltage": round((proto.read_voltage(servo_id) or 0) * 0.1, 2),
        "moving": proto.read_move(servo_id),
        "goal": proto.read_goal(servo_id),
    }

    bus.close()
    output(payload)


def move(port, baud_rate, timeout_ms, servo_id, series, goal, speed, acc, torque):
    bus = open_bus(port, baud_rate, timeout_ms, series)
    proto = servo.Servo(bus)
    servo_id = int(servo_id)

    if torque:
        proto.enable_torque(servo_id, 1)

    result = proto.write_pos_ex(int(servo_id), int(goal), int(speed), int(acc))
    bus.close()
    output(
        {
            "ok": result == 0,
            "result": result,
            "servoId": servo_id,
            "goal": int(goal),
            "speed": int(speed),
            "acc": int(acc),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("list-ports")

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--port", required=True)
    scan_parser.add_argument("--baud-rate", default="1000000")
    scan_parser.add_argument("--timeout-ms", default="50")
    scan_parser.add_argument("--start-id", default="0")
    scan_parser.add_argument("--end-id", default="12")

    status_parser = subparsers.add_parser("read-status")
    status_parser.add_argument("--port", required=True)
    status_parser.add_argument("--baud-rate", default="1000000")
    status_parser.add_argument("--timeout-ms", default="50")
    status_parser.add_argument("--servo-id", required=True)
    status_parser.add_argument("--series", required=True)

    move_parser = subparsers.add_parser("move")
    move_parser.add_argument("--port", required=True)
    move_parser.add_argument("--baud-rate", default="1000000")
    move_parser.add_argument("--timeout-ms", default="50")
    move_parser.add_argument("--servo-id", required=True)
    move_parser.add_argument("--series", required=True)
    move_parser.add_argument("--goal", required=True)
    move_parser.add_argument("--speed", default="300")
    move_parser.add_argument("--acc", default="10")
    move_parser.add_argument("--torque", default="1")

    args = parser.parse_args()

    try:
      if args.action == "list-ports":
          list_ports()
      elif args.action == "scan":
          scan(args.port, args.baud_rate, args.timeout_ms, args.start_id, args.end_id)
      elif args.action == "read-status":
          read_status(args.port, args.baud_rate, args.timeout_ms, args.servo_id, args.series)
      elif args.action == "move":
          move(
              args.port,
              args.baud_rate,
              args.timeout_ms,
              args.servo_id,
              args.series,
              args.goal,
              args.speed,
              args.acc,
              args.torque != "0",
          )
    except Exception as error:
        output({"ok": False, "message": str(error)})
        sys.exit(1)


if __name__ == "__main__":
    main()
