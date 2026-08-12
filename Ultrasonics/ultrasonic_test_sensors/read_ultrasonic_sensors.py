import serial
import time

PORT = "/dev/ttyACM0"
BAUD = 115200

# Sensor positions
LABELS = [
    "Right Front",
    "Left Front ",
    "Left Side  ",
    "Right Side ",
    "Rear       ",
]

ser = serial.Serial(PORT, BAUD, timeout=1)

# Arduino resets when serial opens
time.sleep(5)

ser.reset_input_buffer()

print("Reading distances... Ctrl+C to stop\n")

try:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode(errors="ignore").strip()

            if not line:
                continue

            try:
                values = list(map(float, line.split(",")))
                if len(values) != 5:
                    print("Skipped:", line)
                    continue

                readings = "   ".join(
                    f"{label}: {v:5.1f} cm" if v >= 0 else f"{label}:   --- cm"
                    for label, v in zip(LABELS, values)
                )
                print(readings)

            except ValueError:
                print("Skipped:", line)

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    ser.close()
