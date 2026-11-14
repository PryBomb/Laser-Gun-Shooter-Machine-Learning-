# Laser Gun Shooter (Machine Learning)

A camera-based laser gun game controlled by hand gestures. A webcam tracks your hands with **MediaPipe Hands**, and an **ESP32** moves a servo-mounted laser and switches target LEDs over **WebSocket (Wi-Fi)**.

- **Right hand** aims and shoots.
- **Left hand** reloads.
- Three **virtual LEDs** on screen are synced with three **real LEDs**. When you hit a target, the virtual one pops and the real one turns off.

> Final activity: Machine Learning-Based Hand Gesture Recognition for Embedded Device Control Using a Camera.

---

## How it works

```
Webcam -> MediaPipe Hands -> Gesture logic (Python) -> WebSocket over Wi-Fi -> ESP32 -> Servo + Laser + 3 LEDs
```

1. **Camera** captures live video (OpenCV).
2. **MediaPipe Hands** is a pretrained ML model that detects the palm and predicts 21 hand landmarks per hand.
3. **Gesture logic** turns landmarks into actions:
   - Index fingertip (landmark 8) x-position maps to the servo angle (0 to 180).
   - Thumb tip (4) touching the middle finger joint (10) is the shoot gesture. The distance is scaled by hand size, so it works at any distance from the camera.
   - Left-hand fist (all four fingertips closer to the wrist than their middle joints) held for about 0.6 s reloads.
4. **WebSocket** sends short text commands to the ESP32.
5. **ESP32** moves the servo, flashes the laser and switches the target LEDs.

## Controls

| Hand | Gesture | Action |
|---|---|---|
| Right | Move index finger left or right | Aim the laser |
| Right | Touch thumb tip to middle finger | Shoot (uses 1 bullet) |
| Left | Close fist and hold about 0.6 s | Reload (back to 6 bullets) |
| Keyboard | `r` | Reset targets and ammo |
| Keyboard | `q` | Quit |

## Commands sent to the ESP32

| Command | Action |
|---|---|
| `A<0-180>` | Move the servo to that angle |
| `F` | Fire (laser flashes) |
| `K1` / `K2` / `K3` | Turn off target LED 1, 2 or 3 |
| `R` | Reset (all LEDs on) |

## Hardware

- ESP32 development board
- SG90 (or similar) servo motor
- Laser module (low power, Class 1 to 2, under 5 mW)
- 3 LEDs with 220 ohm resistors
- Laptop with a webcam
- USB data cable, jumper wires, optional external 5V supply

### Wiring

| Part | ESP32 pin |
|---|---|
| Servo signal | GPIO 18 |
| Servo power | 5V / VIN |
| Laser signal | GPIO 19 |
| Laser power | 5V / VIN (through a transistor if the module draws more than 20 mA) |
| LED 1 (via 220 ohm) | GPIO 21 |
| LED 2 (via 220 ohm) | GPIO 22 |
| LED 3 (via 220 ohm) | GPIO 23 |
| All grounds | GND (shared) |

Mount the laser on the servo horn so it sweeps horizontally, and place the three real LEDs within its sweep.

If the servo jitters or the ESP32 resets when firing, use a separate 5V supply for the servo and laser and share its ground with the ESP32.

## Project structure

```
.
├── laser_gun.py                 # Python program (camera, ML, gesture logic, WebSocket client)
├── esp32_laser_gun/
│   └── esp32_laser_gun.ino      # ESP32 firmware (WebSocket server)
├── .gitignore
└── README.md
```

## Setup

### 1. ESP32 (Arduino IDE)

1. Install the **ESP32** board package (Boards Manager, "esp32" by Espressif).
2. Install the libraries (Library Manager):
   - **WebSockets** by Markus Sattler
   - **ESP32Servo** by Kevin Harrington
3. Open `esp32_laser_gun/esp32_laser_gun.ino`, select **ESP32 Dev Module** and your COM port, then upload.
4. Open the Serial Monitor at **115200 baud** and press **EN**. You should see:
   ```
   Join Wi-Fi: LaserGun
   ESP32 IP: 192.168.4.1
   ```

By default the ESP32 creates its own Wi-Fi network (`USE_AP_MODE = true`), so no router is needed. To join an existing **2.4 GHz** network instead, set `USE_AP_MODE = false` and fill in `WIFI_SSID` and `WIFI_PASS`.

### 2. Python (Windows, VS Code)

Python **3.10 to 3.12** is required. MediaPipe does not support newer versions yet.

```
python -m venv C:\laser_venv
C:\laser_venv\Scripts\python.exe -m pip install mediapipe==0.10.14 opencv-python==4.10.0.84 "numpy<2" websocket-client
```

MediaPipe is pinned to **0.10.14** because newer releases removed the `mediapipe.solutions` API used here. A clean virtual environment avoids protobuf conflicts with TensorFlow.

Install the packages while on a network with internet access.

## Run

1. Power the ESP32 and press **EN**.
2. Connect the laptop to the **LaserGun** Wi-Fi (password in the firmware, default `12345678`). Windows may say "No internet", which is normal.
3. Check the link with `ping 192.168.4.1`.
4. Start the program:
   ```
   C:\laser_venv\Scripts\python.exe laser_gun.py
   ```
5. The camera window should show **ESP32: connected** in green.

## Calibration

Edit the settings at the top of `laser_gun.py`:

| Setting | Purpose |
|---|---|
| `ESP32_IP` | IP of the ESP32 (`192.168.4.1` in AP mode) |
| `LED_ANGLES` | Servo angle of real LED 1, 2, 3 |
| `TOLERANCE` | Degrees counted as a hit |
| `INVERT` | Set `True` if the laser moves opposite to your hand |
| `SWAP_HANDS` | Set `True` if left and right are detected backwards |
| `MAX_AMMO` | Bullets per magazine |
| `RELOAD_HOLD` | Seconds the fist must be held to reload |

To calibrate: aim until the laser dot lands on a real LED, read the `Angle:` value on screen, and put it into `LED_ANGLES`. Repeat for each LED.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ESP32: offline` | Laptop must be on the **LaserGun** network. Check `ipconfig` shows `192.168.4.x`, then `ping 192.168.4.1` |
| `module 'mediapipe' has no attribute 'solutions'` | Install `mediapipe==0.10.14` in a clean virtual environment |
| protobuf or TensorFlow import errors | Use the clean venv above, without TensorFlow |
| Laggy video | Close heavy apps, use good lighting, keep the whole hand in view. Network sending runs in a background thread so the camera never waits |
| Camera does not open | Change `cv2.VideoCapture(0, cv2.CAP_DSHOW)` to `cv2.VideoCapture(0)` or try index `1` |
| Servo jitters or ESP32 resets | Use an external 5V supply and a shared ground |
| ESP32 will not join a hotspot | The ESP32 only supports **2.4 GHz** (on iPhone enable "Maximize Compatibility") |

## Limitations and fixes

- **Landmark jitter:** fixed with exponential smoothing of the aim angle.
- **Repeated firing while holding the pose:** fixed by firing only on the rising edge of the gesture.
- **Gesture threshold changes with distance:** fixed by scaling the threshold to hand size.
- **Servo flooding:** aim messages are throttled and only the newest one is sent.
- **Camera freezing on network problems:** all WebSocket work runs in background threads.
- **Left and right hands:** the video is mirrored before processing, so MediaPipe's labels match the real hands.

## Safety

Use only a low-power laser (Class 1 to 2, under 5 mW). Never point it at eyes or at reflective surfaces. Keep the demo area clear of people.

## Tech stack

Python, OpenCV, MediaPipe Hands, WebSocket (`websocket-client`), ESP32 (Arduino framework), `WebSockets` and `ESP32Servo` libraries.
