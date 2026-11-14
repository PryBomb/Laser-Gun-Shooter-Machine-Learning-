"""
Laser Gun (3 targets, ammo + reload) - low-lag version
  RIGHT hand = gun    : index finger aims, thumb-to-middle-finger pinch shoots
  LEFT hand  = reload : close your fist and hold it briefly

Pipeline: Camera -> MediaPipe Hands -> gesture logic -> WebSocket -> ESP32
Keys: q = quit, r = reset all targets and ammo
"""
import cv2, math, time, threading, queue, socket
import mediapipe as mp
from websocket import create_connection

ESP32_IP = "192.168.4.1"           # ESP32's own network (LaserGun)
LED_ANGLES = [45, 90, 135]         # servo angle of real LED 1, 2, 3 (calibrate each!)
TOLERANCE = 8                      # degrees counted as a hit
INVERT = False                     # flip if the laser moves the wrong way
SWAP_HANDS = False                 # set True if left/right are detected backwards
MAX_AMMO = 6                       # bullets per magazine
RELOAD_HOLD = 0.6                  # seconds the left fist must be held to reload
SEND_INTERVAL = 0.03               # max ~33 aim messages per second
NUM_LEDS = len(LED_ANGLES)

# ---------- WebSocket (all network work happens in background threads) ----------
ws = None
connecting = False
send_q = queue.Queue()

def _connect_worker():
    global ws, connecting
    try:
        sock = create_connection(
            f"ws://{ESP32_IP}:81", timeout=1.5,
            sockopt=((socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),))   # no packet batching
        sock.settimeout(0.5)
        ws = sock
        print("Connected to ESP32")
    except Exception as e:
        ws = None
        print("ESP32 not reachable:", e)
    connecting = False

def try_connect():
    global connecting
    if connecting or ws is not None:
        return
    connecting = True
    threading.Thread(target=_connect_worker, daemon=True).start()

def _sender_loop():
    """Sends queued commands. Only the newest aim (A) message is kept, so aiming never lags."""
    global ws
    while True:
        msgs = [send_q.get()]
        while not send_q.empty():
            msgs.append(send_q.get_nowait())
        last_aim = None
        others = []
        for m in msgs:
            if m.startswith("A"):
                last_aim = m
            else:
                others.append(m)
        out = ([last_aim] if last_aim else []) + others
        for m in out:
            if ws is None:
                break
            try:
                ws.send(m)
            except Exception:
                ws = None

threading.Thread(target=_sender_loop, daemon=True).start()

def send(msg):
    """Never blocks the camera loop: just drops the command into the queue."""
    if ws is not None:
        send_q.put(msg)

# ---------- helpers ----------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def is_fist(lm):
    """Fist = index, middle, ring, pinky tips closer to the wrist than their PIP joints."""
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return all(dist(lm[t], lm[0]) < dist(lm[p], lm[0]) for t, p in zip(tips, pips))

# ---------- camera (small, fast, no buffering) ----------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)     # if the camera fails to open, use cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, model_complexity=0,   # 0 = lightest/fastest model
                       min_detection_confidence=0.7, min_tracking_confidence=0.6)
draw = mp.solutions.drawing_utils

alive = [True] * NUM_LEDS
flash_until = [0.0] * NUM_LEDS
ammo = MAX_AMMO
was_firing = False
fist_start = None
reload_done = False
msg_text, msg_until = "", 0.0
angle_smooth, angle = 90.0, 90
last_sent_angle, last_sent_time = -1, 0
last_retry = 0.0
fps, fps_count, fps_t = 0, 0, time.time()

try_connect()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)     # mirrored, so MediaPipe's Left/Right = your real hands
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    now = time.time()

    if ws is None and now - last_retry > 3:              # non-blocking auto-reconnect
        last_retry = now
        try_connect()

    # --- sort detected hands into right (gun) and left (reload) ---
    right_lm = left_lm = None
    if res.multi_hand_landmarks:
        for hand, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
            label = handed.classification[0].label       # "Left" or "Right"
            if SWAP_HANDS:
                label = "Left" if label == "Right" else "Right"
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            if label == "Right":
                right_lm = hand.landmark
            else:
                left_lm = hand.landmark

    # --- RIGHT HAND: aim + shoot ---
    if right_lm is not None:
        lm = right_lm
        target = lm[8].x * 180                           # index fingertip x -> angle
        if INVERT:
            target = 180 - target
        angle_smooth = 0.7 * angle_smooth + 0.3 * target
        angle = int(angle_smooth)
        if angle != last_sent_angle and now - last_sent_time > SEND_INTERVAL:
            send(f"A{angle}")
            last_sent_angle, last_sent_time = angle, now

        hand_size = dist(lm[0], lm[9])
        firing = dist(lm[4], lm[10]) < 0.35 * hand_size  # thumb tip touches middle PIP
        if firing and not was_firing:                    # rising edge only
            if ammo <= 0:
                msg_text, msg_until = "EMPTY! Reload with LEFT fist", now + 1.0
            else:
                ammo -= 1
                send("F")
                best, best_diff = None, TOLERANCE + 1
                for i in range(NUM_LEDS):
                    diff = abs(angle - LED_ANGLES[i])
                    if alive[i] and diff <= TOLERANCE and diff < best_diff:
                        best, best_diff = i, diff
                if best is not None:
                    alive[best] = False
                    send(f"K{best + 1}")
                    flash_until[best] = now + 0.3
        was_firing = firing

        cx = int(lm[8].x * w)
        cv2.line(frame, (cx, 0), (cx, h), (0, 255, 255), 1)
        cv2.putText(frame, f"Angle: {angle}  {'FIRE' if firing else 'aim'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        was_firing = False

    # --- LEFT HAND: reload (hold a fist) ---
    if left_lm is not None and is_fist(left_lm):
        if fist_start is None:
            fist_start = now
        progress = min((now - fist_start) / RELOAD_HOLD, 1.0)
        if ammo < MAX_AMMO:
            cv2.rectangle(frame, (10, 110), (10 + int(200 * progress), 125),
                          (255, 200, 0), -1)
            cv2.rectangle(frame, (10, 110), (210, 125), (255, 255, 255), 1)
            cv2.putText(frame, "RELOADING...", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        if progress >= 1.0 and ammo < MAX_AMMO and not reload_done:
            ammo = MAX_AMMO
            reload_done = True
            msg_text, msg_until = "RELOADED!", now + 0.8
    else:
        fist_start = None
        reload_done = False

    # --- virtual LEDs ---
    for i in range(NUM_LEDS):
        led_x = int((LED_ANGLES[i] / 180) * w)
        if INVERT:
            led_x = w - led_x
        if alive[i]:
            cv2.circle(frame, (led_x, h // 2), 25, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (led_x - 8, h // 2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif now < flash_until[i]:
            cv2.circle(frame, (led_x, h // 2), 45, (255, 255, 255), 3)

    # --- HUD ---
    remaining = sum(alive)
    hud = "ALL CLEARED! Press R" if remaining == 0 else f"Targets left: {remaining}"
    cv2.putText(frame, hud, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Ammo: {ammo}/{MAX_AMMO}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if ammo else (0, 0, 255), 2)
    for b in range(MAX_AMMO):                            # bullet icons
        color = (0, 220, 255) if b < ammo else (80, 80, 80)
        cv2.circle(frame, (w - 20 - b * 22, 25), 8, color, -1)
    if now < msg_until:
        cv2.putText(frame, msg_text, (w // 2 - 160, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    fps_count += 1
    if now - fps_t >= 1.0:
        fps, fps_count, fps_t = fps_count, 0, now
    if ws:
        status, scol = "ESP32: connected", (0, 255, 0)
    elif connecting:
        status, scol = "ESP32: connecting...", (0, 200, 255)
    else:
        status, scol = "ESP32: offline", (0, 0, 255)
    cv2.putText(frame, status, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, scol, 2)
    cv2.putText(frame, f"FPS: {fps}", (w - 100, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

    cv2.imshow("Laser Gun", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    if k == ord('r'):
        alive = [True] * NUM_LEDS
        ammo = MAX_AMMO
        send("R")

cap.release()
cv2.destroyAllWindows()
if ws:
    ws.close()