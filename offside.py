import cv2
import torch
from ultralytics import YOLO
import numpy as np

# --- Configuración ---
model = YOLO("runs/detect/train_gpu_rtx30505/weights/best.pt")
video_path = "partido.mp4"
pause_on_offside = True

COLORS = {
    "ball": (0, 255, 255),
    "player": (0, 255, 0),
    "goalkeeper": (255, 0, 0),
    "referee": (128, 128, 128)
}

# --- Video ---
cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter("salida_offside.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                      int(cap.get(cv2.CAP_PROP_FPS)),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

def get_bottom_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)

def detect_team_color(frame, box):
    """Detecta si el jugador es rojo o verde"""
    x1, y1, x2, y2 = map(int, box)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "unknown"
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Máscara roja
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv_crop, lower_red1, upper_red1) + cv2.inRange(hsv_crop, lower_red2, upper_red2)
    # Máscara verde
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv_crop, lower_green, upper_green)

    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)

    if red_pixels > green_pixels:
        return "red"
    elif green_pixels > red_pixels:
        return "green"
    else:
        return "unknown"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.5, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    players = []
    goalkeepers = []
    ball = None

    for *xyxy, conf, cls_id in detections:
        cls = model.names[int(cls_id)]
        color = COLORS.get(cls, (255, 255, 255))
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{cls} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if cls == "player":
            team_color = detect_team_color(frame, (x1, y1, x2, y2))
            players.append({"pos": get_bottom_center((x1, y1, x2, y2)), "team": team_color})
        elif cls == "goalkeeper":
            goalkeepers.append(get_bottom_center((x1, y1, x2, y2)))
        elif cls == "ball":
            ball = get_bottom_center((x1, y1, x2, y2))

    offside_detected = False
    if ball and players:
        # Elegir equipo atacante (puede ajustarse dinámicamente si quieres)
        attacking_team = "red"
        defenders = [p for p in players if p["team"] != attacking_team]
        defenders_pos = [p["pos"] for p in defenders] + goalkeepers

        if len(defenders_pos) >= 2:
            offside_line_x = sorted(defenders_pos, key=lambda p: p[0])[-2][0]
            cv2.line(frame, (offside_line_x, 0), (offside_line_x, frame.shape[0]), (0, 0, 255), 2)
            cv2.putText(frame, "Offside line", (offside_line_x + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            for p in players:
                if p["team"] == attacking_team and p["pos"][0] > offside_line_x:
                    offside_detected = True
                    break
            if ball[0] > offside_line_x:
                offside_detected = True

            if offside_detected:
                cv2.putText(frame, "⚠ FUERA DE JUEGO ⚠", (50, 80),
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Deteccion de fuera de juego", frame)
    out.write(frame)

    if offside_detected and pause_on_offside:
        print("\n⚠ Se detectó un posible FUERA DE JUEGO — presiona cualquier tecla para continuar...")
        while True:
            key = cv2.waitKey(0)
            if key != -1:
                break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
