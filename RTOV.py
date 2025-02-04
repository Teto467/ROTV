import cv2
import torch
import torchvision
from torchvision import transforms as T
import numpy as np
import random

# COCOデータセットのクラスID → クラス名の辞書
COCO_INSTANCE_CATEGORY_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

# GPUが使用可能ならGPUを設定
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 学習済みFaster R-CNNモデルの読み込み（weightsパラメーターを使用）
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.to(device)
model.eval()

# 入力画像前処理のための変換
transform = T.Compose([T.ToTensor()])

# カメラ映像の取得
cap = cv2.VideoCapture(0)

# 信頼度の閾値
score_threshold = 0.8

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # OpenCVはBGR形式のためRGBに変換
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb_frame).to(device)

    with torch.no_grad():
        predictions = model([input_tensor])

    # 検出結果の取得
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_boxes  = predictions[0]['boxes'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()

    cell_phone_detected = False  # cell phoneを検知したかどうかのフラグ

    for score, box, label in zip(pred_scores, pred_boxes, pred_labels):
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        label_text = COCO_INSTANCE_CATEGORY_NAMES.get(label, "Unknown")
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        
        # 検出されたクラスが cell phone ならフラグを立てる
        if label_text == "cell phone":
            cell_phone_detected = True

    # cell phoneが検知されていたら5〜10のランダムな値をターミナルに出力（各フレームにつき一度）
    if cell_phone_detected:
        random_value = random.randint(5, 10)
        print(f"cell phone detected: {random_value}")

    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
