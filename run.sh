# 0.設定
MOVNAME="62thSingleDance_178_182_1k"                                # 動画名(data/mp4/MOVNAME.mp4)
YOLO_WEIGHTS="weights/yolov8x.pt"                                   # YOLOv8 weights
YOLO_FACE_WEIGHTS="weights/yolov8n-face.pt"                         # YOLO face weights
FACENET_WEIGHTS="weights/facenet_62thsingle.pt"                     # FaceNet weights
TRACKING_YAML="data/yaml/bytetrack.yaml"                            # TrackingModel(botsort/bytetrack)
MEMBER_LIST="face_identification/member_list/62thsingle.txt"        # member list
MEMBER_ENJP_LIST="face_identification/member_list/member.csv"       # member en/jp list
FONT="data/font/NotoSansJP-Black.ttf"                               # Font 
DEVICE="mps"                                                        # cuda or mps or cpu

# 実行　bash run.sh
# 1.トラッキング
python mov2ids.py --mov $MOVNAME --yolo-weights $YOLO_WEIGHTS --tracking-yaml $TRACKING_YAML
# 2.顔識別
python id2name.py --mov $MOVNAME --member-list $MEMBER_LIST --yolo-face-weights $YOLO_FACE_WEIGHTS --facenet-weights $FACENET_WEIGHTS --device $DEVICE
# 3.動画作成
python make_annotated_mov.py --mov $MOVNAME --member-enjp-list $MEMBER_ENJP_LIST --font $FONT