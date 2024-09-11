# 0.設定
MOVNAME="62thSingleMV_1k"                                           # movie name (data/mp4/MOVNAME.mp4)
YOLO_WEIGHTS="weights/yolov8x-seg.pt"                               # YOLOv8 weights (yolov8X.pt)
FACENET_WEIGHTS="weights/facenet_62thsingle_mtcnn.pt"               # FaceNet weights (facenet_X.pt)
TRACKING_YAML="data/yaml/bytetrack.yaml"                            # Tracking Model (botsort/bytetrack.yaml)
MEMBER_LIST="face_identification/member_list/62thsingle.txt"        # member list
MEMBER_ENJP_LIST="face_identification/member_list/member.csv"       # member en/jp list
FONT="data/font/NotoSansJP-Black.ttf"                               # Font 
DEVICE="cuda:0"                                                     # cuda or mps or cpu

# 実行　bash run/run_seg.sh
# 1.トラッキング
python seg_tracking/mov2ids.py --mov $MOVNAME --yolo-weights $YOLO_WEIGHTS --tracking-yaml $TRACKING_YAML
# 2.顔識別
python seg_tracking/id2name.py --mov $MOVNAME --member-list $MEMBER_LIST --facenet-weights $FACENET_WEIGHTS --device $DEVICE
# 3.動画作成
python seg_tracking/make_mov.py --mov $MOVNAME --member-enjp-list $MEMBER_ENJP_LIST --font $FONT