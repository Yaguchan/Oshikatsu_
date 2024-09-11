import os
import cv2
import argparse
from tqdm import tqdm
from ultralytics import YOLO


# python mov2ids.py --mov 62thSingleDance_178_182_1k --yolo-weights weights/yolov8x.pt --tracking-yaml yaml/bytetrack.yaml


def main(args):
    
    # yolo model
    model = YOLO(args.yolo_weights)
    
    # text/video setting
    text_outpath = f'./output/labels/{args.mov}/t_id_xyxy.txt'
    os.makedirs('/'.join(text_outpath.split('/')[:-1]), exist_ok=True)
    if os.path.exists(text_outpath): os.remove(text_outpath)
    video_inpath = f'./data/mp4/{args.mov}.mp4'
    video_outpath = f'./output/track_mov/{args.mov}.mp4'
    os.makedirs('/'.join(video_outpath.split('/')[:-1]), exist_ok=True)
    cap = cv2.VideoCapture(video_inpath)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(video_outpath, fourcc, fps, (frame_width, frame_height))

    # tracking
    print('mov -> ids')
    frame_count = 0
    pbar = tqdm(total=num_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, tracker=args.tracking_yaml, persist=True, classes=0, verbose=False)
        annotated_frame = results[0].plot()
        items = results[0]
        for item in items:
            if item.boxes.id is None: continue
            x1, y1, x2, y2, idx, conf, cls = item.boxes.data.cpu().numpy()[0]
            with open(text_outpath, 'a') as f:
                f.write(f'{frame_count} {int(idx)} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n')
        out.write(annotated_frame)
        frame_count += 1
        pbar.update(1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows() 
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mov', type=str, help='動画(mp4)のパスを指定してください', required=True)
    parser.add_argument('--yolo-weights', type=str, help='YOLOの重みを指定してください', required=True)
    parser.add_argument('--tracking-yaml', type=str, help='トラッキングモデル(.yaml)を指定してください', required=True)
    args = parser.parse_args()
    main(args)