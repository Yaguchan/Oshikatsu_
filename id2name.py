import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
from collections import Counter
from torchvision import transforms
from face_identification.model import FaceNet


# python id2name.py --mov 62thSingleDance_178_182_1k --member-list face_identification/member_list/62thsingle.txt --yolo-face-weights weights/yolov8n-face.pt --facenet-weights weights/facenet_62thsingle.pt --device cuda
THRES = 0.99


transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def main(args):
    
    # path
    video_path = f'data/mp4/{args.mov}.mp4'
    text_path = f'output/labels/{args.mov}/t_id_xyxy.txt'
    os.makedirs('/'.join(text_path.split('/')[:-1]), exist_ok=True)
    
    # video data
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # class
    with open(args.member_list, 'r', encoding='utf-8') as f:
        classes = f.read().splitlines()
    classes.append('nan')
    num_classes = len(classes)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # model
    device = torch.device(args.device)
    yolo = YOLO(args.yolo_face_weights)
    facenet = FaceNet(num_classes-1, device)
    facenet.load_state_dict(torch.load(args.facenet_weights, map_location=device))
    facenet.to(device)
    facenet.eval()
    
    # tracking data
    ts = [[] for _ in range(num_frame+1)]
    idxs = []
    with open(text_path, 'r', encoding='utf-8') as f:
        track_datas = [[int(value) for value in line.split()] for line in f]
    for track_data in track_datas:
        t, idx, x1, y1, x2, y2 = track_data
        ts[t].append([idx, x1, y1, x2, y2])
        idxs.append(idx)
    idxs = sorted(list(set(idxs)))
    
    # 顔識別
    print('id -> name')
    frame_count = 0
    pbar = tqdm(total=num_frame)
    frame_size = max(frame_width, frame_height)
    outputs = [[] for _ in range(idxs[-1]+1)]
    all_probs = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # その時間のトラッキングの結果を yolo-face -> facenet
        for idx_xyxy in ts[frame_count]:
            idx, h_x1, h_y1, h_x2, h_y2 = idx_xyxy
            human_image = frame[h_y1:h_y2, h_x1:h_x2]
            boxes = yolo.predict(human_image, verbose=False)[0].boxes
            xyxy_list = boxes.xyxy.tolist()
            conf_list = boxes.conf.tolist()
            xyxy_list2 = []
            conf_list2 = []
            # 足切り①（画像サイズ, yolo-confidence）
            for f_xyxy_, f_conf_ in zip(xyxy_list, conf_list):
                f_x1_, f_y1_, f_x2_, f_y2_ = f_xyxy_
                if min(f_x2_-f_x1_, f_y2_-f_y1_) < 30 or f_conf_ < 0.5: continue
                xyxy_list2.append(f_xyxy_)
                conf_list2.append(f_conf_)
            if len(xyxy_list2) == 0: continue
            l = 10 ** 10
            for (f_x1_, f_y1_, f_x2_, f_y2_), f_conf_ in zip(xyxy_list2, conf_list2):
                if f_conf_ < 0.5: continue
                l_ = (f_x1_ + f_x2_ + h_x1 - h_x2) ** 2 + (f_y1_ + f_y2_) ** 2
                if l_ < l:
                    l = l_
                    f_x1, f_y1, f_x2, f_y2 = f_x1_, f_y1_, f_x2_, f_y2_
                    f_conf = f_conf_
            human_image = human_image[..., ::-1]
            face_image = Image.fromarray(human_image[int(f_y1):int(f_y2), int(f_x1):int(f_x2)])
            face_image = transform(face_image)
            with torch.no_grad():
                prob, pred_idx = facenet.inference(face_image.unsqueeze(0))
                prob, pred_idx = prob.item(), pred_idx.item()
            # 足切り②（FaceNet出力確率）
            if prob < THRES: continue
            outputs[idx].append([pred_idx, prob])
            all_probs.append(prob)
        frame_count += 1
        pbar.update(1)
    cap.release()
    
    # FaceNet結果 -> 名前
    names = []
    for idx in idxs:
        if len(outputs[idx]) > 0:
            sorted_output = sorted(outputs[idx], key=lambda x: x[1])
            list_class = [output[0] for output in sorted_output]
            list_prob = [output[1] for output in sorted_output]
            counter = Counter(list_class)
            most_common_class, count = counter.most_common(1)[0]
            if count/len(list_class) >= 0.5:
                names.append(idx_to_class[most_common_class])
            else:
                names.append('nan')
        else:
            names.append('nan')
    
    # output
    output_path = os.path.join('/'.join(text_path.split('/')[:-1]), 'id_name.txt')
    if os.path.exists(output_path): os.remove(output_path)
    with open(output_path, 'a') as f:
        for name in names:
            f.write(f'{name}\n')           
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mov', type=str, help='動画(mp4)のパスを指定してください', required=True)
    parser.add_argument('--member-list', type=str, help='メンバーリスト(.txt)のパスを指定してください', required=True)
    parser.add_argument('--yolo-face-weights', type=str, help='YOLO(顔検出)の重みを指定してください', required=True)
    parser.add_argument('--facenet-weights', type=str, help='FaceNetの重みを指定してください', required=True)
    parser.add_argument('--device', type=str, help='cuda device or cpu', required=True)
    args = parser.parse_args()
    main(args)