# Oshikatsu_
動画からトラッキングしたアイドルの顔検出/識別を行い、プロットした動画を作成  
![proposed](https://github.com/user-attachments/assets/f9139ddf-8f7a-4ba1-8de4-4f1518be09fc)


## 実行
### 環境構築
必要なものを以下URLからダウンロードしてください
<details><summary>ダウンロード</summary>

・[YOLOv8 weights](https://github.com/ultralytics/ultralytics)  
・[Tracking Model](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)

</details>

```
conda env create -f env.yaml
```

### 推論
設定の項目を埋めて実行
<details><summary>設定項目</summary>

・`MOVNAME`             ：movie name (data/mp4/MOVNAME.mp4)  
・`YOLO_WEIGHTS`        ：YOLOv8 weights (yolov8X.pt)  
・`FACENET_WEIGHTS`     ：FaceNet weights (facenet_X.pt)  
・`TRACKING_YAML`       ：Tracking Model (botsort or bytetrack.yaml)  
・`MEMBER_LIST`         ：メンバーのリスト  
・`MEMBER_ENJP_LIST`    ：メンバーの名前の日本語/英語データ  
・`FONT_PATH`           ：使用するフォント  
・`DEVICE`              ：cuda or mps or cpu  

</details>

セグメンテーション無しver
```
bash run/run.sh
```
セグメンテーション有りver
```
bash run/run_seg.sh
```


## サンプル
元動画：[【Dance Practice】AKB48 「アイドルなんかじゃなかったら」 Moving ver.](https://www.youtube.com/watch?v=rslcM7e-7WI)
### 全てのメンバーを表示
<div><video controls src="https://github.com/user-attachments/assets/b979c8d6-d863-4f08-a82a-c2f4fc81e27a" muted="false"></video></div>  

### 特定のメンバーを表示
<div><video controls src="https://github.com/user-attachments/assets/814d5e7b-c836-4717-97e4-ab614efc5699" muted="false"></video></div>
