# Oshikatsu_
・トラッキングは[YOLOv8](https://github.com/ultralytics/ultralytics)を使用  
・顔の切り出しは[YOLOv8-face](https://github.com/akanametov/yolo-face)を使用  
・顔識別は[facenet-pytorch](https://github.com/timesler/facenet-pytorch)を使用
![proposed](https://github.com/user-attachments/assets/f9139ddf-8f7a-4ba1-8de4-4f1518be09fc)

## 実行
必要なものは以下URLからダウンロードしてください
### 環境構築
```
conda env create -f env.yaml
```
### URL
・[YOLOv8 weights](https://github.com/ultralytics/ultralytics)  
・[YOLO face weights](https://github.com/akanametov/yolo-face)  
・[Tracking Model](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)
### コマンド
設定の項目を埋めて実行
```
bash run.sh
```


## サンプル
元動画：[【Dance Practice】AKB48 「アイドルなんかじゃなかったら」 Moving ver.](https://www.youtube.com/watch?v=rslcM7e-7WI)
<div><video controls src="https://github.com/user-attachments/assets/b979c8d6-d863-4f08-a82a-c2f4fc81e27a" muted="false"></video></div>
