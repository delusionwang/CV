import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# 配置参数
DATA_DIR = "./object_detection/data"
OUTPUT_DIR = "./result2"
IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp"]
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化EasyOCR文本检测器
reader = easyocr.Reader(['ch_sim','en'])

# 读取数据集与JSON标注
def load_dataset(data_dir):
    data_pairs = []
    all_files = os.listdir(data_dir)
    img_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in IMAGE_EXT)]
    for img_file in img_files:
        img_path = os.path.join(data_dir, img_file)
        json_file = os.path.splitext(img_file)[0] + ".json"
        json_path = os.path.join(data_dir, json_file)
        if os.path.exists(json_path):
            data_pairs.append((img_path, json_path))
    print(f"Loaded {len(data_pairs)} image-annotation pairs from {data_dir}")
    return data_pairs

def parse_json_annotation(json_path):
    """解析LabelMe格式JSON标注，提取文字外接矩形"""
    with open(json_path, "r", encoding="utf-8") as f:
        anno = json.load(f)
    text_boxes = []
    if "shapes" in anno:
        for shape in anno["shapes"]:
            if shape.get("label", "").lower() == "text":
                points = np.array(shape["points"], dtype=np.int32)
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                text_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    return text_boxes

# EasyOCR文本检测
def detect_text_easyocr(image):
    """检测文本，返回 (xmin, ymin, xmax, ymax) 格式框"""
    temp_img = image.copy()
    result = reader.readtext(temp_img, detail=1)
    det_boxes = []
    for res in result:
        # 提取四点坐标，转为外接矩形
        points = np.array(res[0], dtype=np.int32)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        det_boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    return det_boxes

# 可视化
def visualize_detection(image, gt_boxes, det_boxes, save_path=None):
    vis_img = image.copy()
    # 绿色：真实标注框
    for (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 红色：模型检测框
    for (x1, y1, x2, y2) in det_boxes:
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if save_path:
        cv2.imwrite(save_path, vis_img)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Green: GT(标注), Red: Detection(检测)")
    plt.show()
    return vis_img

# IOU计算 + 评估指标
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - y1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

def evaluate_detection(gt_boxes, det_boxes, iou_thresh=0.5):
    tp, fp, fn = 0, 0, len(gt_boxes)
    matched_gt = set()
    for det_box in det_boxes:
        best_iou, best_gt_idx = 0, -1
        for idx, gt_box in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            current_iou = iou(det_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = idx
        if best_iou >= iou_thresh and best_gt_idx != -1:
            tp += 1
            fn -= 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    return tp, fp, fn

def manual_check_ui(gt_boxes, det_boxes):
    tp, fp, fn = evaluate_detection(gt_boxes, det_boxes)
    print(f"单图结果：TP={tp} 正确检测 | FP={fp} 误检 | FN={fn} 漏检")
    return tp, fp, fn

# 主流程
def main():
    data_pairs = load_dataset(DATA_DIR)
    if not data_pairs:
        print("未找到图片-JSON标注对，请检查路径！")
        return

    import random
    sample_pairs = random.sample(data_pairs, min(24, len(data_pairs)))
    print(f"开始测评 {len(sample_pairs)} 张样本...")

    total_tp, total_fp, total_fn = 0, 0, 0
    for idx, (img_path, json_path) in enumerate(sample_pairs):
        print(f"\n===== 第{idx+1}张：{os.path.basename(img_path)} =====")
        image = cv2.imread(img_path)
        if image is None:
            print("图片读取失败，跳过")
            continue
        # 读取标注 + 执行检测
        gt_boxes = parse_json_annotation(json_path)
        det_boxes = detect_text_easyocr(image)
        # 可视化保存
        save_path = os.path.join(OUTPUT_DIR, f"result_{idx+1}.jpg")
        visualize_detection(image, gt_boxes, det_boxes, save_path)
        # 指标统计
        tp, fp, fn = manual_check_ui(gt_boxes, det_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # 整体汇总
    print("\n========== 实验总结果 ==========")
    print(f"总计正确检测(TP): {total_tp}")
    print(f"总计误检(FP): {total_fp}")
    print(f"总计漏检(FN): {total_fn}")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    print(f"精确率(Precision): {precision:.4f}")
    print(f"召回率(Recall): {recall:.4f}")

if __name__ == "__main__":
    main()