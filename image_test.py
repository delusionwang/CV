import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# 配置
QUERY_DIR = "./image_retrieval/query"
FEATURE_PATH = "./result1/base_features.npy"  # 训练阶段保存的特征
PATHS_PATH = "./result1/base_paths.npy"       # 训练阶段保存的图片路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LANDMARK_LIST = ["fhy", "jx", "kx", "mh", "nm", "sjz", "sy", "tsg", "ty", "yf", "yk", "zx"]
K_LIST = [20, 40, 60]  # 计算精度的K值
SAVE_DIR = "result1"

# 模型与预处理
model = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 工具函数
def get_landmark_from_filename(file_path):
    """从文件名提取地点编码"""
    filename = os.path.basename(file_path)
    landmark = filename.split("-")[0]
    # 仅返回指定的12个地点，否则为无效类别
    return landmark if landmark in LANDMARK_LIST else "invalid"

def extract_query_feature(img_path):
    """提取查询图片特征"""
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = feature_extractor(img_tensor).squeeze()
    return feature.cpu().numpy()


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 加载base库特征
    print("加载base库特征...")
    base_features = np.load(FEATURE_PATH)
    base_paths = np.load(PATHS_PATH)
    print(f"加载完成：{len(base_features)} 张图片特征")

    # 加载query图片
    query_paths = []
    for root, _, files in os.walk(QUERY_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', 'webp')):
                query_paths.append(os.path.join(root, file))

    # 过滤有效查询图片
    valid_query = []
    valid_landmarks = []
    for path in query_paths:
        landmark = get_landmark_from_filename(path)
        if landmark != "invalid":
            valid_query.append(path)
            valid_landmarks.append(landmark)

    print(f"有效查询图片：{len(valid_query)} 张")

    # 提取query特征
    print("提取查询图片特征...")
    query_features = [extract_query_feature(p) for p in tqdm(valid_query)]
    query_features = np.array(query_features)

    # 余弦相似度检索
    print("执行图片检索...")
    similarity_matrix = cosine_similarity(query_features, base_features)

    # 计算P@K
    landmark_result = defaultdict(lambda: defaultdict(list))
    overall_result = defaultdict(list)

    for idx in range(len(valid_query)):
        query_landmark = valid_landmarks[idx]
        sim_scores = similarity_matrix[idx]
        # 按相似度降序排序
        sorted_idx = np.argsort(sim_scores)[::-1]
        sorted_base = base_paths[sorted_idx]

        # 计算TopK精度
        for k in K_LIST:
            top_k = sorted_base[:k]
            # 统计匹配同地点的数量
            correct = sum(1 for p in top_k if get_landmark_from_filename(p) == query_landmark)
            p_at_k = correct / k
            overall_result[k].append(p_at_k)
            landmark_result[query_landmark][k].append(p_at_k)

    # 输出整体精度
    print("\n" + "=" * 50)
    print("整体检索精度 P@K")
    print("=" * 50)
    for k in K_LIST:
        print(f"P@{k} = {np.mean(overall_result[k]):.4f}")

    # 输出每个地点的精度 + 生成曲线图
    print("\n" + "=" * 50)
    print("12个地点单独检索精度")
    print("=" * 50)
    for landmark in LANDMARK_LIST:
        if landmark not in landmark_result:
            print(f"{landmark}：无查询图片")
            continue

        k_vals = []
        p_vals = []
        for k in K_LIST:
            avg_p = np.mean(landmark_result[landmark][k])
            k_vals.append(k)
            p_vals.append(avg_p)
            print(f"{landmark} | P@{k} = {avg_p:.4f}")

        # 绘制并保存曲线
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.figure()
        plt.plot(k_vals, p_vals, marker='o', linewidth=2, color='#2E86AB')
        plt.title(f"检索精度 - {landmark}")
        plt.xlabel("K值")
        plt.ylabel("精确率(Precision)")
        plt.xticks(K_LIST)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(SAVE_DIR, f"{landmark}_precision.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n测试完成！所有地点曲线已保存")
    print(f"曲线文件：{[f'{lm}_precision.png' for lm in LANDMARK_LIST]}")