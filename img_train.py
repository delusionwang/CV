import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from PIL import Image

# 配置
BASE_DIR = "./image_retrieval/base"
FEATURE_SAVE_PATH = "./result1/base_features.npy"
PATHS_SAVE_PATH = "./result1/base_paths.npy"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载
# 加载预训练ResNet50，删除分类层，仅保留特征提取器
model = models.resnet50(pretrained=True)
# 移除最后一层全连接层，输出2048维特征
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 特征提取函数
def extract_image_feature(img_path):
    """提取单张图片的特征向量"""
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feature = feature_extractor(img_tensor).squeeze()
    return feature.cpu().numpy()


if __name__ == "__main__":
    # 遍历base目录，收集所有图片路径
    image_paths = []
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                image_paths.append(os.path.join(root, file))
    print(f"共找到 {len(image_paths)} 张图片")

    # 批量提取特征
    all_features = []
    all_file_paths = []
    for path in tqdm(image_paths, desc="提取图片特征中..."):
        feat = extract_image_feature(path)
        all_features.append(feat)
        all_file_paths.append(path)

    # 保存特征和路径
    np.save(FEATURE_SAVE_PATH, np.array(all_features))
    np.save(PATHS_SAVE_PATH, np.array(all_file_paths))

    print(f"\n特征提取完成！")
    print(f"特征文件：{FEATURE_SAVE_PATH}")
    print(f"路径文件：{PATHS_SAVE_PATH}")
