import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
from cnn.resnet import wide_resnet50_2 as wrn50_2
from utils.adaptor import adaptor
from casnet import Model as casnet
from utils.metric import upsample, gaussian_smooth


def parse_args():
    parser = argparse.ArgumentParser('PCSNet inference')
    parser.add_argument('--image_path', type=str, required=True,
                        help='单张图片路径或目录（自动遍历子目录）')
    parser.add_argument('--checkpoint_path', type=str, default='./result',
                        help='模型权重目录，其下应有 {class_name}/adaptor.pth 和 {class_name}/cas.pth')
    parser.add_argument('--class_name', type=str, default='pcba1',
                        help='类别名称，对应训练时使用的 --class_name')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='异常判定阈值')
    parser.add_argument('--device', type=str, default='cpu',
                        help='运行设备 (cpu / cuda)')
    parser.add_argument('--save_heatmap', action='store_true',
                        help='保存热图到同级目录（_heatmap.png 后缀）')
    return parser.parse_args()


def get_image_paths(path):
    exts = ('.png', '.jpg', '.jpeg')
    paths = []
    if os.path.isfile(path):
        paths = [path]
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in sorted(files):
                if f.lower().endswith(exts):
                    paths.append(os.path.join(root, f))
    else:
        raise FileNotFoundError('path not found: %s' % path)
    return paths


def preprocess(image_path):
    transform = T.Compose([
        T.Resize(256, T.InterpolationMode.LANCZOS),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


@torch.no_grad()
def infer_one(image_path, backbone, A, CAS, device):
    x = preprocess(image_path).to(device)
    ori_feature = backbone(x)
    score, feature = A(ori_feature, 2, None)
    out = CAS(feature, score)

    heatmap = out.cpu().detach()
    heatmap = torch.mean(heatmap, dim=1)  # [1, 224, 224]
    heatmap = upsample(heatmap, size=224, mode='bilinear')
    heatmap = gaussian_smooth(heatmap, sigma=4)
    img_score = float(heatmap.max())
    return img_score, heatmap


def save_heatmap(image_path, heatmap, save_path):
    import matplotlib.pyplot as plt
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_img = (heatmap * 255).astype(np.uint8).squeeze()
    plt.imsave(save_path, heatmap_img, cmap='jet', vmin=0, vmax=255)


def main():
    args = parse_args()
    device = torch.device(args.device)

    image_paths = get_image_paths(args.image_path)
    if len(image_paths) == 0:
        print('No images found.')
        return

    # load models
    backbone = wrn50_2(pretrained=True, progress=False).to(device)
    backbone.eval()

    A = adaptor(None, None, 1, device, args.class_name,
                skip_centroid_init=True).to(device)
    A.eval()
    ckpt_dir = os.path.join(args.checkpoint_path, args.class_name)
    A.load_state_dict(torch.load(os.path.join(ckpt_dir, 'adaptor.pth'),
                                 map_location=device))

    CAS = casnet().to(device)
    CAS.eval()
    CAS.load_state_dict(torch.load(os.path.join(ckpt_dir, 'cas.pth'),
                                   map_location=device))

    # per-image inference
    scores = []
    print('%s | threshold=%.2f' % (args.class_name, args.threshold))
    print('-' * 50)
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        score, heatmap = infer_one(img_path, backbone, A, CAS, device)
        scores.append(score)
        pred = 'ANOMALY' if score > args.threshold else 'NORMAL'
        print('%-40s score=%.4f  [%s]' % (fname, score, pred))

        if args.save_heatmap:
            base, _ = os.path.splitext(img_path)
            save_path = base + '_heatmap.png'
            save_heatmap(img_path, heatmap, save_path)

    print('-' * 50)
    print('score range: [%.4f, %.4f]' % (min(scores), max(scores)))


if __name__ == '__main__':
    main()
