import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import random
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import DataLoader
from casnet import Model as casnet
from cnn.resnet import wide_resnet50_2 as wrn50_2
from datasets.mvtec_test import MVTecDataset
from torchvision import transforms as T
from utils.adaptor import *
from utils.metric import *
from utils.visualizer import *
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
import math
from datasets.mvtec_train import SelfSupMVTecDataset

# 每类配置 — 新增类别时在此添加对应条目
# width_bounds_pct: ((最小宽度比例, 最大宽度比例), (最小高度比例, 最大高度比例))
# 小缺陷用 (0.01, 0.05)，大缺陷用 (0.03, 0.4)
WIDTH_BOUNDS_PCT = {
    'with_defect': ((0.01, 0.05), (0.01, 0.05)),
    '01': ((0.01, 0.05), (0.01, 0.05)),
    '02': ((0.01, 0.05), (0.01, 0.05)),
    '03': ((0.01, 0.05), (0.01, 0.05)),
}
# 合成补丁与前景目标的最小重叠比例
MIN_OVERLAP_PCT = {
    'with_defect': 0.25,
    '01': 0.25,
    '02': 0.25,
    '03': 0.25,
}
# 补丁区域内前景像素占比下限（值越低，补丁越容易落在背景区域）
MIN_OBJECT_PCT = {
    'with_defect': 0.7,
    '01': 0.7,
    '02': 0.7,
    '03': 0.7,
}
# 每张图像合成补丁数量
NUM_PATCHES = {
    'with_defect': 1,
    '01': 3,
    '02': 3,
    '03': 1,
}
# (k, x0) — 合成异常标签的 logistic 强度映射参数
INTENSITY_LOGISTIC_PARAMS = {
    'with_defect': (1 / 12, 24),
    '01': (1 / 12, 24),
    '02': (1 / 12, 24),
    '03': (1 / 12, 24),
}
# (背景亮度阈值, 容差) — 背景感知跳过，补丁仅放置于前景区域
# 设为空字典 {} 则禁用（所有区域均可放置补丁）
BACKGROUND = {}

DEFAULT_SELF_SUP = {
    'width_bounds_pct': ((0.01, 0.08), (0.01, 0.08)),
    'intensity_logistic_params': (1 / 6, 15),
    'num_patches': 2,
    'min_object_pct': 0.5,
    'min_overlap_pct': 0.25,
}

#####################################
def parse_args():
    parser = argparse.ArgumentParser('CFA configuration')
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--save_path', type=str, default='./result')
    parser.add_argument("-s", "--setting", type=str, default="Shift-Intensity-923874273")
    parser.add_argument('--Rd', type=bool, default=False)
    parser.add_argument('--size', type=int, choices=[224, 256], default=224)
    parser.add_argument('--gamma_c', type=int, default=1)
    parser.add_argument('--class_name', type=str, default='toothbrush')

    return parser.parse_args()

def weight_init(m):
    if isinstance(m, nn.Conv3d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()

def run():
    seed = 512
    random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    args = parse_args()
    class_names = [args.class_name]

    for class_name in class_names:

        best_img_roc = -1
        print(' ')
        print('%s | newly initialized...' % class_name)
        BACKGROUND = {}

        # load data
        train_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(256),
        ])

        train_dat = SelfSupMVTecDataset(root_path=args.data_path, class_name=class_name, is_train=True,
                                       transform=train_transform)

        train_dat.configure_self_sup(self_sup_args={'gamma_params': (2, 0.05, 0.03), 'resize': False,
                                                    'shift': True, 'same': True, 'mode': 'swap',
                                                    'label_mode': 'binary'})
        train_dat.configure_self_sup(self_sup_args={'skip_background': BACKGROUND.get(class_name)})
        train_dat.configure_self_sup(on=True, self_sup_args={
            'width_bounds_pct': WIDTH_BOUNDS_PCT.get(class_name, DEFAULT_SELF_SUP['width_bounds_pct']),
            'intensity_logistic_params': INTENSITY_LOGISTIC_PARAMS.get(class_name, DEFAULT_SELF_SUP['intensity_logistic_params']),
            'num_patches': NUM_PATCHES.get(class_name, DEFAULT_SELF_SUP['num_patches']),
            'min_object_pct': MIN_OBJECT_PCT.get(class_name, DEFAULT_SELF_SUP['min_object_pct']),
            'min_overlap_pct': MIN_OVERLAP_PCT.get(class_name, DEFAULT_SELF_SUP['min_overlap_pct'])})

        test_dataset = MVTecDataset(dataset_path=args.data_path,
                                    class_name=class_name,
                                    resize=256,
                                    cropsize=args.size,
                                    is_train=False,
                                    wild_ver=args.Rd)

        train_loader = DataLoader(dataset=train_dat,
                                  batch_size=2,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=True, )

        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 pin_memory=True,
                                 drop_last=True)

        model = wrn50_2(pretrained=True, progress=False)

        model = model.to(device)
        A = adaptor(model, train_loader, args.gamma_c, device, class_name).to(device)
        A.apply(weight_init)
        CAS = casnet().to(device)
        CAS.apply(weight_init)
        epochs = 50
        optimizer = optim.Adam([
            {'params': A.parameters(), 'lr': 0.001},
            {'params': CAS.parameters(), 'lr': 0.0001}],
            weight_decay=1e-5,
            amsgrad=True)

        for epoch in tqdm(range(epochs), '%s -->' % (class_name)):
            r'TEST PHASE'
            A.train()
            CAS.train()
            model.eval()
            MSE_loss = torch.nn.MSELoss().to(device)
            tr_entropy_loss_func = torch.nn.CrossEntropyLoss(reduction='sum').to(device)
            epoch_loss_nfc = 0
            epoch_loss_afs = 0
            epoch_loss_pdc = 0
            epoch_loss_seg = 0
            num_batches = 0
            for (x, aug_img, _, _, mask0, mask1) in train_loader:
                if x.shape[0] % 2 == 1:
                    a = x.shape[0] / 2 + 1
                else:
                    a = x.shape[0] / 2
                res = random.sample(range(0, x.shape[0]), int(a))
                mix_img_list = x.clone()
                mix_img_list[res] = aug_img[res]
                mix_mask0_list = torch.zeros_like(mask0)
                mix_mask0_list[res] = mask0[res]
                target = []
                target.extend([0] * x.shape[0])
                target = torch.tensor(target)
                target[res] = int(1)
                optimizer.zero_grad()
                with torch.no_grad():
                    normal_ori_feature = model(x.to(device))  ## normal
                    aug_ori_feature = model(aug_img.to(device))  ## abnormal
                    mix_ori_feature = model(mix_img_list.to(device))  ### Some parts are normal, some parts are abnormal.
                L_NFC, _, normal_score, normal_feature = A(normal_ori_feature, 0, mask1.to(device))
                L_AFS, _, aug_score, aug_feature = A(aug_ori_feature, 1, mask1.to(device))
                _, score_1, mix_score, mix_feature = A(mix_ori_feature, 2, mask1.to(device))
                out = CAS(mix_feature, mix_score)
                L_SEG = MSE_loss(out.to(device), mix_mask0_list.to(device))
                L_PDC = tr_entropy_loss_func(score_1.squeeze(), target.float().to(device))
                loss = L_NFC + L_AFS + L_PDC * 50 + L_SEG * 40
                loss.backward()
                optimizer.step()
                epoch_loss_nfc += L_NFC.item()
                epoch_loss_afs += L_AFS.item()
                epoch_loss_pdc += L_PDC.item()
                epoch_loss_seg += L_SEG.item()
                num_batches += 1

            avg_nfc = epoch_loss_nfc / num_batches
            avg_afs = epoch_loss_afs / num_batches
            avg_pdc = epoch_loss_pdc / num_batches
            avg_seg = epoch_loss_seg / num_batches
            avg_total = avg_nfc + avg_afs + avg_pdc * 50 + avg_seg * 40
            print('[%d/%d] NFC: %.4f | AFS: %.4f | PDC: %.4f | SEG: %.4f | total: %.4f'
                  % (epoch + 1, epochs, avg_nfc, avg_afs, avg_pdc, avg_seg, avg_total))

            # 每轮评估
            gt_list = list()
            heatmaps = None
            A.eval()
            CAS.eval()
            model.eval()
            for x, y in test_loader:
                gt_list.extend(y.cpu().detach().numpy())
                with torch.no_grad():
                    ori_feature = model(x.to(device))
                    score, feature = A(ori_feature, 2, None)
                    score = CAS(feature, score)
                heatmap = score.cpu().detach()
                heatmap = torch.mean(heatmap, dim=1)
                heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps != None else heatmap
            heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
            heatmaps = gaussian_smooth(heatmaps, sigma=4)

            scores = rescale(heatmaps)

            img_roc_auc = cal_img_roc(scores, gt_list)[2]
            best_img_roc = img_roc_auc if img_roc_auc > best_img_roc else best_img_roc

            print('[%d/%d] imgAUROC: %.3f | best: %.3f'
                  % (epoch + 1, epochs, img_roc_auc, best_img_roc))

        print('image ROCAUC: %.3f' % (best_img_roc))

        # save trained models
        save_dir = os.path.join(args.save_path, class_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(A.state_dict(), os.path.join(save_dir, 'adaptor.pth'))
        torch.save(CAS.state_dict(), os.path.join(save_dir, 'cas.pth'))
        print('Models saved to %s' % save_dir)

if __name__ == '__main__':
    run()
