import argparse
import numpy as np
import os
import math
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

import datasets.mvtec as mvtec


def parse_args():
    parser = argparse.ArgumentParser('MahalanobisAD')
    parser.add_argument("--model_name", type=str, default='efficientnet-b4')
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def main():

    args = parse_args()
    assert args.model_name.startswith('efficientnet-b'), 'only support efficientnet variants, not %s' % args.model_name

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model
    model = EfficientNetModified.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)

    total_roc_auc = []
    total_tn = 0
    total_tp = 0
    total_fn = 0
    total_fp = 0
    for class_name in mvtec.CLASS_NAMES:

        train_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=True)
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        test_dataset = mvtec.MVTecDataset(class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
        if args.model_name == 'efficientnet-b4'
            train_outputs = [[] for _ in range(9)]
            test_outputs = [[] for _ in range(9)]
        elif args.model_name == 'efficientnet-b0'
            train_outputs = [[] for _ in range(7)]
            test_outputs = [[] for _ in range(7)]

        # extract train set features
        train_feat_filepath = os.path.join(args.save_path, 'temp', 'train_%s_%s.pkl' % (class_name, args.model_name))
        if not os.path.exists(train_feat_filepath):
            for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    feats = model.extract_features(x.to(device))
                for f_idx, feat in enumerate(feats):
                    train_outputs[f_idx].append(feat)

            # fitting a multivariate gaussian to features extracted from every level of ImageNet pre-trained model
            for t_idx, train_output in enumerate(train_outputs):
                mean = torch.mean(torch.cat(train_output, 0).squeeze(), dim=0).cpu().detach().numpy()
                # covariance estimation by using the Ledoit. Wolf et al. method
                cov = LedoitWolf().fit(torch.cat(train_output, 0).squeeze().cpu().detach().numpy()).covariance_
                train_outputs[t_idx] = [mean, cov]

            # save extracted feature
            with open(train_feat_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature distribution from: %s' % train_feat_filepath)
            with open(train_feat_filepath, 'rb') as f:
                train_outputs = pickle.load(f)

        gt_list = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            gt_list.extend(y.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                feats = model.extract_features(x.to(device))
            for f_idx, feat in enumerate(feats):
                test_outputs[f_idx].append(feat)
        for t_idx, test_output in enumerate(test_outputs):
            test_outputs[t_idx] = torch.cat(test_output, 0).squeeze().cpu().detach().numpy()

        # calculate Mahalanobis distance for each level of EfficientNet
        dist_list = []
        D = []
        for t_idx, test_output in enumerate(test_outputs):
            mean = train_outputs[t_idx][0]
            D.append(mean.shape[0])
            cov_inv = np.linalg.inv(train_outputs[t_idx][1])
            dist = [mahalanobis(sample, mean, cov_inv) for sample in test_output]
            dist_list.append(np.array(dist))

        # Anomaly score is followed by unweighted summation of the Mahalanobis distances
        dist_list = np.array(dist_list)
        scores = np.sum(dist_list, axis=0)

        #calculate the thresholds
        thresh = []
        for d in D:
            thresh.append(math.sqrt(chi2.ppf(0.95, df = d))) # Setting expected FPR = 0.05

        #threshold the distances
        for i, t in enumerate(thresh):
            dist_list[i] = dist_list[i]>t

        # calculate image-level ROC AUC score
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        # calculate confusion matrix of third last layer features (7th level)
        tn, fp, fn, tp = confusion_matrix(gt_list, dist_list[-3]).ravel()
        total_tp+=tp
        total_tn+=tn
        total_fp+=fp
        total_fn+=fn
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f TN: %.3f FP: %.3f FN: %.3f TP: %.3f' % (class_name, roc_auc, tn, fp, fn, tp))
        plt.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

    print('Average ROCAUC: %.3f TN: %.3f FP: %.3f FN: %.3f TP: %.3f' % (np.mean(total_roc_auc), total_tn, total_fp, total_fn, total_tp))
    plt.title('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.save_path, 'roc_curve_%s.png' % args.model_name), dpi=200)


class EfficientNetModified(EfficientNet):

    def extract_features(self, inputs):
        """ Returns list of the feature at each level of the EfficientNet """

        feat_list = []

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        feat_list.append(F.adaptive_avg_pool2d(x, 1))

        # Blocks
        x_prev = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if (x_prev.shape[1] != x.shape[1] and idx != 0) or idx == (len(self._blocks) - 1):
                feat_list.append(F.adaptive_avg_pool2d(x_prev, 1))
            x_prev = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        feat_list.append(F.adaptive_avg_pool2d(x, 1))

        return feat_list


if __name__ == '__main__':
    main()
