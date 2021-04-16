from efficientnet_pytorch import EfficientNet
import torch
import torch.nn.functional as F
import numpy as np


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

    def predict(self, dataloader, device):
        outputs = [[] for _ in range(9)]
        for i, img in enumerate(dataloader):
            with torch.no_grad():
                feats = self.extract_features(img.to(device))
            for f_idx, feat in enumerate(feats):
                outputs[f_idx].append(feat)

        X_data = []
        for t_idx, output in enumerate(outputs):
            X_data.append(torch.cat(output, 0).squeeze().cpu().detach().numpy())

        X = np.concatenate(X_data, 1)

        return X

