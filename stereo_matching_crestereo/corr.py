import torch
import torch.nn.functional as F
from .utils.utils import bilinear_sampler, coords_grid
import torch.nn as nn

# from spatial_correlation_sampler import SpatialCorrelationSampler

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        # self.conv1 = nn.Conv2d(15, 128, kernel_size=1).to(fmap1.device)
        # self.conv2 = nn.Conv2d(9, 128, kernel_size=1).to(fmap1.device)
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3]).to(
            fmap1.device
        )

    def __call__(self, flow, small_patch=False):
        corr = self.corr(self.fmap1, self.fmap2, flow, small_patch)
        return corr

    def corr(self, left_feature, right_feature, flow, small_patch):

        if not small_patch:
            psize = (1, 9)
        else:
            psize = (3, 3)

        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        right_feature = bilinear_sampler(right_feature, coords)

        # correlation_sampler = SpatialCorrelationSampler(
        #     kernel_size=1,
        #     patch_size=psize,
        #     stride=1,
        #     padding=0,
        #     dilation=1,
        #     dilation_patch=1)
        # corr = correlation_sampler(left_feature, right_feature)  # [N, ph, pw, H, W]
        # N, ph, pw, H, W = corr.size()
        # corr = corr.view(N, ph*pw, H, W)

        corr = self.get_correlation(left_feature, right_feature, psize)

        # if not small_patch:
        #     corr = self.conv1(corr)
        # else:
        #     corr = self.conv2(corr)

        # print("=====>", corr.size())

        return corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3)):

        N, C, H, W = left_feature.size()

        pady, padx = psize[0] // 2, psize[1] // 2
        right_pad = F.pad(right_feature, [padx, padx, pady, pady], mode="replicate")

        corr_list = []
        for h in range(0, pady * 2 + 1):
            for w in range(0, padx * 2 + 1):
                # start_h = pady*2 - h
                # start_w = padx*2 - w
                start_h = h
                start_w = w
                right_crop = right_pad[
                    :, :, start_h : start_h + H, start_w : start_w + W
                ]
                assert right_crop.size() == left_feature.size()
                corr = (left_feature * right_crop).mean(dim=1, keepdim=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final
