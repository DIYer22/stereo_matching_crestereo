import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import CorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


"""
raft_infer.py 和 raft.py 的不同:
- iters 参数不同(可能因为训练显存的原因?):
    - raft_infer 的 iters 参数是可变的
    - raft.py 在小尺寸网络上实际 iters 减半
- raft_infer 支持 flow_init, This skips the small refinement steps (1/16 and 1/8) and directly starts from the large refine (1/4) step
"""


class RAFT(nn.Module):
    def __init__(self, max_disp=192, mixed_precision=False, test_mode=False):
        super(RAFT, self).__init__()

        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        self.corr_levels = 4
        self.corr_radius = 4

        self.dropout = 0
        # self.iters = 10

        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", dropout=self.dropout
        )
        # self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
        # self.update_block = BasicUpdateBlock(hidden_dim=hdim, cor_planes=self.corr_levels * (2*self.corr_radius + 1)**2)
        self.update_block = BasicUpdateBlock(hidden_dim=hdim, cor_planes=9, mask_size=4)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask, rate=4):
        """Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def zero_init_flow(self, fmap):
        N, C, H, W = fmap.shape

        flow_u = torch.zeros([N, 1, H, W], dtype=torch.float)
        flow_v = torch.zeros([N, 1, H, W], dtype=torch.float)

        flow = torch.cat([flow_u, flow_v], dim=1).to(fmap.device)
        return flow

    def forward(self, image1, image2, iters=10, flow_init=None):
        """Estimate optical flow between pair of frames"""

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        # run the context network
        with autocast(enabled=self.mixed_precision):
            # cnet = self.cnet(image1)
            net, inp = torch.split(fmap1, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

            # 1/4 -> 1/8
            # feature
            s_fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            s_fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            # context
            s_net = F.avg_pool2d(net, 2, stride=2)
            s_inp = F.avg_pool2d(inp, 2, stride=2)

            # 1/4 -> 1/16
            # feature
            ss_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
            ss_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)
            # context
            ss_net = F.avg_pool2d(net, 4, stride=4)
            ss_inp = F.avg_pool2d(inp, 4, stride=4)

        corr_fn = CorrBlock(fmap1, fmap2)
        s_corr_fn = CorrBlock(s_fmap1, s_fmap2)
        ss_corr_fn = CorrBlock(ss_fmap1, ss_fmap2)

        flow_predictions = []
        # --------------- two stage refinement (1/16 + 1/8 + 1/4) ---------------
        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # init flow
            ss_flow = self.zero_init_flow(ss_fmap1)

            # small refine: 1/16
            for itr in range(iters):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                # --------------- update1 ---------------
                ss_flow = ss_flow.detach()  # detatch from graph, no gradient
                out_corrs = ss_corr_fn(ss_flow, small_patch)

                with autocast(enabled=self.mixed_precision):
                    ss_net, up_mask, delta_flow = self.update_block(
                        ss_net, ss_inp, out_corrs, ss_flow
                    )

                ss_flow = ss_flow + delta_flow
                flow = self.upsample_flow(ss_flow, up_mask, rate=4)
                flow_up = -4 * F.interpolate(
                    flow,
                    size=(4 * flow.shape[2], 4 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_predictions.append(flow_up)

            # small refine: 1/8
            scale = s_fmap1.shape[2] / flow.shape[2]
            s_flow = -scale * F.interpolate(
                flow,
                size=(s_fmap1.shape[2], s_fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
            for itr in range(iters):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                # --------------- update2 ---------------
                s_flow = s_flow.detach()  # detatch from graph, no gradient
                out_corrs = s_corr_fn(s_flow, small_patch)

                with autocast(enabled=self.mixed_precision):
                    s_net, up_mask, delta_flow = self.update_block(
                        s_net, s_inp, out_corrs, s_flow
                    )

                s_flow = s_flow + delta_flow
                flow = self.upsample_flow(s_flow, up_mask, rate=4)
                flow_up = -2 * F.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                flow_predictions.append(flow_up)

            # large refine: 1/4
            scale = fmap1.shape[2] / flow.shape[2]
            flow = -scale * F.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        for itr in range(iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            # --------------- update3 ---------------
            flow = flow.detach()
            out_corrs = corr_fn(flow, small_patch)

            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = -self.upsample_flow(flow, up_mask, rate=4)
            flow_predictions.append(flow_up)

        if self.test_mode:
            return flow_up

        return flow_predictions
