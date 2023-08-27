#!/usr/bin/env python3
import os
import boxx
import torch
import cv2
import numpy as np

with boxx.inpkg():
    # from .raft_infer import RAFT as Crestereo
    from .raft import RAFT as Crestereo


class CrestereoMatching(torch.nn.Module):
    def __init__(
        self,
        cfg=None,
    ):
        super().__init__()
        cfg = cfg or {}
        cfg.setdefault("max_disp", 256)
        default_pth = os.path.abspath(os.path.join(__file__, "../epoch-570.pth"))
        cfg.setdefault("pth", default_pth)
        self.crestereo = Crestereo(
            cfg["max_disp"], cfg.get("mixed_precision", False), test_mode=True
        )
        self.load_pth(cfg["pth"])
        self.crestereo.eval()
        self.cfg = cfg

    def __call__(self, rgb1, rgb2):
        device = self.device
        to_tensor = lambda rgb: torch.from_numpy(
            np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1)[None])
        ).to(device)
        bgr1 = to_tensor(rgb1)
        bgr2 = to_tensor(rgb2)

        with torch.no_grad():
            output = self.crestereo(bgr1, bgr2)
        disparity = output[0, 0].data.cpu().numpy()
        return dict(disparity=disparity)

    def load_pth(self, pth):
        from collections import OrderedDict

        state_dict = torch.load(pth, map_location=self.device)
        state_dict["state_dict"] = OrderedDict(
            [(k.replace("module.", ""), v) for k, v in state_dict["state_dict"].items()]
        )
        self.crestereo.load_state_dict(state_dict["state_dict"], strict=True)
        return self

    @property
    def device(self):
        param0 = next(self.crestereo.parameters())
        device = param0.device
        return device


if __name__ == "__main__":
    from boxx import *
    import tempfile
    import urllib.request

    matching = CrestereoMatching()

    def download(url, path):
        if not os.path.isfile(path):
            print(f'Test image "{path}" not exisit, download from: "{url}"')
            with open(path, "wb") as f, urllib.request.urlopen(url) as response:
                f.write(response.read())
        # return open(path, "rb").read()
        return path

    url1 = "https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png"
    url2 = "https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png"
    rgb1 = cv2.imread(
        download(url1, os.path.join(p/tempfile.gettempdir(), "crestereo_img1.png"))
    )[..., ::-1]
    rgb2 = cv2.imread(
        download(url2, os.path.join(tempfile.gettempdir(), "crestereo_img2.png"))
    )[..., ::-1]

    d = matching(rgb1, rgb2)
    tree - d
    show - d
