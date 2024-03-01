#!/usr/bin/env python3
import boxx

if __name__ == "__main__":
    from boxx import *


    cam1, cam2, camd = Cam.get_test_cams()
    key = list(cam1)[0]
    img1 = boxx.imread(cam1[key]["path"])
    img2 = boxx.imread(cam2[key]["path"])
    stereo = Stereo(cam1, cam2)