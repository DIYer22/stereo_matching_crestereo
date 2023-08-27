# Effortlessly deploy CREStereo in PyTorch with a simple `pip install`.

![](https://github.com/megvii-research/CREStereo/blob/master/img/teaser.jpg?raw=true)
## Features
- Effortlessly by `pip install`
- PyTorch version of CREStereo
- Support both CUDA and CPU
- Combining with [calibrating](https://github.com/DIYer22/calibrating) to calibrate stereo cameras and get aligned pair of RGB and depth

## Install
```bash
# Install
pip install git+https://github.com/DIYer22/stereo_matching_crestereo.git

# Run stereo matching demo
python -m stereo_matching_crestereo.stereo_matching
```
## Python example
```Python
from stereo_matching_crestereo import CrestereoMatching
matching = CrestereoMatching()
disparity = matching(rgb1, rgb2)["disparity"]

# Combining with calibrating's stereo to get aligned pair of RGB and depth
# https://github.com/DIYer22/calibrating
stereo.set_stereo_matching(matching)
re = stereo.get_depth(rgb1, rgb2)
depth = re['unrectify_depth']
img1_undistort = re['undistort_img1']
```

## Citation
If you find the code or datasets helpful in your research, please cite:

```
@inproceedings{li2022practical,
  title={Practical stereo matching via cascaded recurrent network with adaptive correlation},
  author={Li, Jiankun and Wang, Peisen and Xiong, Pengfei and Cai, Tao and Yan, Ziwei and Yang, Lei and Liu, Jiangyu and Fan, Haoqiang and Liu, Shuaicheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16263--16272},
  year={2022}
}
```

