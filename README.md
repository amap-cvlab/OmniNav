# OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation

[![Home Page](https://img.shields.io/badge/Project-OmniNav-blue.svg)](https://astra-amap.github.io/omninav.github.io)
[![arXiv](https://img.shields.io/badge/Arxiv-2509.25687-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.25687)

## 🔥 Latest News!!
* February 12, 2026: We release the slow-fast collaboration system inference checkpoint for OVON benchmark.
* December 30, 2025: We release the slow-fast collaboration system inference code for OVON benchmark.
* December 11, 2025: We release the training code as well as the fast system (vision-only) inference code for the R2R, RxR, and OVON benchmarks.


## Quickstart
### 🧰 Installation

Clone the repo:

```
git clone https://github.com/amap-cvlab/OmniNav.git
```

Install Training dependencies:
```
# Ensure torch >= 2.6.0
cd train_code
pip install -r requirements.txt
```

Install habitat-sim and habitat-lab for inference
```
r2r&rxr and ovon have different version dependencies.
● habitat-sim
r2r&rxr: git clone https://github.com/facebookresearch/habitat-sim.git && cd habitat-sim && git checkout v0.1.7 
Ovon: git clone https://github.com/facebookresearch/habitat-sim.git && cd habitat-sim && git checkout v0.2.3
pip install -r requirements.txt
python setup.py install --headless
● habitat-lab
r2r&rxr: git clone https://github.com/facebookresearch/habitat-lab && cd habitat-lab && git checkout v0.1.7
ovon: git clone https://github.com/chongchong2025/habitat-lab && cd habitat-lab && git checkout v0.2.3_waypoint
python -m pip install -r habitat-baselines/habitat_baselines/rl/requirements.txt
python -m pip install -r habitat-baselines/habitat_baselines/rl/ddppo/requirements.txt
pip install -e .
cd habitat-baselines
pip install -e .
```

Optional Docker installation for R2R/RxR:

```
git clone https://github.com/amap-cvlab/OmniNav.git
cd OmniNav
docker build -t omninav:r2r-rxr .
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --shm-size=32g \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/checkpoints:/workspace/checkpoints \
  omninav:r2r-rxr
```

For OVON, follow the same Docker setup but build a separate image using our
provided Habitat-Sim `v0.2.3` and Habitat-Lab `v0.2.3_waypoint`.

### 🎁 Model
| Models                                  |                       Download Link                                           |
|-----------------------------------------|-------------------------------------------------------------------------------|
| Visual-only for r2r, rx2 & ovon         | [ModelScope](https://www.modelscope.ai/models/chongchongjj/OmniNav/)
| Slow-fast collaboration system for ovon | [ModelScope](https://www.modelscope.ai/models/chongchongjj/OmniNav_Slowfast/)



### ⚡ Inference
● r2r & rxr
``` sh
● visual_only
cd infer_r2r_rxr
bash eval_r2r.sh
bash eval_rxr.sh
```
● ovon
``` sh
● visual_only
cd infer_ovon
bash eval_ovon.sh

● frontier based slow-fast 
cd infer_ovon_slowfast
bash eval_ovon_slowfast

```


### ⚡ Training
``` sh
cd train_code
bash run_train_demo.sh
```

## 🏛️ Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{xue2025omninav,
  title={OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation},
  author={Xue, Xinda and Hu, Junjun and Luo, Minghua and Shichao, Xie and Chen, Jintao and Xie, Zixun and Kuichen, Quan and Wei, Guo and Xu, Mu and Chu, Zedong},
  journal={arXiv preprint arXiv:2509.25687},
  year={2025}
}
```

## Acknowledgments
Thanks to [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE), [MTU3D](https://github.com/MTU3D/MTU3D), and [OVON](https://github.com/naokiyokoyama/ovon) for open-sourcing the construction of training data and the closed-loop inference code. Their contributions have significantly enriched the open-source community.
