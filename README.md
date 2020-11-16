# ApproxDet: Content and Contention-Aware Approximate Object Detection for Mobiles
[[Paper]](https://starsthu2016.github.io/uploads/sensys20-xu-ran.pdf) [[Short Video]](https://www.youtube.com/watch?v=8i749DG2wqQ) [[Presentation Video]](https://www.youtube.com/watch?v=nMlPStUkTp4) [[Project]](https://starsthu2016.github.io/projects/approxdet/approxdet_sensys20.html)  
Authors: [Ran Xu](https://starsthu2016.github.io/), [Chen-lin Zhang](http://www.lamda.nju.edu.cn/zhangcl/), [Pengcheng Wang](https://www.schaterji.io/students/pengcheng-wang.html), [Jayoung Lee](https://www.schaterji.io/students/jayoung-lee.html), [Subrata Mitra](https://research.adobe.com/person/subrata-mitra/), [Somali Chaterji](https://schaterji.io/), [Yin Li](https://www.biostat.wisc.edu/~yli/), [Saurabh Bagchi](https://engineering.purdue.edu/~sbagchi/)

Advanced video analytic systems, including scene classification and object detection, have seen widespread success in various domains such as smart cities and autonomous transportation. With an ever-growing number of powerful client devices, there is incentive to move these heavy video analytics workloads from the cloud to mobile devices to achieve low latency and real-time processing and to preserve user privacy. However, most video analytic systems are heavyweight and are trained offline with some pre-defined latency or accuracy requirements. This makes them unable to adapt at runtime in the face of three types of dynamism -- the input video characteristics change, the amount of compute resources available on the node changes due to co-located applications, and the user's latency-accuracy requirements change. In this paper we introduce ApproxDet, an adaptive video object detection framework for mobile devices to meet accuracy-latency requirements in the face of changing content and resource contention scenarios. To achieve this, we introduce a multi-branch object detection kernel (layered on Faster R-CNN), which incorporates a data-driven modeling approach on the performance metrics, and a latency SLA-driven scheduler to pick the best execution branch at runtime. We couple this kernel with approximable video object tracking algorithms to create an end-to-end video object detection system. We evaluate ApproxDet on a large benchmark video dataset and compare quantitatively to AdaScale and YOLOv3. We find that ApproxDet is able to adapt to a wide variety of contention and content characteristics and outshines all baselines, e.g., it achieves 52% lower latency and 11.1% higher accuracy over YOLOv3.

# Contact
If you have any questions or suggestions, feel free to email Ran Xu (xu943@purdue.edu).

# Cite
If you find our work useful and relevant to your paper, please consider to cite,
```
@inproceedings{xu2020approxdet,
  title={ApproxDet: Content and contention-aware approximate object detection for mobiles},
  author={Xu, Ran and Zhang, Chen-lin and Wang, Pengcheng and Lee, Jayoung and Mitra, Subrata and Chaterji, Somali and Li, Yin and Bagchi, Saurabh},
  booktitle={Proceedings of the 18th ACM Conference on Embedded Networked Sensor Systems (SenSys)},
  pages={1--14},
  year={2020}
}
```

# Hardware Prerequisite
* An NVIDIA Jetson TX2 board, for teaser experiments.
* A high-end server is needed for profiling experiments.
* Note: an equivalent embedded device (like NVIDIA Jetson Nano, Xavier, and Raspberry Pi) is acceptable but you will not be able to replicate our results quickly as you will need to do the time-consumping profiling and modeling experiments on your device.

# Installation
We provide the installation guide on the NVIDIA Jetson TX2. The installation on the other device may vary.

* Install conda4aarch64: please follow this [instruction](https://github.com/jjhelmus/conda4aarch64).
* Install numba (Python package), please follow this [instruction](https://numba.pydata.org/numba-doc/latest/user/installing.html#installing-on-linux-armv8-aarch64-platforms) and install numba version 0.46.0.
* Install TensorFlow (Python package), please follow this [instruction](https://forums.developer.nvidia.com/t/tensorflow-for-jetson-tx2/64596) and install tensorflow-gpu version 1.14.0+nv19.7.
* Install opencv version 4.1.1 (Python package) ([reference](https://qengineering.eu/install-opencv-4.5-on-jetson-nano.html)).
* Install other Python package,
```
pip install gdown opencv-python opencv-contrib-python
conda install pillow tqdm
```
* numpy version: make sure the numpy version is <= 1.16.4
* compile C code,
```
cd utils_approxdet
make
```

# Large File and Dataset Placement
Due to the 100 MB file limit on the GitHub, we upload the [model weights](https://drive.google.com/file/d/1wRv8PEYCfBC2bwZoVXLKk-kngiGKjZ6c/view?usp=sharing) and [video dataset (2.7 GB)](https://drive.google.com/file/d/13z3qMObub1-GNv6HR1A56USdPBLcx6sq/view?usp=sharing) on the Google Drive. You may use the following commands to download and extract the content.
```
gdown https://drive.google.com/uc?id=13z3qMObub1-GNv6HR1A56USdPBLcx6sq
gdown https://drive.google.com/uc?id=1wRv8PEYCfBC2bwZoVXLKk-kngiGKjZ6c
tar -xvf Data.tar
tar -xvf models.tar
```
The model weights and video datasets should be placed in the proper path so that they can be found. Generally, you need to specify "--dataset-prefix" argument for the video datasets. We highly recommand the following file path so that you can use the default commands,
```
/home/nvidia/ILSVRC2015/Data  % the video dataset
/home/nvidia/ApproxDet_Replicate  % src path
/home/nvidia/ApproxDet_Replicate/models  % model path
```

# Teaser Experiments
## Case Study w/ Increasing Contention
In this experiment, we inject increasing GPU contention in the background from 0% to 60% with an increment of 20% every 200 frames. ApproxDet is able to adapt to this increasing resource contention. Here is how the replicate,  
Run the contention generator program in the background,
```
screen
python3 ApproxDet_CG.py
```

Run ApproxDet,
```
python3 ApproxDet.py --input=test/VID_testimg_cs1.txt --preheat=1 \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --weight=models/ApproxDet.pb \
  --output=test/VID_testset_ApproxDet_cs1.txt
```

## Case Study w/ Increasing Latency Budget
In this experiment, we increase the latency requirement from 80ms to 300ms per frame with an increment every 200 frames. ApproxDet is able to adapt to this changing latency requirement. Here is how the replicate,  
Run the contention generator program in the background,
```
screen
python3 ApproxDet_CG.py
```

Run ApproxDet,
```
python3 ApproxDet.py --input=test/VID_testimg_cs2.txt --preheat=1 \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --weight=models/ApproxDet.pb \
  --output=test/VID_testset_ApproxDet_cs2.txt
```

# Larger Scale Experiments
The following experiments take much long time and may be harder to understand. Please read our paper throughly before doing the experiments.  
**The contention generator in the background is not needed in the following experiments, so remember to kill it by,**
```
sudo pkill -f ApproxDet_CG
sudo pkill -f contention_module
```

## 1. Object Detector and Object Trackers with Approximate Knobs
### 1.1 Test and Examine the mAP on the Object Detector
```
python3 ApproxDetection.py --imagefiles=test/VID_testimg_00106000.txt \
  --repeat=1 --preheat=1 \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --shape=576 --nprop=100 \
  --weight=models/ApproxDet.pb \
  --output=test/VID_tmp.txt
python3 utils_approxdet/compute_mAP.py --gt=test/VID_testgt_full.txt \
  --detection=test/VID_tmp_nprop100_shape576_det.txt \
  --video=Data/VID/val/ILSVRC2015_val_00106000
```
Note: you may need to configure SWAP memory to run this experiment ([reference](https://www.jetsonhacks.com/2019/11/28/jetson-nano-even-more-swap/)).

### 1.2 Test and Examine the mAP on the Object Trackers
```
python3 ApproxTracking.py --imagefiles=test/VID_testimg_00106000.txt \
  --detection_file=test/VID_testset_nprop100_shape576_det.txt \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --repeat=1 --si=8 --tracker_ds=medianflow_ds4 \
  --output=test/VID_valset_nprop100_shape576.txt
python3 utils_approxdet/compute_mAP.py --gt=test/VID_testgt_full.txt \
  --detection=test/VID_valset_nprop100_shape576_medianflow_ds4_det.txt \
  --video=Data/VID/val/ILSVRC2015_val_00106000
```

### 1.3 Test and Examine the mAP on the Object Detector + Tracker
```
python3 ApproxKernel.py --imagefiles=test/VID_testimg_00106000.txt \
  --repeat=1 --preheat=1 \
  --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --shape=576 --nprop=100 --si=8 --tracker_ds=medianflow_ds4 \
  --weight=models/ApproxDet.pb \
  --output=test/VID_tmp.txt
python3 utils_approxdet/compute_mAP.py --gt=test/VID_testgt_full.txt \
  --detection=test/VID_tmp_nprop100_shape576_medianflow_ds4_det.txt \
  --video=Data/VID/val/ILSVRC2015_val_00106000
```

## 2. Offline Profiling
### 2.1 Latency Profiling of the Object Detector
```
python3 ApproxDetctionProfiler.py --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --weight=models/ApproxDet.pb
```

### 2.2 Latency Profiling of the Object Tracker
```
python3 ApproxTrackingProfiler.py --dataset_prefix=/home/nvidia/ILSVRC2015/
```

### 2.3 Accuracy Profiling for the Object Detector
```
python3 ApproxDetection.py --imagefiles=test/VID_valimg_full.txt \
  --repeat=1 \
  --dataset_prefix=/data1/group/mlgroup/train_data/ILSVRC2015/ \
  --shape=576 --nprop=100 \
  --weight=models/ApproxDet.pb \
  --output=test/VID_valset.txt
```

### 2.4 Accuracy Profiling for the Object Detector + Tracker
```
python3 ApproxAccProfilerMaster.py --port=5050 --logs=test/VID_tmp.log
python3 ApproxAccProfilerWorker.py --ip=localhost --port=5050 \
  --dataset_prefix=/data1/group/mlgroup/train_data/ILSVRC2015/ --cpu=0
```

### 2.5 Switching Overhead Profiling
```
python3 ApproxDetctionProfilerSw.py --dataset_prefix=/home/nvidia/ILSVRC2015/ \
  --weight=models/ApproxDet.pb --output=test/VID_switchingoverhead_run0.txt
```
