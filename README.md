# Batch inference

This fork is very much a work in progress.
This fork implements batch inference with support for mask detections. It does not support all bells and whistles that Detectron comes with, and an exception will be raised notifying you when a feature is requested that isn't supported.

Caveats:
* Due to the way images within a batch are grouped to form a uniformly sized tensor, having images of different sizes and aspect ratios is very detrimental to performance. This was designed to work on videos, in which case this is't a problem.

# Video processing

This branch was modified to use videos directly instead of converting to images first.

Usage:
```bash
python test_net.py --cfg ../configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x_batch.yaml --multi-gpu-testing --video_list video_list.txt
```
where *video_list.txt* is a plain text file where each line contains the full path to a video that should be processed. This will run batch inference on the specified list of videos, using multiple gpus (as specified by the MODEL > NUM_GPUS value in the config file). The resulting detections will be saved in a pandas dataframe, which is pickled to the default location Detectron uses. The config file (here *../configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x_batch.yaml*) should contain a value under TEST, named IMS_PER_BATCH to set the number of images per batch for batch inference. In our case, *5* seems to be a good value.

Caveats:
* Currently, the videos consist of the combined feed of two cameras. Therefore, in [the dataset script](./detectron/datasets/video_dataset.py) each frame is split into a left and right part. If this isn't required, it should be straightforward to modify this to work with a single feed.
* Taskset is required to run for now, as explicitly setting the affinity of the subprocesses (during multi-gpu-testing) significantly boosts performance.

# Original README: Detectron

Detectron is Facebook AI Research's software system that implements state-of-the-art object detection algorithms, including [Mask R-CNN](https://arxiv.org/abs/1703.06870). It is written in Python and powered by the [Caffe2](https://github.com/caffe2/caffe2) deep learning framework.

At FAIR, Detectron has enabled numerous research projects, including: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), [Mask R-CNN](https://arxiv.org/abs/1703.06870), [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333), [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), [Non-local Neural Networks](https://arxiv.org/abs/1711.07971), [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370), [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440), [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/abs/1802.00434), and [Group Normalization](https://arxiv.org/abs/1803.08494).

<div align="center">
  <img src="demo/output/33823288584_1d21cf0a26_k_example_output.jpg" width="700px" />
  <p>Example Mask R-CNN output.</p>
</div>

## Introduction

The goal of Detectron is to provide a high-quality, high-performance
codebase for object detection *research*. It is designed to be flexible in order
to support rapid implementation and evaluation of novel research. Detectron
includes implementations of the following object detection algorithms:

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) -- *Marr Prize at ICCV 2017*
- [RetinaNet](https://arxiv.org/abs/1708.02002) -- *Best Student Paper Award at ICCV 2017*
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [RPN](https://arxiv.org/abs/1506.01497)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [R-FCN](https://arxiv.org/abs/1605.06409)

using the following backbone network architectures:

- [ResNeXt{50,101,152}](https://arxiv.org/abs/1611.05431)
- [ResNet{50,101,152}](https://arxiv.org/abs/1512.03385)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) (with ResNet/ResNeXt)
- [VGG16](https://arxiv.org/abs/1409.1556)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](#references) below.

## Update

- 4/2018: Support Group Normalization - see [`GN/README.md`](./projects/GN/README.md)

## License

Detectron is released under the [Apache 2.0 license](https://github.com/facebookresearch/detectron/blob/master/LICENSE). See the [NOTICE](https://github.com/facebookresearch/detectron/blob/master/NOTICE) file for additional details.

## Citing Detectron

If you use Detectron in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```
@misc{Detectron2018,
  author =       {Ross Girshick and Ilija Radosavovic and Georgia Gkioxari and
                  Piotr Doll\'{a}r and Kaiming He},
  title =        {Detectron},
  howpublished = {\url{https://github.com/facebookresearch/detectron}},
  year =         {2018}
}
```

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for Caffe2 and Detectron in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using Detectron

After installation, please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for brief tutorials covering inference and training with Detectron.

## Getting Help

To start, please check the [troubleshooting](INSTALL.md#troubleshooting) section of our installation instructions as well as our [FAQ](FAQ.md). If you couldn't find help there, try searching our GitHub issues. We intend the issues page to be a forum in which the community collectively troubleshoots problems.

If bugs are found, **we appreciate pull requests** (including adding Q&A's to `FAQ.md` and improving our installation instructions and troubleshooting documents). Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information about contributing to Detectron.

## References

- [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440).
  Ilija Radosavovic, Piotr Dollár, Ross Girshick, Georgia Gkioxari, and Kaiming He.
  Tech report, arXiv, Dec. 2017.
- [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370).
  Ronghang Hu, Piotr Dollár, Kaiming He, Trevor Darrell, and Ross Girshick.
  Tech report, arXiv, Nov. 2017.
- [Non-Local Neural Networks](https://arxiv.org/abs/1711.07971).
  Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
  Tech report, arXiv, Nov. 2017.
- [Mask R-CNN](https://arxiv.org/abs/1703.06870).
  Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
  Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He.
  Tech report, arXiv, June 2017.
- [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333).
  Georgia Gkioxari, Ross Girshick, Piotr Dollár, and Kaiming He.
  Tech report, arXiv, Apr. 2017.
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
  Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
  Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](http://arxiv.org/abs/1605.06409).
  Jifeng Dai, Yi Li, Kaiming He, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2016.
- [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497)
  Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2015.
- [Fast R-CNN](http://arxiv.org/abs/1504.08083).
  Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2015.
