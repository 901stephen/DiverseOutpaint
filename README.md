# Multimodal Image Outpainting
This is a Pytorch implementation of our paper "Multimodal Image Outpainting". 

[Multimodal Image Outpainting With Regularized Normalized Diversification](http://openaccess.thecvf.com/content_WACV_2020/papers/Zhang_Multimodal_Image_Outpainting_With_Regularized_Normalized_Diversification_WACV_2020_paper.pdf) <br />
[Lingzhi Zhang](https://owenzlz.github.io/), Jiancong Wang, [Jianbo Shi](https://www.cis.upenn.edu/~jshi/)  <br />
GRASP Laboratory, University of Pennsylvania

In Winter Conference on Applications of Computer Vision (WACV), 2020.

## Introduction

We study the problem of generating a set of realistic and diverse backgrounds when given only a small foreground region, which we formulate as image outpainting task. We propose a generative model by improving the normalized diversification framework to encourage diverse sampling in this conditional synthesis task. The results show that our proposed approach can produce more diverse images with similar or better quality compare to the state-of-the-arts methods.

<img src='demo_imgs/car.png' align="middle" width=720>

## Modifications
1. Input and output dims changed from 128x128 to 64x64.
2. Provide train path at img_dir and mask path at mask_dir.
3. Tested on stanford car dataset.
4. https://ai.stanford.edu/~jkrause/cars/car_dataset.html.

## Mask Generation
1. Install depedencies for detectron2.
2. Setup detectron2 from https://github.com/facebookresearch/detectron2.
3. Copy mask_generation.py file provided in real_image directory to detectron2/demos.
4. Modify train/ dir and mask/ dir paths in mask_generation.py and.
5. Execute the file from detectron2 directory.
6. python demo/mask_generation.py.
 

## Citation
If you use this code for your research, please cite the [paper](http://openaccess.thecvf.com/content_WACV_2020/papers/Zhang_Multimodal_Image_Outpainting_With_Regularized_Normalized_Diversification_WACV_2020_paper.pdf):

```
@inproceedings{zhang2020multimodal,
  title={Multimodal Image Outpainting With Regularized Normalized Diversification},
  author={Zhang, Lingzhi and Wang, Jiancong and Shi, Jianbo},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={3433--3442},
  year={2020}
}
```
