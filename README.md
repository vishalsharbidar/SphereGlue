# SphereGlue
SphereGlue: A Graph Neural Network based feature matching for high-resolution spherical images

[Full paper PDF](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Gava_SphereGlue_Learning_Keypoint_Matching_on_High_Resolution_Spherical_Images_CVPRW_2023_paper.pdf)

![Architecture](https://github.com/vishalsharbidar/SphereGlue/assets/68814138/b9197d32-4470-41e8-b533-9278f5d6bd98)

# Repo Structure
------------

    ├── data
    │   ├── akaze                <- Data from akaze detector.
    │   ├── kp2d                 <- Data from kp2d detector.
    │   ├── sift                 <- Data from sift detector.
    │   ├── superpoint           <- Data from superpoint detector.
    |   └── superpoint_tf        <- Data from superpoint_tf detector.
    |
    ├── images                   <- Spherical images for visualizing matches
    |
    ├── matches                  <- Matches folder to save drawn matches (will be created automatically)
    │
    ├── models             
    │   └── spherglue.py         <- Trained and serialized models, model predictions, or model summaries     
    │
    ├── model_weights        
    │   ├── akaze                <- Model weights for akaze detector.
    │   ├── kp2d                 <- Model weights for kp2d detector.
    │   ├── sift                 <- Model weights for sift detector.
    │   ├── superpoint           <- Model weights for superpoint detector.
    |   └── superpoint_tf        <- Model weights for superpoint_tf detector.
    │
    ├── output                   <- Output folder to save the predictions (will be created automatically)
    |
    ├── utils              
    |   ├── demo_mydataset.py    <- Data loader
    |   └── Utils.py             <- Util file
    |
    ├── demo_SphereGlue.py       <- Demo code to run SphereGlue
    │
    ├── LICENSE
    |
    ├── README.md                <- The top-level README for developers using this project.
    |
    └── requirements.txt         <- The requirements file for reproducing the analysis environment, e.g.generated with `pip freeze > requirements.txt`
   


--------

# Dependencies
    Python 3 >= 3.9
    PyTorch >= 1.10
    Pytorch geometric >= 2.0
    OpenCV >= 4.5
    Matplotlib >= 3.5
    NumPy >= 1.21

Or simply run ``` pip install -r requirements.txt ```

# Citation
If you are using this code in your research, please cite our paper
```
@InProceedings{Gava_2023_CVPR,
    author    = {Gava, Christiano and Mukunda, Vishal and Habtegebrial, Tewodros and Raue, Federico and Palacio, Sebastian and Dengel, Andreas},
    title     = {SphereGlue: Learning Keypoint Matching on High Resolution Spherical Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {6133-6143}
}
```
