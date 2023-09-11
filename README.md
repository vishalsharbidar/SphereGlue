# SphereGlue
SphereGlue: A Graph Neural Network based feature matching for high-resolution spherical images

## Abstract
Traditionally, spherical keypoint matching has been performed using greedy algorithms, such as Nearest Neighbors (NN) search. NN based algorithms often lead to erroneous or insufficient matches as they fail to leverage global keypoint neighborhood information. Inspired by a recent learned perspective matching approach we introduce SphereGlue: a Graph Neural Network based feature matching for high-resolution spherical images. The proposed model naturally handles the severe distortions resulting from geometric transformations. Rigorous evaluations demonstrate the efficacy of SphereGlue in matching both learned and handcrafted keypoints, on synthetic and real high-resolution spherical images. Moreover, SphereGlue generalizes well to previously unseen real-world and synthetic scenes. Results on camera pose estimation show that SphereGlue can directly replace state-of-the-art matching algorithms, in downstream tasks.

## Network Architecture
![Architecture](https://github.com/vishalsharbidar/SphereGlue/assets/68814138/b9197d32-4470-41e8-b533-9278f5d6bd98)

[Full paper PDF](https://openaccess.thecvf.com/content/CVPR2023W/IMW/papers/Gava_SphereGlue_Learning_Keypoint_Matching_on_High_Resolution_Spherical_Images_CVPRW_2023_paper.pdf)


# Repo Structure
------------

    ├── data                     <- Keypoints information from two images. 
    │   ├── akaze                <- Data from akaze detector.
    │   ├── kp2d                 <- Data from kp2d detector.
    │   ├── sift                 <- Data from sift detector.
    │   ├── superpoint           <- Data from superpoint detector.
    |   └── superpoint_tf        <- Data from superpoint_tf detector.
    |
    ├── images                   <- Equirectangular images for visualizing matches
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

# Structure of Dataset

Keypoint Coordinates, Keypoint Descriptors, and Keypoint Scores can be extracted from:
1. SuperPoint: [Code](https://github.com/magicleap/SuperPointPretrainedNetwork)
2. KP2D: [Code](https://github.com/TRI-ML/KP2D)
3. Superpoint_tf: [Code](https://github.com/rpautrat/SuperPoint)
4. Akaze: : Will be added soon
5. Sift: : Will be added soon


Keypoint Coordinates used in SphereGlue are in spherical coordinates. The keypoint coordinates obtained from the above detectors will be in pixel coordinates. To convert this use: 
```
def PixelToSpherical(pixel_coordinates:np.array, imgWidth:int, imgHeight:int):
    x,y = np.hsplit(pixel_coordinates,2)
    theta = (1. - (x + .5) / imgWidth) * 2*np.pi
    phi = ((y + .5) * np.pi) / imgHeight
    return np.hstack((phi, theta))
```   

Create a list of image pair and merge the Keypoint Coordinates, Keypoint Descriptors, and Keypoint Scores of two images into a npz file. The structure of the npz file (dictionary) can be seen below:
``` 
{keypointCoords0: Keypoint Coordinates of image 0,
keypointCoords1: Keypoint Coordinates of image 1, 
keypointDescriptors0: Keypoint Descriptors of image 0,
keypointDescriptors1: Keypoint Descriptors of image 1,
keypointScores0: Keypoint Scores of image 0,
keypointScores1: Keypoint Scores of image 1
}
```

# Demo
To run the demo on the data, use ``` python demo_SuperGlue.py --save_npz True ```

There are 4 flags:
1. ``` --save_npz ```, when True, it will save the npz files in the folder output.
2. ``` --draw_matches ```, when True, it will save the drawn matches in the folder matches.
3. ``` --display_matches ```, when True, it will display the drawn matches.
4. ``` --detector ```, can be used to change the detector.
   ``` python demo_SuperGlue.py --save_npz True --detector 'sift' ```. 

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
