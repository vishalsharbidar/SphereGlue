SphereGlue
==============================

Project Organization
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

