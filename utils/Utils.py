
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
from torch_geometric.nn import knn_graph
import json
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# This function converts Spherical coordinates to Cartesian.........................................####
def sphericalToCartesian(phi, theta, radius):
    x = radius*torch.cos(theta)*torch.sin(phi) 
    y = radius*torch.sin(theta)*torch.sin(phi) 
    z = radius*torch.cos(phi)
    xyz = torch.stack((x,y,z), dim=1)
    return xyz


def SphericalToUnitCartesian(phi, theta, radius):
    #print(phi, theta, radius)
    x = radius*torch.cos(theta)*torch.sin(phi) 
    y = radius*torch.sin(theta)*torch.sin(phi) 
    z = radius*torch.cos(phi)
    xyz = torch.stack((x,y,z), dim=1)
    return xyz

def SphericalToPixel(phi,theta, imgWidth, imgHeight):
    KPi2Inverted= 1/(2*np.pi)
    KPiInverted = 1/(np.pi)                          #spherical is actual sphere. spheremap is equirectangular image 
    x=imgWidth * (1. - (theta * KPi2Inverted)) - 0.5     #considering it as a unit sphere
    y=(imgHeight * phi) * KPiInverted - 0.5
    x=np.round(x)
    y=np.round(y)
    return x.view(-1).int().numpy(),y.view(-1).int().numpy()

def constrainSphericalBoundaries(keypointCoords):
    Pi = np.pi
    Pi2 = 2*np.pi
    new = []
    for ele in keypointCoords:   
        phi, theta = ele
        while (phi < 0.):
            phi += Pi2
        while (phi >= Pi2):
            phi -= Pi2

        if (phi >= Pi):
            phi = Pi2 - phi
            theta += Pi

        while (theta < 0.):
            theta += Pi2
        while (theta >= Pi2):
            theta -= Pi2

        new.append([phi,theta])

    return torch.tensor(new)

def draw_keypoints(img1, img2, data, radius=None, thickness=None):    
    # Function to draw Keypoints on two images
    '''
    inputs: img1: image 1, 
            img2: image 1, 
            data, 
    
    output: plots keypoints on two images
    '''
    if radius == None:
        radius=5
    if thickness == None:
        thickness=10
        
    # Info from Groundtruth
    kpt_img1, kpt_img2 = constrainSphericalBoundaries(data['keypointCoords0']), constrainSphericalBoundaries(data['keypointCoords1'])

    # Splitting phi and theta
    phi_img1, theta_img1 = torch.split(kpt_img1, 1, dim=1)
    phi_img2, theta_img2 = torch.split(kpt_img2, 1, dim=1)
    
    # Image shape info
    imgHeight,imgWidth = img1.shape[:2]
    # Converting color space of loaded image
    
    img1 = np.zeros((imgHeight, imgWidth, 1))
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Image shape info
    imgHeight,imgWidth = img1.shape[:2]

    # Calculating pixel coordinates from spherical coordinates
    x1,y1 = SphericalToPixel(phi_img1, theta_img1, imgWidth, imgHeight)
    x2,y2 = SphericalToPixel(phi_img2, theta_img2, imgWidth, imgHeight)

    #v_concat_img = cv2.vconcat([img1, img2])

    # Drawing keypoints on images
    for i in range(x1.shape[0]):
        v_concat_img = cv2.circle(img1, (x1[i],y1[i]), radius=radius, color=(255, 0, 0), thickness=thickness)
        #v_concat_img = cv2.circle(v_concat_img, (x2[i],imgHeight+y2[i]), radius=radius, color=(255, 0, 0), thickness=thickness)

    figure(figsize=(10, 10), dpi=160)
    plt.imshow(v_concat_img)
    #plt.savefig('sandbox/00000000')
    plt.show()
    
    
def match_color(score):    
    if score >= 0.8:
        color = (255, 0, 0)
    elif score < 0.8 and score >=0.4:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    return color


def draw_matches(img1_path, img2_path, data, out_path, arg, thickness=None):    
    # Function to draw Matches between two images
    '''
    inputs: img1: image 1, 
            img2: image 1, 
            data
    
    output: plots matches between image pair 
    '''
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if thickness == None:
        thickness=2
    
    # Info from Groundtruth
    correspondence = torch.tensor(data['correspondences']).long()    
    scores = torch.tensor(data['scores'])

    # Constrain Spherical Boundaries 
    kpt_img1, kpt_img2 = constrainSphericalBoundaries(data['keypointCoords0']), constrainSphericalBoundaries(data['keypointCoords1'])

    # Splitting phi and theta
    phi_img1, theta_img1 = torch.split(kpt_img1, 1, dim=1)
    phi_img2, theta_img2 = torch.split(kpt_img2, 1, dim=1)

    # Converting color space of loaded image
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Image shape info
    imgHeight,imgWidth = img1.shape[:2]

    # Calculating pixel coordinates from spherical coordinates
    x1,y1 = SphericalToPixel(phi_img1, theta_img1, imgWidth, imgHeight)
    x2,y2 = SphericalToPixel(phi_img2, theta_img2, imgWidth, imgHeight)
   
    mask = correspondence.ge(0)
    kpts_with_corr_img1 = torch.masked_select(torch.arange(correspondence.shape[0]), mask)
    kpts_with_corr_img2 = torch.masked_select(correspondence, mask)
    corresponding_kpts_index = torch.stack((kpts_with_corr_img1, kpts_with_corr_img2), 1)
    corresponding_kpts_scores = scores[kpts_with_corr_img1]
    
    v_concat_img = cv2.vconcat([img1, img2])
    for i in range(corresponding_kpts_index.shape[0]):
        ele = corresponding_kpts_index[i]
        score = corresponding_kpts_scores[i]
        v_concat_img = cv2.line(v_concat_img, (x1[ele[0]],y1[ele[0]]), (x2[ele[1]], imgHeight+y2[ele[1]]), color=(0,255,0), thickness=thickness)

    figure(figsize=(10, 10), dpi=80)
    plt.imshow(v_concat_img)
    if arg.save_drawn_matches is True:
        plt.axis('off')
        plt.savefig(out_path, bbox_inches='tight')

    plt.show()
    