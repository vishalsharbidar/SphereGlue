# %BANNER_BEGIN%
# ------------------------------------------------------------------------------------
# 
# 
# 
# 
# 
#
# ------------------------------------------------------------------------------------
# %AUTHORS_BEGIN%
# 
# Creator: Vishal Sharbidar Mukunda
#
# %AUTHORS_END%
# ------------------------------------------------------------------------------------
# %BANNER_END%


import glob
import torch
from torch.utils.data import Dataset
from utils.Utils import sphericalToCartesian
from torch_geometric.nn import knn_graph
import json
import numpy


class MyDataset(Dataset):
    def __init__(self, knn, input, folder, device):
        self.knn = knn
        self.device = device
        self.folder = folder 
        self.gt_path = input
        self.data = glob.glob(self.gt_path + "*")
      
    def __len__(self):
        return len(self.data)
    
    def __UnitCartesian(self, points):     
        # Collecting keypoints infocc
        phi, theta =  torch.split(torch.as_tensor(points), 1, dim=1)
        unitCartesian = sphericalToCartesian(phi, theta, 1)
        return unitCartesian.squeeze(2).to(self.device)
    
    def __img2_corr(self, keypointCoords1, correspondences):
        target = -1*torch.ones(len(keypointCoords1)).to(self.device)
        mask = correspondences.ge(0)
        kptidx_with_corr_img0 = torch.masked_select(torch.arange(mask.shape[0]).to(self.device), mask).to(torch.int).to(target)
        kptidx_with_corr_img1 = torch.masked_select(correspondences, mask).to(torch.int64).to(self.device)
        corr_img1 = target.scatter_(0, kptidx_with_corr_img1, kptidx_with_corr_img0).long()
        return corr_img1.to(self.device)
     
        
    def datapreprocessor(self, gt):    
        # Loading the data
        keypoints1= gt['keypointCoords0']
        keypoints2 = gt['keypointCoords1']
        
        if self.folder == 'akaze':
            h1 = torch.as_tensor(gt['keypointDescriptors0']).to(self.device)
            pad1 = torch.zeros(h1.shape[0],3).to(self.device)
            h1 = torch.cat((h1,pad1), dim=-1)

            h2 = torch.as_tensor(gt['keypointDescriptors1']).to(self.device)
            pad2 = torch.zeros(h2.shape[0],3).to(self.device)
            h2 = torch.cat((h2,pad2), dim=-1)
        
        else:    
            h1 = torch.as_tensor(gt['keypointDescriptors0']).to(self.device)
            h2 = torch.as_tensor(gt['keypointDescriptors1']).to(self.device)

        
        scores_t1 = torch.as_tensor(gt['keypointScores0']).to(self.device)
        scores_t2 = torch.as_tensor(gt['keypointScores1']).to(self.device)

        
        # Conversion from Spherical coordinates to Unit Cartesian  
        unitCartesian1 = self.__UnitCartesian(keypoints1)
        unitCartesian2 = self.__UnitCartesian(keypoints2)  
        
        # getting nearest neighbours
        edges1 = knn_graph(unitCartesian1, k=self.knn, flow= 'target_to_source', cosine=True)        
        edges2 = knn_graph(unitCartesian2, k=self.knn, flow= 'target_to_source', cosine=True)    
        
        y_true = {'gt_matches0': [], 
                'gt_matches1': []
                 }

        data = {'unitCartesian1':unitCartesian1,
                'unitCartesian2':unitCartesian2,
                'h1':h1, 
                'h2':h2,
                'edges1':edges1,  
                'edges2':edges2, 
                'scores1': scores_t1,
                'scores2': scores_t2,
                'keypointCoords0': gt['keypointCoords0'],
                'keypointCoords1': gt['keypointCoords1'],
                }

        return data, y_true
    
    def __getitem__(self, idx):
        if self.data[idx].split('.')[-1] == 'json': 
            gt = json.load(open(self.data[idx]))
        if self.data[idx].split('.')[-1] == 'npz': 
            try:
                gt = dict(numpy.load(self.data[idx]))   
            except EOFError as e:
                print(e, self.data[idx].split('/')[-1])

            
            
            
        processed_data, y_true = self.datapreprocessor(gt)
        processed_data['name'] = self.data[idx].split('/')[-1]
        return processed_data, y_true
        
        #gt = json.load(open(self.data[idx]))
        #processed_data, y_true = self.datapreprocessor(gt)
        #processed_data['name'] = self.data[idx].split('/')[-1]
        #return processed_data, y_true