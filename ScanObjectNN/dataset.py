import os
import sys
import h5py
import time
import numpy as np
import torch
import pccnn_lib

class ScanObjectNNDataSet:
    """ScanObjectNNt dataset class.

    Attributes:
        numPts_ (int): Number of points.
        catNames_ (list of string): List of name of categories.
        models_ (list of numpy arrays of float): List of models
        categories_ (list of int): List of category index for each model.
        randomState_ (numpy RandomState): Random state.
        iterator_ (int): Model iterator.
        randList_ (int): List of random indexs.
        permute_ (bool): Boolean that indicates if the list of objects will be 
                permuted or not.
    """

    def __init__(self, 
        pDataSet = "training", 
        pNumPts = 1024,
        pRandSeed = None, 
        pPermute = True):

        self.numPts_ = pNumPts

        #Get the file list.
        dataSet = h5py.File("./data/main_split_nobg/"+pDataSet+"_objectdataset.h5", 'r')
        self.models_ = dataSet['data'][:]
        self.categories_ = dataSet['label'][:]
        #self.mask_ = dataSet['mask'][:]

        # Normalize.
        min_pts = np.amin(self.models_[:,:,0:3], axis=1, keepdims = True)
        max_pts = np.amax(self.models_[:,:,0:3], axis=1, keepdims = True)
        center = (max_pts + min_pts)*0.5
        self.models_[:,:,0:3] = self.models_[:,:,0:3] - center
        aabb_sizes = max_pts - min_pts
        max_aabb_sizes = np.amax(aabb_sizes, axis=-1, keepdims=True)
        self.models_[:,:,0:3] = (self.models_[:,:,0:3] / max_aabb_sizes)*2.0

        # Save the permutation flag.
        self.permute_ = pPermute

        # Iterator. 
        self.randomState_ = np.random.RandomState(pRandSeed)
        self.iterator_ = 0
        if self.permute_:
            self.randList_ = self.randomState_.permutation(len(self.models_))
        else:
            self.randList_ = np.arange(len(self.models_))
        

    def get_num_models(self):
        """Method to get the number of models in the dataset.

        Return:
            (int): Number of models.
        """
        return len(self.models_)


    def start_epoch(self):
        """Method to start a new epoch.
        """
        self.iterator_ = 0
        if self.permute_:
            self.randList_ = self.randomState_.permutation(len(self.models_))


    def get_next_batch(self, 
        pBatchSize, 
        pAugment = False, 
        pAgumentNoise = False, 
        pAugmentScaling = False, 
        pAugmentRot = False,
        pRotMatricesList = None,
        pReturnIndexs = False,
        pUpAxis = 1):
        """Method to get the next batch. If there are not enough models to fill
            the batch, None is returned.

        Args:
            pBatchSize (int): Size of the batch.
            pAugment (bool): Boolean that indicates the data has to be augmented.
            pAgumentNoise (bool): Use noise for the data augmentation.
            pAugmentScaling (bool): Use scaling for the data augmentation.
            pAugmentTrans (bool): Use rotation for data augmentation.
            pRotMatricesList (numpy array mx3x3): List of rotation matrices used for
                data augmentation.
        Returns:
            (float np.array nx6): Output points.
            (int np.array n): Output batch ids.
            (int np.array b): List of labels.
        """

        #Check for the validity of the input parameters.
        if pBatchSize <= 0:
            raise RuntimeError('Only a positive batch size is allowed.')

        #If there are enough models left.
        if self.iterator_ <= (len(self.models_)-pBatchSize):
            
            #Compute the total number of points.
            batchNumPts = pBatchSize*self.numPts_
            
            #Create the output buffers.
            pts = []
            batchIds = []
            labels = np.full((pBatchSize), 0, np.int32)
            batch_indexs = []

            #Get the data.
            for curIter in range(pBatchSize):

                #Select the model.
                curModel = np.copy(self.models_[self.randList_[self.iterator_+curIter]])
                batch_indexs.append(self.randList_[self.iterator_+curIter])

                # Sub select points.
                ptsChoice = self.randomState_.choice(curModel.shape[0], self.numPts_, replace=False)
                curModel = curModel[ptsChoice, :]
                    
                #Augment the data.
                if pAugment:

                    #Scale the data.
                    if pAugmentScaling:
                        curModel = pccnn_lib.py_utils.anisotropic_scale_pc(self.randomState_, curModel)

                    #Jitter and scale point cloud.
                    if pAgumentNoise:
                        curNoise = 0.002*self.randomState_.random_sample()
                        curModel[:, 0:3] = pccnn_lib.py_utils.jitter_pc(self.randomState_, curModel[:, 0:3], 
                            curNoise, curNoise*5)

                    #Rotate the model.
                    if pAugmentRot:
                        curModel[:, 0:3], rot_mat = pccnn_lib.py_utils.rotate_pc_3d(
                                self.randomState_, curModel[:, 0:3], 2.0*np.pi, [1])
                
                if pUpAxis != 1:
                    aux_coords = np.copy(curModel[:, 1])
                    curModel[:, 1] = np.copy(curModel[:, pUpAxis])
                    curModel[:, pUpAxis] = aux_coords
    
                #Save the augmented model.
                batchIds.append(np.full((len(curModel)), curIter, dtype=np.int32))
                pts.append(curModel)
                labels[curIter] = self.categories_[self.randList_[self.iterator_+curIter]]

            #Increment iterator.
            self.iterator_ += pBatchSize

            # Create the pytorch tensors.
            pts = np.concatenate(pts, axis = 0).astype(np.float32)
            batchIds = np.concatenate(batchIds)
            device = torch.device("cuda:0")
            torch_pts = torch.as_tensor(pts, device=device)
            torch_pts_batch_ids = torch.as_tensor(batchIds, device=device)
            torch_labels = torch.as_tensor(labels, device=device).to(torch.int64)
            
            #Return the current batch.
            if pReturnIndexs:
                return torch_pts, torch_pts_batch_ids, torch_labels, batch_indexs
            else:
                return torch_pts, torch_pts_batch_ids, torch_labels

        else:
            return None, None, None