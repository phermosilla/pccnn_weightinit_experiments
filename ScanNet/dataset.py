
import os
import sys
import h5py
import time
import numpy as np
import torch

import pccnn_lib

class ScanNetDataSet:
    """ScanNet dataset class.
    """

    def __init__(self, 
        p_dataset = 0, 
        p_use_color = False, 
        p_inner_radius = 0.5, 
        p_oute_radius = 2.0,
        p_sample_points = 20000, 
        p_path="./data", 
        p_balanced_weighting = True, 
        p_rand_seed = None,
        p_min_pts = 0):
        """Constructor.

        Args:
            p_dataset (int): Dataset loaded. 0 - Training, 1 - Validation, 2 - Testing.
            p_use_color (bool): Boolean that indicates if we are in train mode.
            p_inner_radius (float): Inner radius used to select the points.
            p_oute_radius (float): Outer radius used to select the points.
            p_path (string): Path to the folder with the data.
            p_balanced_weighting (bool): Boolean that indicates if the balanced weighting
                is used.
            p_rand_seed (int): Random seed.
        """

        # Sample points.
        self.sample_points_ = p_sample_points
        self.min_pts_ = p_min_pts

        # Balanced variables.
        self.balanced_weighting_ = p_balanced_weighting

        # Save selection radii.
        self.inner_radius_ = p_inner_radius
        self.outer_radius_ = p_oute_radius

        # Use colors.
        self.use_olors_ = p_use_color

        # Save the dataset.
        self.dataset_ = p_dataset

        # Get the room list.
        rooms_file_path = p_path+"/rooms"
        rooms_num_pts_file_path = p_path+"/num_points"
        if self.dataset_ == 2:
            rooms_file_path = rooms_file_path + "_test"
            rooms_num_pts_file_path = rooms_num_pts_file_path+"_test"
        rooms_file_path = rooms_file_path + ".txt"
        rooms_num_pts_file_path = rooms_num_pts_file_path+".txt"
        self.rooms_ = np.loadtxt(rooms_file_path, dtype='str')
        self.room_num_pts_ = np.loadtxt(rooms_num_pts_file_path)

        # Get the rooms of the dataset.
        if self.dataset_ == 0:
            sel_rooms = set(np.loadtxt(p_path+"/scannet_train.txt", dtype='str'))
        elif self.dataset_ == 1:
            sel_rooms = set(np.loadtxt(p_path+"/scannet_val.txt", dtype='str'))
        elif self.dataset_ == 2:
            sel_rooms = set(np.loadtxt(p_path+"/scannet_test.txt", dtype='str'))
        rooms_indexs = np.array([i for i in range(len(self.rooms_)) if self.rooms_[i] in sel_rooms])
        self.rooms_ = self.rooms_[rooms_indexs]
        self.room_num_pts_ = self.room_num_pts_[rooms_indexs]

        # Initialize the random seed.
        if not(p_rand_seed is None):
            self.random_state_ = np.random.RandomState(p_rand_seed)
        else:
            self.random_state_ = np.random.RandomState(int(time.time()))

        # Load the labels identifiers and the weights.
        self.sem_labels_ = np.loadtxt(p_path+"/labels.txt", dtype='str', delimiter=':')
        weights = np.loadtxt(p_path+"/weights.txt")
        weights[0] = 1.0/np.log(1.2 + weights[0])
        weights[1] = 1.0/np.log(1.2 + weights[1])
        self.weights_ = weights[0]
        self.norm_weights_ = 1.0/self.weights_
        self.norm_weights_ = self.norm_weights_/np.amax(self.norm_weights_)
        self.weights_[0] = 0.0
        self.norm_weights_[0] = 0.0
        self.weights_ = self.weights_.astype(np.float32)
        self.norm_weights_ = self.norm_weights_.astype(np.float32)

        # Load data.
        print()
        print("######################## Loading dataset")
        self.min_potentials_ = []
        self.data_ = []
        max_label = 0
        for cur_room_iter, cur_room in enumerate(self.rooms_):
            cur_data = h5py.File(p_path+"/"+cur_room+".hdf5", 'r')
            cur_pos = cur_data['positions'][()]
            cur_colors = cur_data['colors'][()]
            cur_labels = cur_data['labels'][()]
            max_label = max(max_label, np.amax(cur_labels))
            cur_data.close()
            cur_colors = cur_colors.astype(np.float32)/255.0
            potentials = self.random_state_.uniform(0.0, 1e-3, (cur_pos.shape[0]))
            potentials[cur_labels==0] = 3.0e+37
            self.data_.append([cur_pos, cur_colors, cur_labels, potentials])
            self.min_potentials_.append(np.amin(potentials))
            if cur_room_iter%100 == 0:
                print("Rooms loaded:", cur_room_iter)

        # Random list.
        self.rand_list_ = self.random_state_.permutation(len(self.rooms_))
        self.room_iterator_ = 0


    def get_num_rooms(self):
        """Method to ge the number of rooms.

        Returns:
            int: Number of rooms.
        """
        return len(self.rooms_)


    def get_num_pts_room(self, p_room_id):
        """Method to ge the number of points for a given room.

        Returns:
            int: Number of pts.
        """
        return len(self.data_[p_room_id][3])

    
    def get_room_labels(self, p_room_id):
        """Method to ge the number of points for a given room.

        Returns:
            int: Number of pts.
        """
        return self.data_[p_room_id][2]
        
        
    def get_labels(self):
        """Constructor.

        Returns:
            String list: Label names.
        """
        return self.sem_labels_
        
        
    def _get_weights(self, p_labels):   
        """Method to get the weights for a set o points.

        Args:
            p_labels (np.array n): Labels of each point.
        Returns:
            float list: Weight list.
        """            
        out_weights = self.weights_[p_labels.reshape(-1)]
        return out_weights


    def _get_norm_weights(self, p_labels):   
        """Method to get the normalized weights for a set o points.

        Args:
            p_labels (np.array n): Labels of each point.
        Returns:
            float list: Weight list.
        """            
        out_weights = self.norm_weights_[p_labels.reshape(-1)]
        return out_weights
        
    
    def _get_accuracy_masks(self, p_labels):
        """Method to get the mask of valid points.

        Args:
            p_labels (np.array n): Labels of each point.
        Returns:
            float list: Mask for each point.
        """  
        out_masks = np.minimum(p_labels.reshape(-1).astype(np.float), 1.0)
        return out_masks


    def restart_potentials(self):
        """Method to restart the potentials used for training.
        """ 
        for cur_room_iter, cur_room in enumerate(self.rooms_):
            cur_labels = self.data_[cur_room_iter][2]
            potentials = self.random_state_.uniform(0.0, 1e-3, (cur_labels.shape[0]))
            potentials[cur_labels==0] = 3.0e+37
            self.data_[cur_room_iter][3] = potentials
            self.min_potentials_[cur_room_iter] = np.amin(potentials)


    def compute_variance_color(self):
        """Method to compute the variance of the color channels.

        Returns:
            (float): Variance of the color channels
        """
        variances = []
        for cur_room_iter, cur_room in enumerate(self.rooms_):
            variances.append(np.var(self.data_[cur_room_iter][1]))
        return np.mean(variances)

    
    def get_next_batch(self, 
        p_num_rooms, 
        p_augment=False, 
        p_agument_noise = False, 
        p_agument_mirror = False, 
        p_augment_scaling = False, 
        p_augment_rot = False, 
        p_rot_angle = None, 
        p_update_pot = True, 
        p_get_ids = False, 
        p_room_id = None):
        """Method to get the next batch.

        Args:
            p_num_rooms (int): Number of rooms to query.
            p_augment (bool): Boolean that indicates if we augment the data.
            p_agument_noise (bool): Boolean that indicates if we augment using noise.
            p_agument_mirror (bool): Boolean that indicates if we augment using mirror.
            p_augment_scaling (bool): Boolean that indicates if we augment via scaling.
            p_augment_rot (bool): Boolean that indicates if we augment by rotating the model.
            p_rot_angle (float): Angle of rotation.
            p_update_pot (bool): Boolean that indicates if we update potentials.
            p_get_ids (bool): Boolean that indicates if the function will return the room ids
                and point ids.
            p_room_id (int): Room identifier to get the batch from.
        Returns:
            Pointcloud: Outer pointcloud.
            float pytorch tensor: Outer point colors.
            Pointcloud: Inner pointcloud.
            float pytorch tensor: Inner point labels.
            float pytorch tensor: Inner point weights.
            float pytorch tensor: Inner point accuracy mask.
        """  
        
        # Declare output variables.
        inner_pts = []
        inner_pt_labels = []
        inner_pts_batch_ids = []
        inner_pt_acc_mask = []
        outer_pts = []
        outer_pt_colors = []
        outer_pts_batch_ids = []
        room_ids = []
        pts_ids = []

        # Iterate over the number of rooms.
        for cur_room_iter in range(p_num_rooms):

            selected_room = False

            while not selected_room:
                # Select room.
                if p_room_id is None:
                    #room_index = np.argmin(self.min_potentials_)
                    if self.room_iterator_ < len(self.rand_list_):
                        room_index = self.rand_list_[self.room_iterator_]
                    else:
                        self.rand_list_ = self.random_state_.permutation(len(self.rooms_))
                        room_index = self.rand_list_[0]
                        self.room_iterator_ = 0
                    self.room_iterator_ += 1
                else:
                    room_index = p_room_id

                # Select center.
                pt_center_id = np.argmin(self.data_[room_index][3])
                pt_center_coords = self.data_[room_index][0][pt_center_id]

                # Remove the center from the coordinates.
                rel_coords = (self.data_[room_index][0] - pt_center_coords)#[:, 0:2]
                rel_dist = np.sum(np.square(rel_coords), axis = 1)
                mask_inner = rel_dist < (self.inner_radius_ ** 2.0)
                mask_outer = rel_dist < (self.outer_radius_ ** 2.0)

                # Get the data.
                cur_inner_pts = self.data_[room_index][0][mask_inner]
                cur_inner_pt_labels = self.data_[room_index][2][mask_inner]
                cur_inner_batch_ids = np.full((cur_inner_pts.shape[0]), cur_room_iter, dtype=np.int32)
                cur_outer_pts = self.data_[room_index][0][mask_outer]
                cur_outer_batch_ids = np.full((cur_outer_pts.shape[0]), cur_room_iter, dtype=np.int32)
                if self.use_olors_:
                    cur_outer_pt_colors = self.data_[room_index][1][mask_outer, 0:3]
                else:
                    cur_outer_pt_colors = np.full((cur_outer_pts.shape[0]), 1.0, dtype=np.float32)

                # Accumulate the ids.
                if p_get_ids:
                    room_ids.append(room_index)
                    pts_ids.append(np.arange(len(self.data_[room_index][0]))[mask_inner])

                # Update potentials.
                if p_update_pot:
                    updates_pot = np.square(1.0 - (rel_dist[mask_inner]/(self.inner_radius_ ** 2.0)))
                    norm_data_weights = self._get_accuracy_masks(cur_inner_pt_labels)
                    self.data_[room_index][3][mask_inner] += updates_pot*norm_data_weights
                    self.min_potentials_[room_index] = np.amin(self.data_[room_index][3])

                # Select a subset of the inner pts.
                if self.dataset_ == 0 and self.sample_points_ > 0:
                    mask_valid_pts = cur_inner_pt_labels>0
                    masked_indexs = np.arange(len(cur_inner_pts))[mask_valid_pts]
                    inner_sel = self.random_state_.choice(masked_indexs, size=self.sample_points_, 
                        replace=len(masked_indexs)<self.sample_points_)
                    cur_inner_pts = cur_inner_pts[inner_sel]
                    cur_inner_pt_labels = cur_inner_pt_labels[inner_sel]
                    cur_inner_batch_ids = cur_inner_batch_ids[inner_sel]

                # Center.
                cur_center = np.mean(cur_inner_pts, axis= 0, keepdims=True)
                cur_inner_pts = cur_inner_pts - cur_center
                cur_outer_pts = cur_outer_pts - cur_center

                # Augment the data.
                if p_augment:

                    #Scale the data.
                    if p_augment_scaling:
                        cur_inner_pts, scaling = pccnn_lib.py_utils.anisotropic_scale_pc(
                            self.random_state_, 
                            cur_inner_pts, 
                            p_return_scaling=True)
                        cur_outer_pts = cur_outer_pts*scaling

                    #Rotate the model.
                    if p_augment_rot:
                        if p_rot_angle is None:
                            cur_inner_pts, rot_mat = pccnn_lib.py_utils.rotate_pc_3d(
                                self.random_state_, cur_inner_pts, 2.0*np.pi, [2])
                            cur_outer_pts = np.dot(cur_outer_pts, rot_mat)
                        else:
                            cos_val = np.cos(p_rot_angle)
                            sin_val = np.sin(p_rot_angle)
                            rot_mat = np.array([ [cos_val, -sin_val, 0.],
                                                [sin_val, cos_val, 0.],
                                                [0., 0., 1.]])
                            cur_inner_pts = np.dot(cur_inner_pts, rot_mat)
                            cur_outer_pts = np.dot(cur_outer_pts, rot_mat)

                    #Mirror the data.
                    if p_agument_mirror:
                        cur_inner_pts, cur_outer_pts = pccnn_lib.py_utils.mirror_pc(
                            self.random_state_, [True, True, False], 
                            cur_inner_pts, cur_outer_pts)

                    #Jitter and scale point cloud.
                    if p_agument_noise:
                        cur_noise = 0.001*self.random_state_.random_sample()
                        cur_inner_pts = pccnn_lib.py_utils.jitter_pc(self.random_state_, 
                            cur_inner_pts, cur_noise, cur_noise*5)
                        cur_outer_pts = pccnn_lib.py_utils.jitter_pc(self.random_state_, 
                            cur_outer_pts, cur_noise, cur_noise*5)

                    cur_inner_pts = cur_inner_pts.astype(np.float32)
                    cur_outer_pts = cur_outer_pts.astype(np.float32)

                # Compute the weights and accuracy mask
                cur_inner_pt_acc_mask = self._get_accuracy_masks(cur_inner_pt_labels)

                if self.min_pts_ > 0:
                    selected_room = cur_inner_pts.shape[0] > self.min_pts_ or self.dataset_ != 0
                else:
                    selected_room = True

            # Append the values to the batch.
            inner_pts.append(cur_inner_pts)
            inner_pt_labels.append(cur_inner_pt_labels)
            inner_pts_batch_ids.append(cur_inner_batch_ids)
            inner_pt_acc_mask.append(cur_inner_pt_acc_mask)
            outer_pts.append(cur_outer_pts)
            outer_pt_colors.append(cur_outer_pt_colors)
            outer_pts_batch_ids.append(cur_outer_batch_ids)

        # Concatenate output into a single list.
        inner_pts = np.concatenate(inner_pts, axis=0)
        inner_pt_labels = np.concatenate(inner_pt_labels, axis = 0)
        inner_pts_batch_ids = np.concatenate(inner_pts_batch_ids, axis = 0)
        inner_pt_acc_mask = np.concatenate(inner_pt_acc_mask, axis = 0)
        outer_pts = np.concatenate(outer_pts, axis = 0)
        outer_pt_colors = np.concatenate(outer_pt_colors, axis = 0)
        outer_pts_batch_ids = np.concatenate(outer_pts_batch_ids, axis = 0)

        
        # Create the pytorch tensors.
        device = torch.device("cuda:0")
        torch_outer_pts = torch.as_tensor(outer_pts, device=device)
        torch_outer_pt_colors = torch.as_tensor(outer_pt_colors, device=device)
        torch_outer_pts_batch_ids = torch.as_tensor(outer_pts_batch_ids, device=device)
        torch_inner_pts = torch.as_tensor(inner_pts, device=device)
        torch_inner_pts_batch_ids = torch.as_tensor(inner_pts_batch_ids, device=device)
        torch_inner_pt_labels = torch.as_tensor(inner_pt_labels, device=device)
        torch_inner_pt_acc_mask = torch.as_tensor(inner_pt_acc_mask, device=device)
        

        # Create the point cloud objects.
        outer_pc = pccnn_lib.pc.Pointcloud(torch_outer_pts, torch_outer_pts_batch_ids, device=device)
        inner_pc = pccnn_lib.pc.Pointcloud(torch_inner_pts, torch_inner_pts_batch_ids, device=device)

        # Return the values.
        if p_get_ids:
            return outer_pc, torch_outer_pt_colors, \
                inner_pc, torch_inner_pt_labels, \
                torch_inner_pt_acc_mask, \
                room_ids, pts_ids
        else:
            return outer_pc, torch_outer_pt_colors, \
                inner_pc, torch_inner_pt_labels, \
                torch_inner_pt_acc_mask