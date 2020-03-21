# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms
from copy import deepcopy

# -----------------------------------------
# modules
# -----------------------------------------
import sys
import re
from PIL import Image
from copy import deepcopy

sys.dont_write_bytecode = True
from lib.rpn_util import *
from lib.util import *
from lib.augmentations import *
from lib.core import *


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 conf,
                 root,
                 cache_folder=None,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        imdb = []

        self.video_det = False if not ('video_det' in conf) else conf.video_det
        self.video_count = 1 if not ('video_count' in conf) else conf.video_count
        self.use_3d_for_2d = ('use_3d_for_2d' in conf) and conf.use_3d_for_2d

        # use cache?
        if (cache_folder is not None) and os.path.exists(os.path.join(cache_folder, 'imdb.pkl')):
            logging.info('Preloading imdb.')
            imdb = pickle_read(os.path.join(cache_folder, 'imdb.pkl'))

        else:

            # cycle through each dataset
            for dbind, db in enumerate(conf.datasets_train):

                logging.info('Loading imdb {}'.format(db['name']))

                # single imdb
                imdb_single_db = []

                # kitti formatting
                if db['anno_fmt'].lower() == 'kitti_det':

                    train_folder = os.path.join(root, db['name'], 'training')

                    ann_folder = os.path.join(train_folder, 'label_2', '')
                    cal_folder = os.path.join(train_folder, 'calib', '')
                    im_folder = os.path.join(train_folder, 'image_2', '')

                    # get sorted filepaths
                    annlist = sorted(glob(ann_folder + '*.txt'))

                    imdb_start = time()

                    self.affine_size = None if not ('affine_size' in conf) else conf.affine_size

                    for annind, annpath in enumerate(annlist):

                        # get file parts
                        base = os.path.basename(annpath)
                        id, ext = os.path.splitext(base)

                        calpath = os.path.join(cal_folder, id + '.txt')
                        impath = os.path.join(im_folder, id + db['im_ext'])
                        impath_pre = os.path.join(train_folder, 'prev_2', id + '_01' + db['im_ext'])
                        impath_pre2 = os.path.join(train_folder, 'prev_2', id + '_02' + db['im_ext'])
                        impath_pre3 = os.path.join(train_folder, 'prev_2', id + '_03' + db['im_ext'])

                        # read gts
                        p2 = read_kitti_cal(calpath)
                        p2_inv = np.linalg.inv(p2)

                        gts = read_kitti_label(annpath, p2, self.use_3d_for_2d)

                        if not self.affine_size is None:

                            # filter relevant classes
                            gts_plane = [deepcopy(gt) for gt in gts if gt.cls in conf.lbls and not gt.ign]

                            if len(gts_plane) > 0:

                                KITTI_H = 1.65

                                # compute ray traces for default projection
                                for gtind in range(len(gts_plane)):
                                    gt = gts_plane[gtind]

                                    #cx2d = gt.bbox_3d[0]
                                    #cy2d = gt.bbox_3d[1]
                                    cy2d = gt.bbox_full[1] + gt.bbox_full[3]
                                    cx2d = gt.bbox_full[0] + gt.bbox_full[2] / 2

                                    z2d, coord3d = projection_ray_trace(p2, p2_inv, cx2d, cy2d, KITTI_H)

                                    gts_plane[gtind].center_in = coord3d[0:3, 0]
                                    gts_plane[gtind].center_3d = np.array(gt.center_3d)


                                prelim_tra = np.array([gt.center_in for gtind, gt in enumerate(gts_plane)])
                                target_tra = np.array([gt.center_3d for gtind, gt in enumerate(gts_plane)])

                                if self.affine_size == 4:
                                    prelim_tra = np.pad(prelim_tra, [(0, 0), (0, 1)], mode='constant', constant_values=1)
                                    target_tra = np.pad(target_tra, [(0, 0), (0, 1)], mode='constant', constant_values=1)

                                affine_gt, err = solve_transform(prelim_tra, target_tra, compute_error=True)

                                a = 1

                        obj = edict()

                        # did not compute transformer
                        if (self.affine_size is None) or len(gts_plane) < 1:
                            obj.affine_gt = None
                        else:
                            obj.affine_gt = affine_gt

                        # store gts
                        obj.id = id
                        obj.gts = gts
                        obj.p2 = p2
                        obj.p2_inv = p2_inv

                        # im properties
                        im = Image.open(impath)
                        obj.path = impath
                        obj.path_pre = impath_pre
                        obj.path_pre2 = impath_pre2
                        obj.path_pre3 = impath_pre3
                        obj.imW, obj.imH = im.size

                        # database properties
                        obj.dbname = db.name
                        obj.scale = db.scale
                        obj.dbind = dbind

                        # store
                        imdb_single_db.append(obj)

                        if (annind % 1000) == 0 and annind > 0:
                            time_str, dt = compute_eta(imdb_start, annind, len(annlist))
                            logging.info('{}/{}, dt: {:0.4f}, eta: {}'.format(annind, len(annlist), dt, time_str))


                # concatenate single imdb into full imdb
                imdb += imdb_single_db

            imdb = np.array(imdb)

            # cache off the imdb?
            if cache_folder is not None:
                pickle_write(os.path.join(cache_folder, 'imdb.pkl'), imdb)

        # store more information
        self.datasets_train = conf.datasets_train
        self.len = len(imdb)
        self.imdb = imdb

        # setup data augmentation transforms
        self.transform = Augmentation(conf)

        # # setup sampler and data loader for this dataset
        # self.sampler = torch.utils.data.sampler.WeightedRandomSampler(balance_samples(conf, imdb), self.len)
        # self.loader = torch.utils.data.DataLoader(self, conf.batch_size, sampler=self.sampler, collate_fn=self.collate)

        # check classes
        cls_not_used = []
        for imobj in imdb:

            for gt in imobj.gts:
                cls = gt.cls
                if not(cls in conf.lbls or cls in conf.ilbls) and (cls not in cls_not_used):
                    cls_not_used.append(cls)

        if len(cls_not_used) > 0:
            logging.info('Labels not used in training.. {}'.format(cls_not_used))

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()



    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, i, side, do_flip)
        im = inputs[('color', 0, -1)]
        im = np.array(im)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)


        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        # transform / data augmentation
        im, imobj = self.transform(im, deepcopy(self.imdb[index]))

        for i in range(int(im.shape[2]/3)):
            # convert to RGB then permute to be [B C H W]
            im[:, :, (i*3):(i*3) + 3] = im[:, :, (i*3+2, i*3+1, i*3)]
        im = np.transpose(im, [2, 0, 1])

        return inputs, im, imobj

    @staticmethod
    def collate(batch):
        """
        Defines the methodology for PyTorch to collate the objects
        of a batch together, for some reason PyTorch doesn't function
        this way by default.
        """
        inputs = []
        imgs = []
        imobjs = []

        # go through each batch
        for sample in batch:
            # append images and object dictionaries
            inputs.append(sample[0])
            imgs.append(sample[1])
            imobjs.append(sample[2])

        # stack images
        imgs = np.array(imgs)
        imgs = torch.from_numpy(imgs)

        return inputs, imgs, imobjs

    def get_color(self, folder, frame_index, i, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError



def read_kitti_cal(calfile):
    """
    Reads the kitti calibration projection matrix (p2) file from disc.

    Args:
        calfile (str): path to single calibration file
    """

    text_file = open(calfile, 'r')

    p2pat = re.compile(('(P2:)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    for line in text_file:

        parsed = p2pat.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:
            p2 = np.zeros([4, 4], dtype=float)
            p2[0, 0] = parsed.group(2)
            p2[0, 1] = parsed.group(3)
            p2[0, 2] = parsed.group(4)
            p2[0, 3] = parsed.group(5)
            p2[1, 0] = parsed.group(6)
            p2[1, 1] = parsed.group(7)
            p2[1, 2] = parsed.group(8)
            p2[1, 3] = parsed.group(9)
            p2[2, 0] = parsed.group(10)
            p2[2, 1] = parsed.group(11)
            p2[2, 2] = parsed.group(12)
            p2[2, 3] = parsed.group(13)

            p2[3, 3] = 1

    text_file.close()

    return p2


def read_kitti_poses(posefile):

    text_file = open(posefile, 'r')

    ppat1 = re.compile(('(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                        '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*[Ee](?:[-+]?[\d]+)?'))

    ppat2 = re.compile(('(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)' +
                       '\s+(fpat)\s+(fpat)\s+(fpat)\s*\n').replace('fpat', '[-+]?[\d]+\.?[\d]*'));

    ps = []

    for line in text_file:

        parsed1 = ppat1.fullmatch(line)
        parsed2 = ppat2.fullmatch(line)

        if parsed1 is not None:
            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed1.group(1)
            p[0, 1] = parsed1.group(2)
            p[0, 2] = parsed1.group(3)
            p[0, 3] = parsed1.group(4)
            p[1, 0] = parsed1.group(5)
            p[1, 1] = parsed1.group(6)
            p[1, 2] = parsed1.group(7)
            p[1, 3] = parsed1.group(8)
            p[2, 0] = parsed1.group(9)
            p[2, 1] = parsed1.group(10)
            p[2, 2] = parsed1.group(11)
            p[2, 3] = parsed1.group(12)

            p[3, 3] = 1

            ps.append(p)

        elif parsed2 is not None:

            p = np.zeros([4, 4], dtype=float)
            p[0, 0] = parsed2.group(1)
            p[0, 1] = parsed2.group(2)
            p[0, 2] = parsed2.group(3)
            p[0, 3] = parsed2.group(4)
            p[1, 0] = parsed2.group(5)
            p[1, 1] = parsed2.group(6)
            p[1, 2] = parsed2.group(7)
            p[1, 3] = parsed2.group(8)
            p[2, 0] = parsed2.group(9)
            p[2, 1] = parsed2.group(10)
            p[2, 2] = parsed2.group(11)
            p[2, 3] = parsed2.group(12)

            p[3, 3] = 1

            ps.append(p)

    text_file.close()

    return ps


def read_kitti_label(file, p2, use_3d_for_2d=False):
    """
    Reads the kitti label file from disc.

    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = []

    text_file = open(file, 'r')

    '''
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    '''

    pattern = re.compile(('([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
                          + '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n')
                         .replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))


    for line in text_file:

        parsed = pattern.fullmatch(line)

        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            cls = parsed.group(1)
            trunc = float(parsed.group(2))
            occ = float(parsed.group(3))
            alpha = float(parsed.group(4))

            x = float(parsed.group(5))
            y = float(parsed.group(6))
            x2 = float(parsed.group(7))
            y2 = float(parsed.group(8))

            width = x2 - x + 1
            height = y2 - y + 1

            h3d = float(parsed.group(9))
            w3d = float(parsed.group(10))
            l3d = float(parsed.group(11))

            cx3d = float(parsed.group(12)) # center of car in 3d
            cy3d = float(parsed.group(13)) # bottom of car in 3d
            cz3d = float(parsed.group(14)) # center of car in 3d
            rotY = float(parsed.group(15))

            # actually center the box
            cy3d -= (h3d / 2)

            elevation = (1.65 - cy3d)

            if use_3d_for_2d and h3d > 0 and w3d > 0 and l3d > 0:

                # re-compute the 2D box using 3D (finally, avoids clipped boxes)
                verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

                # any boxes behind camera plane?
                if np.any(corners_3d[2, :] <= 0):
                    ign = True

                else:
                    x = min(verts3d[:, 0])
                    y = min(verts3d[:, 1])
                    x2 = max(verts3d[:, 0])
                    y2 = max(verts3d[:, 1])

                    width = x2 - x + 1
                    height = y2 - y + 1

            # project cx, cy, cz
            coord3d = p2.dot(np.array([cx3d, cy3d, cz3d, 1]))

            # store the projected instead
            cx3d_2d = coord3d[0]
            cy3d_2d = coord3d[1]
            cz3d_2d = coord3d[2]

            cx = cx3d_2d / cz3d_2d
            cy = cy3d_2d / cz3d_2d

            # encode occlusion with range estimation
            # 0 = fully visible, 1 = partly occluded
            # 2 = largely occluded, 3 = unknown
            if occ == 0: vis = 1
            elif occ == 1: vis = 0.66
            elif occ == 2: vis = 0.33
            else: vis = 0.0

            while rotY > math.pi: rotY -= math.pi * 2
            while rotY < (-math.pi): rotY += math.pi * 2

            # recompute alpha
            alpha = convertRot2Alpha(rotY, cz3d, cx3d)

            obj.elevation = elevation
            obj.cls = cls
            obj.occ = occ > 0
            obj.ign = ign
            obj.visibility = vis
            obj.trunc = trunc
            obj.alpha = alpha
            obj.rotY = rotY

            # is there an extra field? (assume to be track)
            if len(parsed.groups()) >= 16 and parsed.group(16).isdigit(): obj.track = int(parsed.group(16))

            obj.bbox_full = np.array([x, y, width, height])
            obj.bbox_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY]
            obj.center_3d = [cx3d, cy3d, cz3d]

            gts.append(obj)

    text_file.close()

    return gts


def balance_samples(conf, imdb):
    """
    Balances the samples in an image dataset according to the given configuration.
    Basically we check which images have relevant foreground samples and which are empty,
    then we compute the sampling weights according to a desired fg_image_ratio.

    This is primarily useful in datasets which have a lot of empty (background) images, which may
    cause instability during training if not properly balanced against.
    """

    sample_weights = np.ones(len(imdb))

    if conf.fg_image_ratio >= 0:

        empty_inds = []
        valid_inds = []

        for imind, imobj in enumerate(imdb):

            valid = 0

            scale = conf.test_scale / imobj.imH
            igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                           conf.min_gt_h, conf.max_gt_h, scale)

            for gtind, gt in enumerate(imobj.gts):

                if (not igns[gtind]) and (not rmvs[gtind]):
                    valid += 1

            sample_weights[imind] = valid

            if valid>0:
                valid_inds.append(imind)
            else:
                empty_inds.append(imind)

        if not (conf.fg_image_ratio == 2):
            fg_weight = len(imdb) * conf.fg_image_ratio / len(valid_inds)
            bg_weight = len(imdb) * (1 - conf.fg_image_ratio) / len(empty_inds)
            sample_weights[valid_inds] = fg_weight
            sample_weights[empty_inds] = bg_weight

            logging.info('weighted respectively as {:.2f} and {:.2f}'.format(fg_weight, bg_weight))

        logging.info('Found {} foreground and {} empty images'.format(np.sum(sample_weights > 0), np.sum(sample_weights <= 0)))

    # force sampling weights to sum to 1
    sample_weights /= np.sum(sample_weights)

    return sample_weights