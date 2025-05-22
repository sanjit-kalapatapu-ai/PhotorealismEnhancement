import logging
from pathlib import Path

import imageio
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch

from .batch_types import EPEBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, normalize_dim

NUM_APPLIED_SEMANTIC_CLASSES = 34

def center(x, m, s):
	for i in range(x.shape[0]):
		x[i,:,:] = (x[i,:,:] - m[i]) / s[i]
	return x

def material_from_gt_label(gt_labelmap):
	""" Merges several classes. """
	w,h = gt_labelmap.shape
	shader_map = np.zeros((w, h, 12), dtype=np.float32)
	shader_map[:,:,0] = (gt_labelmap == 22).astype(np.float32) # sky
	shader_map[:,:,1] = (np.isin(gt_labelmap, [6, 7, 8, 30])).astype(np.float32) # road / static / sidewalk
	shader_map[:,:,2] = (np.isin(gt_labelmap, [10])).astype(np.float32) # vehicle
	shader_map[:,:,3] = (np.isin(gt_labelmap, [27, 28])).astype(np.float32) # terrain
	shader_map[:,:,4] = (np.isin(gt_labelmap, [9])).astype(np.float32) # vegetation
	shader_map[:,:,5] = (np.isin(gt_labelmap, [4, 32])).astype(np.float32) # person
	shader_map[:,:,6] = (np.isin(gt_labelmap, [2, 5, 23, 24])).astype(np.float32) # infrastructure
	shader_map[:,:,7] = (gt_labelmap == 33).astype(np.float32) # traffic light
	shader_map[:,:,8] = (gt_labelmap == 12).astype(np.float32) # traffic sign
	shader_map[:,:,9] = (gt_labelmap == 10).astype(np.float32) # ego vehicle
	shader_map[:,:,10] = (np.isin(gt_labelmap, [1, 11])).astype(np.float32) # building
	shader_map[:,:,11] = (np.isin(gt_labelmap, [0, 3, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 29, 31])).astype(np.float32) # unlabeled
	return shader_map


class AppliedSyntheticDataset(SyntheticDataset):
	def __init__(self, paths, transform=None, gbuffers='fake'):
		"""


		paths -- list of tuples with (img_path, robust_label_path, gbuffer_path, gt_label_path)
		"""

		super(AppliedSyntheticDataset, self).__init__('GTA')

		# assert gbuffers in ['all', 'img', 'no_light', 'geometry', 'fake']

		self.transform = transform
		self.gbuffers  = gbuffers
		# self.shader    = class_type

		self._paths    = paths
		self._path2id  = {p[0].stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass

		# Get dataset directory from first path
		dataset_dir = Path(paths[0][0]).parent.parent if paths else None

		try:
			# Look for stats file in the dataset directory
			stats_file = dataset_dir / 'gbuffer_stats.npz' if dataset_dir else None
			self._log.debug(stats_file)
			if stats_file and stats_file.exists():
				data = np.load(stats_file)
				self._gbuf_mean = data['g_m']
				self._gbuf_std  = data['g_s']
				self._log.info(f'Loaded dataset stats from {stats_file}')
			else:
				self._gbuf_mean = None
				self._gbuf_std  = None
				self._log.warning(f'No gbuffer_stats.npz found in dataset directory: {dataset_dir}')
		except Exception as e:
			self._gbuf_mean = None
			self._gbuf_std  = None
			self._log.warning(f'Failed to load dataset stats: {e}')
			pass

		# Compute number of g-buffer channels once during initialization
		first_gbuffer_path = self._paths[0][2]  # Assuming index 2 contains g-buffer paths
		gbuffer = np.load(first_gbuffer_path)
		self._num_gbuffer_channels = gbuffer.shape[-1]
		self._log.info(f'G-buffer has {self._num_gbuffer_channels} channels')

		self._log.info(f'Found {len(self._paths)} samples.')
		pass


	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		# return 1
		return self._num_gbuffer_channels

	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return NUM_APPLIED_SEMANTIC_CLASSES


	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				0:lambda g:g[:,15:21,:,:]}
		else:
			return {}


	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):

		index  = index % self.__len__()
		img_path, robust_label_path, gbuffer_path, gt_label_path = self._paths[index]
		
        # Load image
		img = mat2tensor(imageio.imread(img_path).astype(np.float32) / 255.0)

        # Load gbuffers and center using mean and std
		# TODO Sanjit: are the dimensions of the gbuffer correct?
		# Slightly concerned from this comment: https://github.com/isl-org/PhotorealismEnhancement/issues/60#issuecomment-1913178238  
		gbuffers = np.load(gbuffer_path)
		gbuffers = np.transpose(gbuffers, (2, 1, 0))
		# gbuffers = gbuffers[:1,:,:]
		gbuffers = torch.from_numpy(gbuffers.astype(np.float32)).float()
		if self._gbuf_mean is not None:
			gbuffers = center(gbuffers, self._gbuf_mean, self._gbuf_std)
			pass

        # Load gt_labels, map to expected material classes, and potentially normalize
		gt_labels = np.load(gt_label_path).astype(np.float32)
		
		# We aren't using the below method for squashing the classes to 12 classes, and instead are using the raw Applied semantic classes
		# gt_labels = mat2tensor(material_from_gt_label(gt_labels))

		# Convert to one-hot encoded format
		w, h = gt_labels.shape
		one_hot = np.zeros((w, h, NUM_APPLIED_SEMANTIC_CLASSES), dtype=np.float32)
		for i in range(NUM_APPLIED_SEMANTIC_CLASSES):
			one_hot[:, :, i] = (gt_labels == i).astype(np.float32)
		gt_labels = mat2tensor(one_hot)

		gt_labels = np.transpose(gt_labels, (0, 2, 1))

		if gt_labels is not None and torch.max(gt_labels) > 128:
			gt_labels = gt_labels / 255.0
			pass

        # Load robust labels (computed by mapping applied gt semseg labels to mseg taxonomy)
		robust_labels = imageio.imread(robust_label_path)
		robust_labels = torch.LongTensor(robust_labels[:,:]).unsqueeze(0)

		self._log.debug(f"AppliedSyntheticDataset: EPEBatch: Img shape: {img.shape}, G-buffers shape: {gbuffers.shape}, GT labels shape: {gt_labels.shape}, Robust labels shape: {robust_labels.shape}")
		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)
