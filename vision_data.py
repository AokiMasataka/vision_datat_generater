import os
import cv2
import random
import numpy as np


class Ann:
	def __init__(self, bbox: np.ndarray, mask: np.ndarray) -> None:
		self._bbox = bbox
		self._mask = mask
	
	@property
	def bbox(self) -> np.ndarray:
		return self._bbox
	
	@property
	def mask(self) -> np.ndarray:
		return self._mask


def rand_image(size: tuple[int]) -> np.ndarray:
	image = np.random.randint(
		low=0, high=255, size=size + (3, ), dtype=np.uint8
	)
	return image


def circle_data(image: np.ndarray, bbox: bool, mask: bool) -> (np.ndarray, Ann):
	w, h = image.shape[:2]
	xy = (random.randint(0, w), random.randint(0, h))
	wh = (random.randint(0, 128), random.randint(0, 128))
	
	color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	cv2.ellipse(image, xy, wh, 0.0, 0.0, 360.0, color, -1)

	if bbox:
		maked_bbox = np.array([
			0, xy[0] + wh[0] // 2, xy[1] + wh[1] // 2, xy[0] - wh[0] // 2, xy[1] - wh[1] // 2
		])
		maked_bbox = maked_bbox.reshape(-1, 5)
	else:
		maked_bbox = None

	if mask:
		maked_mask = np.zeros((w, h), dtype=np.uint8)
		cv2.ellipse(maked_mask, xy, wh, 0.0, 0.0, 360.0, (1), -1)
	else:
		maked_mask = None
	
	ann = Ann(bbox=maked_bbox, mask=maked_mask)
	return image, ann


def try_data(image: np.ndarray, bbox: bool, mask: bool) -> np.ndarray:
	w, h = image.shape[:2]
	pts = [
		[random.randint(0, w), random.randint(0, h)],
		[random.randint(0, w), random.randint(0, h)],
		[random.randint(0, w), random.randint(0, h)]
	]

	pts = np.array(pts).reshape((-1, 1, 2)).astype(np.int32)

	color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
	cv2.fillPoly(image, [pts], color, lineType=cv2.LINE_8, shift=0)

	if bbox:
		pts = pts.reshape(-1, 2)
		x_max = pts[:, 0].max()
		x_min = pts[:, 0].min()
		y_max = pts[:, 1].max()
		y_min = pts[:, 1].min()
		maked_bbox = np.array((1, x_max, y_max, x_min, y_min))
		maked_bbox = maked_bbox.reshape(-1, 5)
	else:
		maked_bbox = None
	
	if mask:
		pts = np.array(pts).reshape((-1, 1, 2))
		maked_mask = np.zeros((w, h), dtype=np.uint8)
		cv2.fillPoly(maked_mask, [pts], (2,), lineType=cv2.LINE_8, shift=0)
	else:
		maked_mask = None
	
	ann = Ann(bbox=maked_bbox, mask=maked_mask)
	return image, ann


def make_one_sample(size: tuple[int], bbox: bool, mask: bool) -> (np.ndarray, Ann):
	image = rand_image(size=size)

	if random.randint(0, 2) == 0:
		image, ann = circle_data(image=image, bbox=bbox, mask=mask)
	else:
		image, ann = try_data(image=image, bbox=bbox, mask=mask)
	
	return image, ann


def write_image(image: np.ndarray, path: str) -> None:
	cv2.imwrite(path, image)

def write_bbox(bbox: np.ndarray, path: str) -> None:
	string_bbox = '\n'.join([' '.join(list(map(str, b))) for b in bbox.tolist()])

	with open(path, 'w') as f:
		f.write(string_bbox)

def write_mask(mask: np.ndarray, path: str) -> None:
	cv2.imwrite(path, mask)


def write_one_sample(image: np.ndarray, ann: Ann, export: str, prefix: str) -> None:
	image_path = os.path.join(export, 'images', prefix + '.png')
	bbox_path = os.path.join(export, 'bbox', prefix + '.txt')
	mask_path = os.path.join(export, 'mask', prefix + '.png')

	write_image(image, path=image_path)

	if ann.bbox is not None:
		write_bbox(ann.bbox, path=bbox_path)
	
	if ann.mask is not None:
		write_mask(ann.mask, path=mask_path)

	print(f'save to {image_path}')


def make_write_samples(export: str, bbox: bool, mask: bool, size: int = 256, num_samples: int = 128) -> None:
	size = (size, size)

	os.makedirs(os.path.join(export, 'images'), exist_ok=True)
	os.makedirs(os.path.join(export, 'bbox'), exist_ok=True)
	os.makedirs(os.path.join(export, 'mask'), exist_ok=True)

	for i in range(num_samples):
		prefix = str(i).zfill(5)
		image, ann = make_one_sample(size=size, bbox=bbox, mask=mask)
		write_one_sample(image=image, ann=ann, export=export, prefix=prefix)
