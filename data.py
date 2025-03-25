import json
import os
import random
from glob import glob
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from PIL import Image
from scipy.io.matlab import loadmat
from torch.utils.data import Dataset
from tqdm import trange, tqdm

from temporal import preset_temp


def circular_mask(kernel_size: int):
    """
    :param kernel_size:
    :return: masked image with slope with width 1?
    """
    radius = kernel_size / 2 - 0.5 # 8.5 DP
    x = torch.linspace(-radius, radius, kernel_size) # 18 vals from -8.5 to 8.5 spaced by 1.0 DP
    y = torch.linspace(-radius, radius, kernel_size)
    xx, yy = torch.meshgrid(x, y) 
    # more DP notes on how mask works:
        # xx: each row has one val from the vector above repeated 18 times. for ex, first row: -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000, -8.5000  
        # yy: each row has the regular original vector. For example: -8.5000, -7.5000, -6.5000, -5.5000, -4.5000, -3.5000, -2.5000, -1.5000, -0.5000,  0.5000,  1.5000,  2.5000,  3.5000,  4.5000,  5.5000,  6.5000, 7.5000,  8.5000
        # basically, xx is yy.t and vice versa 
        # xxyy matrix consists of two matrices [xx,yy] 
        # xx[ij] is x coordinate 
        # yy[ij] is y coordinate 
        # torch.sqrt(xx ** 2 + yy ** 2) is calculating dist/length for each component of xx and yy 
        # this 
        # mask = torch.clamp(radius - torch.sqrt(xx ** 2 + yy ** 2) + 1, min=0.0001, max=1) 
        # takes radius (8.5) subtracted by dist + 1
        # example radius (8.5) - (dist(xxyy[0,0]) = 12.02) + 1
        # ~ -2.52 
        # so, now you take resulting value and see where it is in the clamp from 0.0001 to 1 
        # mask = torch.clamp(-2.52 , min=0.0001, max=1) 
        # results in tensor(1.0000e-04) or 0.0001
        # if vals < min (0.0001), they clamp to min
        # if vals > max (1), they clamp to max 
        # if vals in between min and max, they stay the same

        # purpose: get full coordinate system of kernel and see if vals fall w/in or out of radius
        # change them accordingly so values near center are 1 and values towards the edge are near 0
        # applying these clamped values multiply corresponding kernel value by 1 or 0.0001 so they are the same or made smaller 
        # creates a circle where values diffuse from the center
        # see demo code of mask in simple_DoG_model script

    mask = torch.clamp(radius - torch.sqrt(xx ** 2 + yy ** 2) + 1, min=0.0001, max=1)
    return mask

# DP comments: 
    # index auto set to 0 
    # calculates covariance of a given segment using preset num of samples (100000 here)
    # so each call to dataset[index] returns a different randomly cropped, flipped, masked and processed segment from the video, not the entire video.
def estimated_covariance(dataset: Dataset, num_samples: int, device: Union[str, torch.device] = None, index=0):
    loop = trange(num_samples, desc="Taking samples for covariance calculation", ncols=99)
    samples = torch.stack([dataset[index].flatten() for _ in loop])  # / dataset.mask
    if device is not None:
        samples = samples.to(device)
    samples -= samples.mean(dim=0, keepdim=True) # mean shift
    C = samples.t() @ samples / num_samples # X @ XT / number of samples
    C = (C + C.t()) / 2.0  # make it numerically symmetric
    return C

# need to go back and verify that the data is grayscaled and has all changes jun uses - also see dif of using ffmpeg DP
class VideoDataset(Dataset):
    def __init__(self,
                 root: str,
                 kernel_size: Union[int, Tuple[int, int]],
                 frames: int,
                 circle_masking: bool,
                 group_size: Optional[int],
                 random_flip: bool): 
        # list of tuples with  [video,      mean,   std]    DP
        self.videos: List[Tuple[np.ndarray, float, float]] = []

        # setting up vars DP
        if isinstance(kernel_size, int):
            self.mask = circular_mask(kernel_size) if circle_masking else torch.ones((kernel_size, kernel_size))
        else:
            self.mask = torch.ones([kernel_size[0], kernel_size[1]])
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        self.frames = frames
        self.group_size = group_size
        self.random_flip = random_flip

        # grabbing npy videos DP
        files = sorted(glob(f"{root}/*.npy"))
        if "FILENAME_PREFIX" in os.environ:
            files = [file for file in files if os.path.basename(file).startswith(os.environ["FILENAME_PREFIX"])]
            print(f"{len(files)} files after filtering for prefix: {os.environ['FILENAME_PREFIX']}")
        self.files = [os.path.basename(f) for f in files]

        if "FILENAME_PREFIX_NOT" in os.environ:
            files = [file for file in files if not os.path.basename(file).startswith(os.environ["FILENAME_PREFIX"])]
            print(f"{len(files)} files after filtering out for prefix: {os.environ['FILENAME_PREFIX']}")

        assert len(files) > 0, f"no .npy files found under directory {root}"

        stats = json.load(open(f"{root}/stats.json"))
        # stats = json.load(open(f"{root}/stats-short800.json"))

        for path in tqdm(files, desc="Loading video info"):
            stat = stats[os.path.basename(path)]
            video = np.load(path, mmap_mode="r")
            self.videos.append((video, stat["mean"], stat["std"]))

    def __len__(self):
        return 128000  # any large number should work, since we don't care about "epoch" for now

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.files.index(index)
        # if none, it grabs random index
        elif self.group_size is None:
            index = np.random.choice(len(self.videos))
        else:
            if index % self.group_size == 0:
                random.shuffle(self.videos)
            index = 0

        video, mean, std = self.videos[index]

        # this section is creating the bounds for a segment: begin, end, top, bottom, left, and right DP
        # begin = np.random.choice(video.shape[0] - self.frames) 
        # video shape is (np.array, mean, std); index 0 = actual movie (Frames, 512, 512) DP
        # chooses random frame from: frame count from movie - 20, DP 
        # this is the starting index DP
        begin = np.random.choice(min(800, video.shape[0]) - self.frames)
        # ending index = start index + 20 frames DP
        end = begin + self.frames
        # getting top of frame boundary: video.shape[1]: 512 - self.kernel_size[0]: 18 = 494 DP 
        # np.random.choice(492) takes in that number and generates a random choice of a 3 digit number DP
        top = np.random.choice(video.shape[1] - self.kernel_size[0])
        # takes resulting random number and adds 18 DP
        bottom = top + self.kernel_size[0]
        left = np.random.choice(video.shape[2] - self.kernel_size[1])
        right = left + self.kernel_size[1]

        # normalizes the segment by subtracting mean and dividing by std
        segment = (video[begin:end, top:bottom, left:right].astype(np.float32) - mean) / std
        if self.random_flip:
            # this will never be true - maybe they were just checking for a bug? DP
            if np.random.rand() > 1.1:
                segment = segment[:1, :, :].repeat(segment.shape[0], axis=0)
            else:
                # horizontal flip if rand val less than 0.5 (probability based flip) DP
                if np.random.rand() < 0.5:
                    segment = segment[:, ::-1, :] 
                # vertical flip if cond met DP
                if np.random.rand() < 0.5:
                    segment = segment[:, :, ::-1]
        # creating a copy since orig data is read only - now stored in memory as tensor DP
        return torch.from_numpy(segment.copy()) * self.mask

    def covariance(self, num_samples: int = 100000, device: Union[str, torch.device] = None, index=0):
        return estimated_covariance(self, num_samples, device, index)


class FilteredVideoDataset(Dataset):
    def __init__(self,
                 root: str,
                 kernel_size: Union[int, Tuple[int, int]],
                 frames: int,
                 circle_masking: bool,
                 group_size: Optional[int],
                 random_flip: bool,
                 neural_type: Optional[str],
                 input_noise: Optional[float]):

        self.temporal_filter = preset_temp(neural_type).reshape(-1, 1, 1)
        self.videos: List[Tuple[np.ndarray, float, float]] = []
        self.input_noise = input_noise

        if isinstance(kernel_size, int):
            self.mask = circular_mask(kernel_size) if circle_masking else torch.ones((kernel_size, kernel_size))
        else:
            self.mask = torch.ones([kernel_size[0], kernel_size[1]])
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        self.frames = frames
        self.group_size = group_size
        self.random_flip = random_flip

        files = sorted(glob(f"{root}/*.npy"))
        if "FILENAME_PREFIX" in os.environ:
            files = [file for file in files if os.path.basename(file).startswith(os.environ["FILENAME_PREFIX"])]
            print(f"{len(files)} files after filtering for prefix: {os.environ['FILENAME_PREFIX']}")
        self.files = [os.path.basename(f) for f in files]

        if "FILENAME_PREFIX_NOT" in os.environ:
            files = [file for file in files if not os.path.basename(file).startswith(os.environ["FILENAME_PREFIX"])]
            print(f"{len(files)} files after filtering out for prefix: {os.environ['FILENAME_PREFIX']}")

        assert len(files) > 0, f"no .npy files found under directory {root}"

        stats = json.load(open(f"{root}/stats.json"))
        # stats = json.load(open(f"{root}/stats-short800.json"))

        for path in tqdm(files, desc="Loading video info"):
            stat = stats[os.path.basename(path)]
            video = np.load(path, mmap_mode="r")
            self.videos.append((video, stat["mean"], stat["std"]))

    def __len__(self):
        return 128000  # any large number should work, since we don't care about "epoch" for now

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.files.index(index)
        elif self.group_size is None:
            index = np.random.choice(len(self.videos))
        else:
            if index % self.group_size == 0:
                random.shuffle(self.videos)
            index = 0

        video, mean, std = self.videos[index]

        # begin = np.random.choice(video.shape[0] - self.frames)
        begin = np.random.choice(min(800, video.shape[0]) - self.frames)
        end = begin + self.frames
        top = np.random.choice(video.shape[1] - self.kernel_size[0])
        bottom = top + self.kernel_size[0]
        left = np.random.choice(video.shape[2] - self.kernel_size[1])
        right = left + self.kernel_size[1]

        segment = (video[begin:end, top:bottom, left:right].astype(np.float32) - mean) / std
        segment += self.input_noise * np.random.randn(*segment.shape)
        if self.random_flip:
            if np.random.rand() > 1.1:
                segment = segment[:1, :, :].repeat(segment.shape[0], axis=0)
            else:
                if np.random.rand() < 0.5:
                    segment = segment[:, ::-1, :]
                if np.random.rand() < 0.5:
                    segment = segment[:, :, ::-1]

        return (torch.from_numpy(segment.copy()) * self.mask * self.temporal_filter).sum(dim=0, keepdim=False)

    def covariance(self, num_samples: int = 100000, device: Union[str, torch.device] = None, index=0):
        return estimated_covariance(self, num_samples, device, index)


class MultivariateGaussianDataset(Dataset):
    """
    A dataset sampling from a multivariate gaussian distribution instead of real images
    """

    def __init__(self, covariance: torch.Tensor):
        assert covariance.dim() == 2 and covariance.shape[0] == covariance.shape[1]
        self.D = covariance.shape[0]
        self.C = covariance
        self.L = self.C.cholesky().to(device="cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return 100000

    def __getitem__(self, index):
        return self.L @ torch.randn(self.D, device=self.L.device)

    def covariance(self, num_samples: int = 100000, device: Union[str, torch.device] = None):
        covariance = self.C
        if device is not None:
            covariance = covariance.to(device)
        return covariance


class KyotoNaturalImages(Dataset):
    """
    A Torch Dataset class for reading the Kyoto Natural Image Dataset, available at:
        https://github.com/eizaburo-doi/kyoto_natim
    This dataset consists of 62 natural images in the MATLAB format, and each image has
    either 500x640 or 640x500 size, from which a rectangular patch is randomly extracted.
    """

    def __init__(self, root, kernel_size, circle_masking, device='cuda'):

        files = [mat for mat in os.listdir(root) if mat.endswith('.mat')]
        print("Loading {} images from {} ...".format(len(files), root))

        images = []
        for file in tqdm(files):
            if file.endswith('.mat'):
                image = loadmat(os.path.join(root, file))['OM'].astype(np.float)
            else:
                image = np.array(Image.open(os.path.join(root, file)).convert('L')).astype(np.float)

            std = np.std(image)
            if std < 1e-4:
                continue
            image -= np.mean(image)
            image /= std
            images.append(torch.from_numpy(image).to(device))

        self.device = device
        self.images = images
        self.kernel_size = kernel_size

        if isinstance(kernel_size, int):
            self.mask = circular_mask(kernel_size) if circle_masking else torch.ones((kernel_size, kernel_size))
        else:
            self.mask = torch.ones([kernel_size[0], kernel_size[1]])
        self.mask = self.mask.to(device)

    def __len__(self):
        """Returns 100 times larger than the number of images, to enable batches larger than 62"""
        return len(self.images) * 100

    def __getitem__(self, index):
        """Slices an [dx, dy] image at a random location"""
        while True:
            index = np.random.randint(len(self.images))
            image = self.images[index]
            dx, dy = self.kernel_size, self.kernel_size
            x = np.random.randint(image.shape[-2] - dx)
            y = np.random.randint(image.shape[-1] - dy)
            result = image[..., x:x+dx, y:y+dy] * self.mask
            return result.float()

    def covariance(self, num_samples: int = 100000, device: Union[str, torch.device] = None, index=0):
        return estimated_covariance(self, num_samples, device, index)


def get_dataset(data: str, kernel_size: Union[int, Tuple[int, int]], frames: int, circle_masking: bool,
                group_size: Optional[int], random_flip: bool, neural_type: Optional[str], input_noise: Optional[float]):
    # if data == "pink":
    #     dataset = VideoDataset("palmer", kernel_size, frames, circle_masking, group_size, random_flip)
    #     covariance = dataset.covariance()
    #     return MultivariateGaussianDataset(covariance)
    # elif data == "pink_tempfilter":
    #     dataset = FilteredVideoDataset("palmer", kernel_size, frames, circle_masking, group_size, random_flip, neural_type, input_noise)
    #     covariance = dataset.covariance()
    #     return MultivariateGaussianDataset(covariance)
    # elif data == "real_tempfilter":
    #     return FilteredVideoDataset("palmer", kernel_size, frames, circle_masking, group_size, random_flip, neural_type, input_noise)
    # elif data == "kyoto":
    #     return KyotoNaturalImages("kyoto", kernel_size, circle_masking, device='cuda')
    if data == "pink":
        dataset = VideoDataset("palmer_full", kernel_size, frames, circle_masking, group_size, random_flip)
        covariance = dataset.covariance()
        return MultivariateGaussianDataset(covariance)
    elif data == "pink_tempfilter":
        dataset = FilteredVideoDataset("palmer_full", kernel_size, frames, circle_masking, group_size, random_flip, neural_type, input_noise)
        covariance = dataset.covariance()
        return MultivariateGaussianDataset(covariance)
    elif data == "real_tempfilter":
        return FilteredVideoDataset("palmer_full", kernel_size, frames, circle_masking, group_size, random_flip, neural_type, input_noise)
    elif data == "kyoto":
        return KyotoNaturalImages("kyoto", kernel_size, circle_masking, device='cuda')

    return VideoDataset(data, kernel_size, frames, circle_masking, group_size, random_flip)
