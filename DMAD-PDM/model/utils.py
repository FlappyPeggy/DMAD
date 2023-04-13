import numpy as np
from collections import OrderedDict
import os
import glob
import cv2
import torch.utils.data as data
import sys
OS_NAME = sys.platform
SEP = '\\' if OS_NAME == "win32" else '/'

rng = np.random.RandomState(2020)

def np_load_frame(filename, resize_height, resize_width, c):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    if c == 1:
        image_decoded = np.repeat(cv2.imread(filename, 0)[:, :, None], 3, axis=-1)
    else:
        image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized




class DataLoader(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=1, c=1):
        self.c = c
        self.dir = video_folder
        self.transform = transform
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        
        
    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for idx, video in enumerate(sorted(videos)):
            video_name = video.split(SEP)[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['idx'] = idx
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])
            
            
    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split(SEP)[-1]
            for i in range(len(self.videos[video_name]['frame'])-self._time_step):
                frames.append(self.videos[video_name]['frame'][i])
                           
        return frames               
            
        
    def __getitem__(self, index):
        video_name = self.samples[index].split(SEP)[-2]
        frame_name = int(self.samples[index].split(SEP)[-1].split('.')[-2])

        batch = []
        for i in range(self._time_step+self._num_pred):
            image = np_load_frame(self.videos[video_name]['frame'][frame_name+i], self._resize_height, self._resize_width, self.c)
            if self.transform is not None:
                batch.append(self.transform(image))

        return np.concatenate(batch, axis=0),self.transform(np.array([[self.videos[video_name]['idx']]]))
        
        
    def __len__(self):
        return len(self.samples)
