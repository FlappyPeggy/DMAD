import numpy as np
import glob
import cv2
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

class Test_Loss(nn.Module):
    def __init__(self, channels=1, ks=(16, 8), alpha=1):
        super(Test_Loss, self).__init__()
        self.alpha = alpha
        self.ks = ks
        self.c = channels
        self.filter = torch.ones((1, 1, ks[0], ks[1]), dtype=torch.float32).repeat(1, channels, 1, 1) / (ks[0] * ks[1])

    def forward(self, gen_frames):
        shape = gen_frames.size()
        b,w,h = shape[0], shape[-2], shape[-1]
        gen_frames = nn.functional.pad(gen_frames.abs().view(b,self.c, w,h), (self.ks[1], self.ks[1], self.ks[0], self.ks[0]))
        gen_dx = nn.functional.conv2d(gen_frames, self.filter).max()

        return gen_dx


def conf_avg(x, size=11, n_conf=5):
    a = x.copy()
    b=[]
    weight = np.array([0.5,0.8,0.8,1,1,2.5,1,1,0.8,0.8,0.5])
    if size != len(weight): weight = np.ones(size)

    for i in range(x.shape[0]-size+1):
        a_ = a[i:i+size].copy()
        u = a_.mean()
        dif = abs(a_ - u)
        sot = np.argsort(dif)[:n_conf]
        mask = np.zeros_like(dif)
        mask[sot] = 1
        weight_ = weight*mask
        b.append(np.sum(a_*weight_)/weight_.sum())
    for _ in range(size // 2):
        b.append(b[-1])
        b.insert(0, b[0])
    return b



r = 2
padding_size = 1
n_hist = 3
padding_size_offset = 1
n_hist_offset = 1
offset_alpha = 2
search_step = 1 # set this param <=1 for fixed value
prefix = "./exp/dif/"
skip_process = True
norm_factor_offset = None # 0.023


if __name__ == '__main__':
    m = np.load("./data/frame_labels_avenue.npy")
    print("label data contains ", len(m), " values")

    if skip_process:
        try:
            processed = np.load('./exp/res_list_avenue.npz')
            patch_res, offset_res = processed['patch_res'], processed['offset_res']
        except:
            print("No result file is available, please check your path or set skip_process=False")
    else:
        img_files = glob.glob(prefix+"*_0.jpg")
        offset_files = np.load("./exp/offset8.npy")

        img_files.sort(key=lambda x: int(x[len(prefix):-6]))
        diff, diff_old = None, [((cv2.resize(cv2.imread(img_files[i]), (int(256//r), int(256//r))).astype(np.float32) / 255) ** 2).max(axis=-1) for i in range(n_hist)]

        mean_res, patch_res, offset_res, grad_res = [], [], [], []
        get_err_offset = Test_Loss(ks=(int(15 // r), int(30 // r)))
        get_err_p = Test_Loss(ks=(int(42 // r), int(32 // r)))

        # check offset npy-file
        if norm_factor_offset is not None:
            if offset_files.dtype == np.int8 and np.abs(offset_files).max() > 120:
                print("you are setting norm_factor_offset=", norm_factor_offset,
                      ", but it seems you have load a npy-file with automatically-setted norm_factor")
            autoset = False
        elif offset_files.dtype == np.int8:
            autoset = True
        else:
            raise ValueError(
                "the npy-file must load as np.int8(norm_factor is set automatically in Evaluate_avenue or you have to set norm_factor_offset Manually)")
        
        # main processing
        for idx, name in enumerate(tqdm(img_files)):
            # Processing Reconstruction
            if diff is not None:
                for i in range(n_hist-1):
                    diff_old[-1-i] = diff_old[-2-i].copy()
                diff_old[0] = diff.copy()
            diff = ((cv2.resize(cv2.imread(name), (int(256//r), int(256//r))).astype(np.float32) / 255) ** 2).max(axis=-1)
            pads = [np.pad(diff_old[i], padding_size)[None] for i in range(n_hist)]
            pad_list = []
            # shift and cat the mat to remove the static anomalies
            for row in range(padding_size * 2 + 1):
                for col in range(padding_size * 2 + 1):
                    for pad in pads:
                        pad_list.append(pad[:, row:int(256//r)+row, col:int(256//r)+col])
            pad_arr = np.concatenate(pad_list, axis=0)
            diff_move = np.abs((pad_arr - diff[None])).min(0)[None, None]

            # Processing offset
            if idx<n_hist_offset:
                diff_offset_move = np.zeros((1,1,int(256//r), int(256//r)))
            else:
                x_cube = np.abs(offset_files[idx - n_hist_offset:idx + 1] / 127) ** offset_alpha if autoset else \
                    np.abs(offset_files[idx - n_hist_offset:idx + 1] / norm_factor_offset) ** offset_alpha

                if n_hist_offset:
                    x_cube_, y_cube_ = [], []
                    for x in x_cube:
                        x_cube_.append(cv2.resize(x.astype(np.float32), (int(256//r), int(256//r)))[None])
                    x_cube = np.concatenate(x_cube_, axis=0)
                    x_pad = np.pad(x_cube[:-1], ((0, 0), (padding_size_offset, padding_size_offset), (padding_size_offset, padding_size_offset)))
                    x_pad_ = []
                    for row in range(padding_size_offset * 2 + 1):
                        for col in range(padding_size_offset * 2 + 1):
                            x_pad_.append(x_pad[:, row:int(256 // r) + row, col:int(256 // r) + col])
                    x_pad = np.concatenate(x_pad_, axis=0)
                    diff_offset_move = np.abs((x_pad - x_cube[-1][None])).min(0)[None, None]
                else:
                    diff_offset_move = x_cube[None]

            patch_res.append(get_err_p.forward(torch.from_numpy(diff_move)).numpy())
            offset_res.append(get_err_offset.forward(torch.from_numpy(diff_offset_move).float()).numpy())

        np.savez('./conv_res.npz', patch_res=patch_res, offset_res=offset_res)

    arr_lists = [np.array(patch_res), np.array(offset_res)]
    arr_lists = np.concatenate([arr_list[:, None] for arr_list in arr_lists], axis=1)

    list1, list2 = arr_lists[:, 1], arr_lists[:, 0]
    list1 = (list1 - list1.min()) / (list1.max() - list1.min())
    list2 = (list2 - list2.min()) / (list2.max() - list2.min())
    list1, list2 = np.array(conf_avg(list1)), np.array(conf_avg(list2))
    best, recoder, recoder_param = 0, None, []
    m_ = m.copy()
    
    if search_step > 1:
        for i in tqdm(range(search_step + 1)):
            alpha = i / search_step
            temp = alpha * list1 + (1-alpha)*list2
            auc = roc_auc_score(y_true=m_, y_score=conf_avg(temp)) * 100
            if auc>best:
                best = auc
                recoder_param = [alpha, 1 - alpha]
    else:
        recoder_param = [0.2, 0.4, 0.6]
        auc = roc_auc_score(y_true=m_, y_score=conf_avg(recoder_param[0] * list1 + recoder_param[1]*list2)) * 100

    print("Avenue res: ", best, recoder_param)
