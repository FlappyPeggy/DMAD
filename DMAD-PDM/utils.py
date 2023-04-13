import numpy as np
import os
import random
import cv2
from collections import OrderedDict
import glob
from sklearn.metrics import roc_auc_score
from multiprocessing import Process
from tqdm import tqdm
from multiprocessing import Pool
import sys
OS_NAME = sys.platform
SEP = '\\' if OS_NAME == "win32" else '/'

n=100
temp = []
for i in range(n+1):
    for j in range(n+1-i):
        for k in range(n+1-i-j):
            temp.append([i,j,k,n -i-j-k])
temp = np.array(temp, dtype=np.float32) / n

def get_bkg(tosize=None, c=3):
    bkg_list = []
    print(sorted(glob.glob("./bkg/*")))
    for img in sorted(glob.glob("./bkg/*")):
        arr = np.repeat(cv2.imread(img, 0)[:, :, None], c, axis=-1)
        if tosize is not None:
            arr = cv2.resize(arr, (tosize,tosize))
        bkg_list.append(arr[None])
    
    for img in sorted(glob.glob("./bkg_/*")):
        arr = np.repeat(cv2.imread(img, 0)[:, :, None], c, axis=-1)
        if tosize is not None:
            arr = cv2.resize(arr, (tosize,tosize))
        bkg_list.append(arr[None])

    return np.concatenate(bkg_list, axis=0)[:,:,:,None] if c==1 else np.concatenate(bkg_list, axis=0)

def mp_get_bkg(root, save_path,n=4):
    print(save_path, os.listdir(".\\"))
    if save_path not in os.listdir(".\\"):
        os.mkdir(".\\"+save_path)
    save_path = ".\\"+save_path+"\\"
    patch_len = len(os.listdir(root))//n+1
    p = [Process(target=gen_bkg, args=(root, save_path, (i*patch_len,(i+1)*patch_len))) for i in range(n)]
    for sub_p in p:
        sub_p.start()


def gen_bkg(root, to_path, bound):
    size = 13
    std = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    kernel = np.arange(size) - size // 2
    kernel = np.exp(-kernel ** 2 / (2 * std ** 2)) / (std * (2 * np.pi) ** 0.5)
    print("processing the range: ", bound)
    for name in tqdm(os.listdir(root)[bound[0]: bound[1]]):
        bg = None
        all_path = glob.glob(root + "/" + name + "/*")
        frames = []
        bg = np.zeros((256,256))

        for i, img_path in enumerate(all_path):
            frames.append(cv2.resize(cv2.imread(img_path, 0).copy(), (256,256))[:,:,None])
        frames = np.concatenate(frames,axis=-1)

        for i in range(256):
            for j in range(256):
                cnt = np.zeros((256))
                for mu in range(256):
                    cnt[mu] = (frames[i,j]==mu).sum()
                bg[i,j] = np.argmax(np.convolve(cnt, kernel, 'same'))
        cv2.imwrite(to_path+name+".jpg", bg.astype(np.uint8))
    print("the range: ", bound, " has been processed")

def gen_bkg_(root, to_path, bound=None, hist=400,n_frame=500):
    for name in os.listdir(root):
        backSub = cv2.createBackgroundSubtractorMOG2()
        bg = None
        w = None
        all_path = glob.glob(root+"/"+name+"/*")
        random.shuffle(all_path)
        
        for i, img_path in enumerate(all_path):
            frame = cv2.imread(img_path).copy()
            if bg is None:
                bg, w = np.zeros_like(frame,dtype=np.float32), np.zeros_like(frame,dtype=np.float32)[:,:,0][:,:,None]
            if i < hist:
                backSub.apply(frame)
            else:
                break
        for i, img_path in enumerate(all_path):
            frame = cv2.imread(img_path).copy()
            if i < n_frame:
                bgMask = 1 - backSub.apply(frame).astype(np.float32)[:,:,None]/255
                bg+=bgMask*frame
                w += bgMask
            else:
                break
        cv2.imwrite(to_path+name+".png", (bg/w).astype(np.uint8))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr+1e-8)))

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    max_ele = np.max(psnr_list)
    min_ele = np.min(psnr_list)
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], max_ele, min_ele))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    try:
        frame_auc = roc_auc_score(y_true=labels, y_score=anomal_scores)
    except:
        frame_auc = roc_auc_score(y_true=labels, y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def sub_auc3(temp, list1, list2, list3, labels):
    rec = None
    maxauc = 0
    for comb in tqdm(temp):
        auc = roc_auc_score(y_true=labels, y_score=comb[0] * list1 + comb[1] * list2 + comb[2] * list3)
        if auc > maxauc:
            maxauc = auc
            rec = comb
    return maxauc, rec

def auc3_mp(list1, list2, list3, labels, weight=None):
    if weight is None:
        temp3 = []
        for i in range(n + 1):
            for j in range(n + 1 - i):
                temp3.append([i, j, n - i - j])
        temp3 = np.array(temp3, dtype=np.float32) / n
    else:
        weight = int(n * weight)
        temp3 = []
        for i in range(n - weight + 1):
            temp3.append([weight, i, n - i - weight])
        # for i in range(n + 1):
        #     temp3.append([weight, i, n - i])
        temp3 = np.array(temp3, dtype=np.float32) / n

    pool = Pool(8)
    length = len(temp3) // 8 +1
    res = []
    list1, list2, list3, labels = np.array(list1), np.array(list2), np.array(list3), np.array(labels)
    for i in range(8):
        res.append(pool.apply_async(sub_auc3, (temp3[length*i:length*(i+1)], list1, list2, list3, labels)))
    pool.close()
    pool.join()

    max_auc = 0
    rec = None
    for p in res:
        auc, hyp = p.get()
        if auc > max_auc:
            max_auc=auc
            rec = hyp

    return max_auc, rec

def sub_auc4(temp, list1, list2, list3, list4, labels):
    rec = None
    maxauc = 0
    for comb in tqdm(temp):
        auc = roc_auc_score(y_true=labels, y_score=comb[0] * list1 + comb[1] * list2 + comb[2] * list3 + comb[3] * list4)
        if auc > maxauc:
            maxauc = auc
            rec = comb
    return maxauc, rec

def auc4_mp(list1, list2, list3, list4, labels):
    pool = Pool(8)
    length = len(temp) // 8 +1
    res = []
    list1, list2, list3, list4, labels = np.array(list1), np.array(list2), np.array(list3), np.array(list4), np.array(labels)
    for i in range(8):
        res.append(pool.apply_async(sub_auc4, (temp[length*i:length*(i+1)], list1, list2, list3, list4, labels)))
    pool.close()
    pool.join()

    max_auc = 0
    rec = None
    for p in res:
        auc, hyp = p.get()
        if auc > max_auc:
            max_auc=auc
            rec = hyp

    return max_auc, rec


def conf_avg(x, size=69, n_conf="robust"):
    if n_conf == "robust":
        n_conf = size//2
    elif n_conf == "average":
        n_conf = int(size *0.9)
    elif n_conf == "robust+":
        n_conf = round(size * 0.36)
    assert isinstance(n_conf ,int)

    a = x.copy()
    b = np.ones_like(x)
    weight = np.ones(size)
    base = (size+1)//2

    for i in range(x.shape[0] - size+1):
        a_ = a[i:i + size].copy()
        u = a_.mean()
        dif = abs(a_ - u)
        sot = np.argsort(dif)[:n_conf]
        mask = np.zeros_like(dif)
        mask[sot] = 1
        weight_ = weight * mask
        b[i+base] = np.sum(a_ * weight_) / weight_.sum()

    return b.tolist()

def filter(a, y, test_folder, R=0.01):
    temp = a.copy()

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split(SEP)[-1]
        videos[video_name] = {}
        videos[video_name]['length'] = len(glob.glob(os.path.join(video, '*.jpg')))

    R = R ** 2
    finished_len = 0
    outputs = np.zeros(len(temp))

    for video in videos_list:
        video_name = video.split(SEP)[-1]
        temp_len = videos[video_name]['length'] - 4

        n_iter = temp_len
        sz = (n_iter,)  # size of array
        z = temp[finished_len:finished_len + temp_len].copy()

        Q = 1e-5
        xhat = np.zeros(sz)
        P = np.zeros(sz)
        xhatminus = np.zeros(sz)
        Pminus = np.zeros(sz)
        K = np.zeros(sz)
        xhat[0] = 0.0
        P[0] = 1.0

        for k in range(1, n_iter):
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]

        xhat[0] = z[0]
        outputs[finished_len:finished_len + temp_len] = xhat
        finished_len += temp_len

    return AUC(outputs, y)