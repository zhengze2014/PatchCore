import os
import cv2
import torch
import shutil
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import confusion_matrix


def copy_files(src, dst, ignores=None):
    if ignores is None:
        ignores = []
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst, file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)


def prep_dirs(root, category):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join('./', 'embeddings', category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    copy_files('./', source_code_save_path,
               ['.git', '.vscode', '__cache__', 'logs', 'README', 'samples', 'LICENSE'])  # copy source code
    return embeddings_path, sample_path, source_code_save_path


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)


def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)


def save_anomaly_map(sample_path,anomaly_map, input_img, gt_img, file_name, x_type):
    if anomaly_map.shape != input_img.shape:
        anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
    anomaly_map_norm = min_max_norm(anomaly_map)
    anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm*255)
    hm_on_img = heatmap_on_image(heatmap, input_img)

    # save images
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}.jpg'), input_img)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
    cv2.imwrite(os.path.join(sample_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)


