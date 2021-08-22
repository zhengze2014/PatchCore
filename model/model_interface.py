import os
import cv2
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
from .utils import  prep_dirs, embedding_concat, save_anomaly_map, reshape_embedding
from sklearn.random_projection import SparseRandomProjection
from .sampling_methods.kcenter_greedy import kCenterGreedy
from .metrics import KNN
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from torchvision import transforms

class MInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super(MInterface, self).__init__()
        self.coreset_sampling_ratio = kwargs['coreset_sampling_ratio']
        self.input_size = kwargs['input_size']
        self.category  = kwargs['category']
        self.init_features()


        def hook_t(modulem, input, output):
            self.features.append(output)
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])


    def forward(self, x_t):
        self.init_features()
        _=self.model(x_t)
        return self.features


    def init_features(self):
        self.features = []


    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []


    def configure_optimizers(self):
        return None

    # Training
    def on_train_start(self):
        self.model.eval()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, self.category)
        self.embedding_list = []

    def training_step(self, batch, batch_idx):  # save locally aware patch features
        x, _, _, file_name, _ = batch

        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))

    def training_epoch_end(self, outputs):
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto',
                                                      eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[],
                                             N=int(total_embeddings.shape[0] * self.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]

        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
            pickle.dump(self.embedding_coreset, f)

    # Testing
    def on_test_start(self):
        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir, self.category)

    def test_step(self, batch, batch_idx):  # Nearest Neighbour Search
        self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
        x, gt, label, file_name, x_type = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
        knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(), k=9)
        score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()

        anomaly_map = score_patches[:, 0].reshape((28, 28))
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_patches[:, 0])  # Image-level score

        gt_np = gt.cpu().numpy()[0, 0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (self.input_size, self.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        self.gt_list_px_lvl.extend(gt_np.ravel())
        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        save_anomaly_map(self.sample_path,anomaly_map_resized_blur, input_x, gt_np * 255, file_name[0], x_type[0])

    def test_epoch_end(self, outputs):
        print("Total pixel-level auc-roc score :")
        pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
        print(pixel_auc)
        print("Total image-level auc-roc score :")
        img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
        print(img_auc)
        print('test_epoch_end')
        values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(values)