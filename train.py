import torch
import random
from tqdm import tqdm
import zipfile
from random import sample
from collections import OrderedDict
from torch.utils.data import DataLoader
import argparse
from scipy.ndimage import gaussian_filter


from src.Mvtec_dataset import MVTecDataset
from model.Model_Zoo import Pretrain_model
from src.utils import *

def train_parse_args():
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--data_path', type=str, default='./MVTec')
    parser.add_argument('--class_name', type=str, default='bottle')
    parser.add_argument('--method', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()

class Padim_model:
    def __init__(self,method ="resnet18"):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.method = method
        self.backbone, self.t_d, self.d = Pretrain_model(method=self.method).call()
        self.idx = torch.tensor(sample(range(0, self.t_d), self.d))
        self.outputs = []
        self.backbone.to(self.device).eval()

        self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 10))
        self.fig_img_rocauc = self.ax[0]
        self.fig_pixel_rocauc = self.ax[1]
        self.total_roc_auc = []
        self.total_pixel_roc_auc = []

        random.seed(1024)
        torch.manual_seed(1024)
        if self.use_cuda:
            torch.cuda.manual_seed_all(1024)

        def hook(module, input, output):
            self.outputs.append(output)

        self.backbone.layer1[-1].register_forward_hook(hook)
        self.backbone.layer2[-1].register_forward_hook(hook)
        self.backbone.layer3[-1].register_forward_hook(hook)

    def train(self, train_dataset, save_filepath):
        self.backbone.eval()
        train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)
        train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for (x, _, _) in tqdm(train_dataloader, '| Train feature extraction |'):
            with torch.no_grad():
                _ = self.backbone(x.to(self.device))

            for k, v in zip(train_outputs.keys(), self.outputs):
                train_outputs[k].append(v.cpu().detach())

            self.outputs = []
        # print("훈련 : ",train_outputs)

        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        embedding_vectors = path_embedding_vector(train_outputs,self.idx)
        embedding_vectors, train_outputs = embedding_vector_calc(embedding_vectors,self.idx)
        save_weight(save_filepath, train_outputs)
        print("---"*10)
        print("Weight 저장 완료.")
        print("---"*10)

    def test(self,test_dataset, class_name, result, flag="each"):
        gt_list = []
        gt_mask_list = []
        test_imgs = []
        self.idx = result[-1]
        self.backbone.eval()
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)
        test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for (x, y, mask) in tqdm(test_dataloader, '| Test feature extraction |'):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            with torch.no_grad():
                _ = self.backbone(x.to(self.device))

            for k, v in zip(test_outputs.keys(), self.outputs):
                test_outputs[k].append(v.cpu().detach())

            self.outputs = []

        for k, v in test_outputs.items():
            test_outputs[k] = torch.cat(v, 0)

        img_size = x.size(2)

        embedding_vectors = path_embedding_vector(test_outputs, self.idx)
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

        score_map = computation_of_anomaly_map(img_size,B, H, W, result, embedding_vectors)

        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # 정규화
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)

        gt_list = np.asarray(gt_list)
        gt_mask = np.asarray(gt_mask_list)

        img_fpr, img_tpr, img_roc_auc = calc_image_level_ROC(gt_list, img_scores)
        pix_fpr, pix_tpr, per_pixel_rocauc = calc_pixel_level_ROC(gt_mask, scores)
        threshold = get_optimal_theshold(gt_mask, scores)
        print(threshold)
        threshold_cl = get_optimal_theshold(gt_list, img_scores)
        print("분류에서 구한 쓰레시 홀더 : ", threshold_cl)
        print("픽셀에서 구한 쓰레시 홀더 : ", threshold)

        self.fig_img_rocauc.plot(img_fpr, img_tpr, label='%s ROCAUC: %.3f' % (class_name, img_roc_auc))
        self.fig_pixel_rocauc.plot(pix_fpr, pix_tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        if(flag != "All"):

            self.fig_img_rocauc.title.set_text(f'{class_name} image ROCAUC : {img_roc_auc:.3f}')
            self.fig_pixel_rocauc.title.set_text(f'{class_name} pixel ROCAUC : {per_pixel_rocauc:.3f}')
            self.fig.tight_layout()
            self.fig.savefig(os.path.join('./result/', f'{class_name}_roc_curve.png'), dpi=100)
        else:
            self.total_roc_auc.append(img_roc_auc)
            self.total_pixel_roc_auc.append(per_pixel_rocauc)

        save_dir = './result' + '/' + f'{self.method}'
        os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir + '/' + class_name  # Class location in folder
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    def All_plot_show(self):
        print("#######"*5)
        print()
        print("\t Describe \t")
        print('Average ROCAUC: %.3f' % np.mean(self.total_roc_auc))
        self.fig_img_rocauc.title.set_text(f'Average image ROCAUC: {np.mean(self.total_roc_auc):.3f}')
        self.fig_img_rocauc.legend(loc="lower right")

        print('Average pixel ROCUAC: %.3f' % np.mean(self.total_pixel_roc_auc))
        self.fig_pixel_rocauc.title.set_text(f'Average pixel ROCAUC: {np.mean(self.total_pixel_roc_auc):.3f}')
        self.fig_pixel_rocauc.legend(loc="lower right")

        self.fig.tight_layout()
        self.fig.savefig(os.path.join('./result', 'MvTec AD roc_curve.png'), dpi=100)
        print()
        print("#######" * 5)

    def inference(self, inference_dataset, class_name, result,theshold):
        plt.figure(figsize=(10,10))
        gt_list = []
        gt_mask_list = []
        inference_imgs = []
        self.idx = result[-1]
        self.backbone.eval()
        inference_dataloader = DataLoader(inference_dataset, batch_size=32, pin_memory=True)
        inference_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for (x, y, mask) in tqdm(inference_dataloader, '| inference feature extraction |'):
            inference_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            with torch.no_grad():
                _ = self.backbone(x.to(self.device))

            for k, v in zip(inference_outputs.keys(), self.outputs):
                inference_outputs[k].append(v.cpu().detach())

            self.outputs = []

        for k, v in inference_outputs.items():
            inference_outputs[k] = torch.cat(v, 0)

        img_size = x.size(2)

        embedding_vectors = path_embedding_vector(inference_outputs, self.idx)
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

        score_map = computation_of_anomaly_map(img_size, B, H, W, result, embedding_vectors)

        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # 정규화
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  # img_scores -> (sample 수)
        # print("이미지 anomaly score 의 shape ", img_scores.shape)

        gt_list = np.asarray(gt_list)
        gt_mask = np.asarray(gt_mask_list)

        img_fpr, img_tpr, img_roc_auc = calc_image_level_ROC(gt_list, img_scores)
        # pix_fpr, pix_tpr, per_pixel_rocauc = calc_pixel_level_ROC(gt_mask, scores)
        threshold_cl = get_optimal_theshold(gt_list, img_scores)
        print("분류에서 구한 쓰레시 홀더 : ",threshold_cl)
        threshold = threshold_cl

        plt.plot(img_fpr, img_tpr, label='%s ROCAUC: %.3f' % (class_name, img_roc_auc))
        # self.fig_pixel_rocauc.plot(pix_fpr, pix_tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        plt.title(f'{class_name} image ROCAUC : {img_roc_auc:.3f}')
        # self.fig_pixel_rocauc.title.set_text(f'{class_name} pixel ROCAUC : {per_pixel_rocauc:.3f}')
        # self.fig.tight_layout()
        plt.savefig(os.path.join('./result/', f'{class_name}_roc_curve.png'), dpi=100)


        save_dir = './custom/result' + '/' + f'{self.method}'
        os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir + '/' + class_name  # Class location in folder
        os.makedirs(save_dir, exist_ok=True)
        plot_inference_fig(inference_imgs, scores, threshold, save_dir, class_name,gt_list, img_scores)


    def inferenceONEofIMAGE(self, inference_dataset, class_name, result,threshold,dir_file_path, filename):
        gt_list = []
        gt_mask_list = []
        inference_imgs = []
        self.idx = result[-1]
        self.backbone.eval()
        inference_dataloader = DataLoader(inference_dataset, batch_size=32, pin_memory=True)
        inference_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for (x, y, mask) in tqdm(inference_dataloader, '| inference feature extraction |'):
            print(f"Test에 대한 이미지 shape의 크기는 : {x.shape}")
            inference_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            with torch.no_grad():
                _ = self.backbone(x.to(self.device))

            for k, v in zip(inference_outputs.keys(), self.outputs):
                inference_outputs[k].append(v.cpu().detach())

            self.outputs = []

        for k, v in inference_outputs.items():
            inference_outputs[k] = torch.cat(v, 0)

        img_size = x.size(2)

        embedding_vectors = path_embedding_vector(inference_outputs, self.idx)
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

        score_map = computation_of_anomaly_map(img_size, B, H, W, result, embedding_vectors)

        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # 정규화
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        #img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  # img_scores -> (sample 수)
        img_scores = scores.reshape(1, -1).max(axis=1)  # img_scores -> (sample 수)
        gt_list = np.asarray(gt_list)

        save_dir = os.path.join(dir_file_path, "result_"+ filename)
        plot_inferenceONE_fig(inference_imgs, scores, threshold , save_dir, class_name, gt_list, img_scores)
        return img_scores


    def inferenceSOMEofIMAGE(self, inference_dataset, class_name, result,threshold,save_folder_name):
        plt.figure(figsize=(10,10))
        gt_list = []
        gt_mask_list = []
        inference_imgs = []
        self.idx = result[-1]
        self.backbone.eval()
        inference_dataloader = DataLoader(inference_dataset, batch_size=32, pin_memory=True)
        inference_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

        for (x, y, mask) in tqdm(inference_dataloader, '| inference feature extraction |'):
            inference_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())

            with torch.no_grad():
                _ = self.backbone(x.to(self.device))

            for k, v in zip(inference_outputs.keys(), self.outputs):
                inference_outputs[k].append(v.cpu().detach())

            self.outputs = []

        for k, v in inference_outputs.items():
            inference_outputs[k] = torch.cat(v, 0)

        img_size = x.size(2)

        embedding_vectors = path_embedding_vector(inference_outputs, self.idx)
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()

        score_map = computation_of_anomaly_map(img_size, B, H, W, result, embedding_vectors)

        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # 정규화
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)  # img_scores -> (sample 수)
        # print("이미지 anomaly score 의 shape ", img_scores.shape)

        gt_list = np.asarray(gt_list)
        gt_mask = np.asarray(gt_mask_list)

        img_fpr, img_tpr, img_roc_auc = calc_image_level_ROC(gt_list, img_scores)
        # pix_fpr, pix_tpr, per_pixel_rocauc = calc_pixel_level_ROC(gt_mask, scores)
        threshold_cl, [pre_score, rec_score,f_score] = get_optimal_theshold2(gt_list, img_scores)
        print("분류에서 구한 쓰레시 홀더 : ",threshold_cl)
        print("입력한 구한 쓰레시 홀더 : ", threshold)
        # threshold = threshold_cl
        threshold = threshold

        plt.plot(img_fpr, img_tpr, label='%s ROCAUC: %.3f' % (class_name, img_roc_auc))
        # self.fig_pixel_rocauc.plot(pix_fpr, pix_tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        plt.title(f'{class_name} image ROCAUC : {img_roc_auc:.3f}')
        # self.fig_pixel_rocauc.title.set_text(f'{class_name} pixel ROCAUC : {per_pixel_rocauc:.3f}')
        # self.fig.tight_layout()
        plt.savefig(os.path.join('./result/', f'{class_name}_roc_curve.png'), dpi=100)


        save_dir = "./response_result/"
        os.makedirs(save_dir, exist_ok=True)
        save_dir = save_dir + '/' + save_folder_name
        os.makedirs(save_dir, exist_ok=True)
        plot_inference_fig(inference_imgs, scores, threshold, save_dir, class_name,gt_list, img_scores)

        print("압축 중....")
        resultZip = zipfile.ZipFile(save_dir + '/result.zip', 'w')
        for file in os.listdir(save_dir+"/"):
            if file.endswith('.png'):
                resultZip.write(os.path.join(save_dir,file), compress_type=zipfile.ZIP_DEFLATED)

        resultZip.close()

        return img_roc_auc,pre_score, rec_score,f_score,save_dir

if __name__ == '__main__':
    args = train_parse_args()
    model = Padim_model(method =args.method)
    save_filepath = f'./model/weight/train_{args.class_name}.pkl'
    model.train(MVTecDataset(args.data_path, class_name=args.class_name, is_train=True),save_filepath)