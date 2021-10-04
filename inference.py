import argparse
from src.Mvtec_dataset import MVTecDataset
from src.utils import *
from src.custom_dataset import *
from train import Padim_model



def inference_parse_args():
    parser = argparse.ArgumentParser('inference')
    parser.add_argument('--train_data_path', type=str,choices=['./MVTec', './archive/experiment/casting/train'], default='./MVTec')
    parser.add_argument('--test_data_path', type=str, default='./archive/experiment/casting/test')
    parser.add_argument('--class_name', type=str, default='casting')
    parser.add_argument('--method', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()



def inference_All_run(args):
    print("inference.py")
    MvTec_AD_Class_Name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    model = Padim_model(method=args.method)
    if(args.class_name in MvTec_AD_Class_Name):
        load_filepath = f'./model/weight/{args.method}/train_{args.class_name}.pkl'
        if not os.path.exists(load_filepath):
            print("---" * 10)
            print("학습된 정보가 없습니다.")
            print("---" * 10)
            model.train(MVTecDataset('./MVTec', class_name=args.class_name, is_train=True), load_filepath)
        else:
            print("---" * 10)
            print("학습된 정보가 있습니다.")
            print("---" * 10)
        result = load_weight(load_filepath)
        inferenc_dataset = CustomDataset_API(args.test_data_path)
        model.inference(inferenc_dataset, args.class_name, result, theshold=0.386)
    else:
        load_filepath = f'./model/weight/{args.method}/custom_{args.class_name}.pkl'
        if not os.path.exists(load_filepath):
            print("---" * 10)
            print("커스텀 데이터셋으로 훈련을 진행합니다.")
            print("---" * 10)
            model.train(CustomDataset_Console(dataset_path =args.train_data_path), save_filepath =load_filepath)
        else:
            print("---" * 10)
            print("학습된 정보가 있습니다.")
            print("---" * 10)
            result = load_weight(load_filepath)
            inferenc_dataset = CustomDataset_Console(args.test_data_path)
            model.inference(inferenc_dataset, args.class_name,result, theshold=0.386)


def inference_one(args):
    print("이미지 한개에 대한 테스트를 하는 API를 호출하였습니다.")
    model = Padim_model(method=args.method)

    if (args.class_name in args.MvTec_AD_Class_Name):
        load_filepath = f'./model/weight/{args.method}/train_{args.class_name}.pkl'
        if not os.path.exists(load_filepath):
            print("---" * 10)
            print("학습된 정보가 없습니다.")
            print("---" * 10)
            model.train(MVTecDataset('./MVTec', class_name=args.class_name, is_train=True), load_filepath)
        else:
            print("---" * 10)
            print("학습된 정보가 있습니다.")
            print("---" * 10)
        result = load_weight(load_filepath)
        print("Test 경로 : ",args.test_data_path)
        inferenc_dataset = CustomDataset_One_API(args.test_data_path)
        img_scores = model.inferenceONEofIMAGE(inferenc_dataset, args.class_name, result, threshold=args.threshold,dir_file_path= args.dir_file_path, filename=args.filename)

    else:
        load_filepath = f'./model/weight/{args.method}/custom_{args.class_name}.pkl'
        if not os.path.exists(load_filepath):
            print("---" * 10)
            print("커스텀 데이터셋으로 훈련을 진행합니다.")
            print("---" * 10)
            model.train(CustomDataset_Console(dataset_path=args.train_data_path), save_filepath=load_filepath)
        else:
            print("---" * 10)
            print("학습된 정보가 있습니다.")
            print("---" * 10)
        result = load_weight(load_filepath)
        print("Test 경로 : ", args.test_data_path)
        inferenc_dataset = CustomDataset_One_API(args.test_data_path,args.y_labels)
        img_scores = model.inferenceONEofIMAGE(inferenc_dataset, args.class_name, result, threshold=args.threshold,dir_file_path= args.dir_file_path, filename=args.filename)

    return img_scores

def inference_some(args):
    print("이미지 여러개에 대한 테스트를 하는 API를 호출하였습니다.")

    model = Padim_model(method=args.method)

    if (args.class_name in args.MvTec_AD_Class_Name):
        load_filepath = f'./model/weight/{args.method}/train_{args.class_name}.pkl'
        if not os.path.exists(load_filepath):
            print("---" * 10)
            print("학습된 정보가 없습니다.")
            print("---" * 10)
            model.train(MVTecDataset('./MVTec', class_name=args.class_name, is_train=True), load_filepath)
        else:
            print("---" * 10)
            print("학습된 정보가 있습니다.")
            print("---" * 10)
        result = load_weight(load_filepath)
        print("Test 경로 : ",args.test_data_path)
        inferenc_dataset = CustomDataset_Some_API(args.test_data_path)
        if(len(inferenc_dataset.y_count) ==2 ):
            img_roc_auc, pre_score, rec_score, f_score,save_dir = model.inferenceSOMEofIMAGE(inferenc_dataset, args.class_name, result, threshold=args.threshold,save_folder_name= args.save_folder_name)
        else:
            return "입력하신 데이터가 형식대로 입력되었는지 확인해주세요."
    else:
        load_filepath = f'./model/weight/{args.method}/custom_{args.class_name}.pkl'
        if not os.path.exists(load_filepath):
            print("---" * 10)
            print("커스텀 데이터셋으로 훈련을 진행합니다.")
            print("---" * 10)
            model.train(CustomDataset_Console(dataset_path=args.train_data_path), save_filepath=load_filepath)
        else:
            print("---" * 10)
            print("학습된 정보가 있습니다.")
            print("---" * 10)
        result = load_weight(load_filepath)
        print("Test 경로 : ", args.test_data_path)
        inferenc_dataset = CustomDataset_Some_API(args.test_data_path,args.y_labels)
        if (len(inferenc_dataset.y_count) == 2):
            img_roc_auc, pre_score, rec_score, f_score ,save_dir = model.inferenceSOMEofIMAGE(inferenc_dataset, args.class_name, result, threshold=args.threshold,save_folder_name= args.save_folder_name)
        else:
            return "입력하신 데이터가 형식대로 입력되었는지 확인해주세요."
    return img_roc_auc,pre_score, rec_score,f_score,save_dir



if __name__ == '__main__':
    args = inference_parse_args()

    inference_All_run(args)