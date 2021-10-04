import argparse
from src.Mvtec_dataset import MVTecDataset
from src.utils import *
from train import Padim_model


def testing_parse_args():
    parser = argparse.ArgumentParser('test')
    parser.add_argument('--data_path', type=str, default='./MVTec')
    parser.add_argument('--class_name', type=str, default='bottle')
    parser.add_argument('--method', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def run():
    args = testing_parse_args()
    os.makedirs(f'./model/weight/{args.method}', exist_ok=True)
    load_filepath = f'./model/weight/{args.method}/train_{args.class_name}.pkl'
    model = Padim_model(method=args.method)
    if not os.path.exists(load_filepath):
        print("####" * 5)
        print("학습된 정보가 없습니다.")
        print("####" * 5)
        model.train(MVTecDataset('./MVTec', class_name=args.class_name, is_train=True), load_filepath)
    else:
        print("####"*5)
        print("학습된 정보가 있습니다.")
        print("####" * 5)
    result = load_weight(load_filepath)
    test_dataset = MVTecDataset('./MVTec', class_name=args.class_name, is_train=False)
    print("테스트를 진행합니다.")
    model.test(test_dataset, args.class_name,result,flag="each")

if __name__ == '__main__':
    run()























