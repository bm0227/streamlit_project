import argparse
from src.Mvtec_dataset import MVTecDataset
from src.utils import *
from train import Padim_model


def All_testing_parse_args():
    parser = argparse.ArgumentParser('All_mvtec_test')
    parser.add_argument('--data_path', type=str, default='./MVTec')
    parser.add_argument('--method', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    return parser.parse_args()


def All_run():
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                   'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    args = All_testing_parse_args()
    os.makedirs(f'./model/weight/{args.method}', exist_ok=True)
    model = Padim_model(method=args.method)

    for class_name in CLASS_NAMES:
        load_filepath = f'./model/weight/{args.method}/train_{class_name}.pkl'
        if not os.path.exists(load_filepath):
            print("####" * 5)
            print("학습된 정보가 없습니다.")
            print("####" * 5)
            model.train(MVTecDataset('./MVTec', class_name=class_name, is_train=True), load_filepath)
        else:
            print("####"*5)
            print("학습된 정보가 있습니다.")
            print("####" * 5)
        result = load_weight(load_filepath)
        test_dataset = MVTecDataset('./MVTec', class_name=class_name, is_train=False)
        print("테스트를 진행합니다.")
        model.test(test_dataset, class_name,result,flag="All")

    model.All_plot_show()

if __name__ == '__main__':
    All_run()























