# PaDim
Full name : a Patch Distribution Modeling Framework for Anomaly Detection and Localization

## First MvTec AD dataset Downloads. (Window Version)
1. 링크(https://github.com/aria2/aria2/releases/tag/release-1.35.0)에 접속해 [aria2-1.35.0-win-64bit-build1.zip] 파일을 설치한다.
2. 설치 완료 후 압축 파일을 해제한다.
3. CMD 창을 열어 해당 파일 경로로 접속한다.
4. Mvtec AD url(https://www.mvtec.com/company/research/datasets/mvtec-ad
)로 접속한다.
5. 해당 Url에서 Download를 클릭한다.
6. ftp url로 제공되는데 ftp를 제외한 url을 복사한다.
7. CMD 창에 접속해 aria2.exe url(복사한 url)을 붙여 넣는다.
8. MVTec DATA라는 폴더명을 MvTec으로 변경한다.

## Second Make a folder
1. 파이썬 터미널 Or CMD에 접속한다.
2. os.mkdir("model/weight")
3. os.mkdir("model/weight/resnet18")
4. os.mkdir("model/weight/wide_resnet50_2")

## Third execute train.py
    ## train.py
    python train.py --data_path=./MvTec --class_name=bottle --method=resnet18
    
## Four Test Dataset
    ## ALL Mvtec Data test
    python All_mvtec_test.py --data_path=./MvTec --method=resnet18

    ## test.py
    python test.py --data_path=./MvTec --class_name=bottle --method=resnet18
     
## Casting Dataset 
    python inference.py --train_data_path=./archive/experiment/casting/train --test_data_path=./archive/experiment/casting/test --class_name=casting --method=wide_resnet50_2

### referance
* https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
* https://github.com/PeterKim1/paper_code_review/tree/master/11.%20PaDiM


## FastAPI
* http://127.0.0.1:3677/docs#/default/get_one_result_onefile_post