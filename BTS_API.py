from fastapi import FastAPI, UploadFile,File,Form
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse
import io
import cv2
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import zipfile
import datetime
from inference import *
from PIL import Image
import numpy as np
from io import BytesIO,StringIO
import aiofiles
from fastapi.responses import StreamingResponse
import io
from starlette.responses import FileResponse
import uuid


# -------------------------------------------------------------------------------------------------------#
# creating FastAPI app --
app = FastAPI()

# define output schema --
class Predictions(BaseModel):
    real_class: str = "비정상"
    anomaly_score: float = 0.8

    class Config:
        orm_mode = True
class OutputSchema(BaseModel):
    input_path: str = "xx.jpg"
    result_path: str = "result_xx.jpg"
    anomaly_score: List[Predictions]
    origin_Image: List[float] = []
    result_Image: List[float] = []

    class Config:
        orm_mode = True
# -------------------------------------------------------------------------------------------------------#
# user functions --
### Define Output Schema
def make_one_image_of_json(filename, result_filename, anomalyScore, y_labels):
    image_results = {
                        "input_path": filename,
                        "result_path": result_filename,
                        "anomaly_score": [
                                            {
                                                "real_class": "정상" if y_labels==0 else "비정상",
                                                "anomaly_score": str(int(anomalyScore[0])),
                                            }
                                        ],
                        "origin_Image": FileResponse(filename),
                        # "origin_Image": [origin_Image],
                        "result_Image":  FileResponse(result_filename),
                        # "result_Image": [result_Image],
                    }
    return image_results
# -------------------------------------------------------------------------------------------------------#
def make_some_image_of_json(filename, result_filename, img_roc_auc,pre_score, rec_score,f_score):
    image_results = {
                        "input_path": filename,
                        "result_path": result_filename,
                        "score": [
                                    {
                                        "ROC_AUC Score": round(img_roc_auc,3),
                                        "Precision Score":round(pre_score,3),
                                        "Recall Score":round(rec_score),
                                        "F1 Score" :round(f_score),
                                    }
                                ],
                    }
    return image_results
# -------------------------------------------------------------------------------------------------------#
# API

def make_folder(flag =True):
    """
    :param flag: zip 파일을 받는다면 True, Zip이 아니라 사진 한장에 대해서면 False에 따라 디렉토리를 생성한다.
    :return:
        저장할 디렉토리 명 uploadfile/년월일_시간정보/
    """
    if flag ==True:
        dir_file_path = "./upload_file"
        dir_image_path = "./upload_image"
        os.makedirs(dir_file_path, exist_ok=True)
        os.makedirs(dir_image_path, exist_ok=True)

        today = datetime.datetime.today()
        save_folder_name = str(today).split(".")[0].replace(" ", "_").replace(":", "_")
        dir_file_path = dir_file_path + "/" + save_folder_name + "/"
        dir_image_path = dir_image_path + "/" + save_folder_name + "/"
        os.makedirs(dir_file_path, exist_ok=True)
        os.makedirs(dir_image_path, exist_ok=True)
        return dir_file_path, dir_image_path,save_folder_name
    else:
        dir_file_path = "./upload_one_file"
        os.makedirs(dir_file_path, exist_ok=True)
        today = datetime.datetime.today()
        save_folder_name = str(today).split(".")[0].replace(" ", "_").replace(":", "_")
        dir_file_path = dir_file_path + "/" + save_folder_name + "/"
        os.makedirs(dir_file_path, exist_ok=True)
        return dir_file_path

class config():
    def __init__(self):
        self.MvTec_AD_Class_Name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
                               'pill','screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        self.train_data_path = ""
        self.test_data_path = ""
        self.y_labels = ""
        self.threshold = ""
        self.class_name = ""
        self.method = ""
        self.dir_file_path =""
        self.filename = ""

    def setParam(self,dir_file_path, filename, className,y_labels,threshold,modelName):
        if (className in self.MvTec_AD_Class_Name):
            print("MvTec 데이터 셋 로드")
            self.train_data_path = './MVTec'
            self.test_data_path = os.path.join(dir_file_path, filename)

        elif (className in ["casting"]):
            print("캐스팅 데이터 셋 로드")
            self.train_data_path = './archive/experiment/casting/train'
            self.test_data_path = os.path.join(dir_file_path, filename)
        else:
            print("클래스를 잘못 입력했습니다.")

        self.y_labels = int(y_labels)
        self.threshold = float(threshold)
        self.class_name = className
        self.method = modelName
        self.dir_file_path = dir_file_path
        self.filename = filename

class config2():
    def __init__(self):
        self.MvTec_AD_Class_Name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
                               'pill',
                               'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        self.train_data_path = ""
        self.test_data_path = ""
        self.threshold = ""
        self.class_name = ""
        self.method = ""
        self.dir_image_path =""
        self.save_folder_name = ""

    def setParam(self,dir_image_path, className,threshold,modelName,save_folder_name):
        if (className in self.MvTec_AD_Class_Name):
            self.train_data_path = './MVTec'
            self.test_data_path = dir_image_path

        elif (className in ["casting"]):
            self.train_data_path = './archive/experiment/casting/train'
            self.test_data_path = dir_image_path
        else:
            print("클래스를 잘못 입력했습니다.")

        self.threshold = float(threshold)
        self.class_name = className
        self.method = modelName
        self.dir_image_path = dir_image_path
        self.save_folder_name = save_folder_name

@app.get('/onefile/result_image')
async def get_one_result_Image(result_filename : str = Form(...)):
    return FileResponse(result_filename)


# -------------------------------------------------------------------------------------------------------#
# async def get_one_result(file: bytes  = File(...), y_labels:str = None, threshold:str = None, className:str=None, modelName:str="wide_resnet50_2",fileName:str=None):
# async def get_one_result(item: Item22, file: UploadFile = File(...)):
@app.post('/onefile/')
async def get_one_result(y_labels: str = Form(...), threshold:str = Form(...), className: str = Form(...), modelName:str = Form(...), fileName:str = Form(...),  file: UploadFile = File(...)):
    """
        files: 이미지 한장을 업로드를 받는 인자.\n
        y_labels: 이미지의 정상 비정상 여부의 라벨을 받는 인자\n
        threshold: threshold 값을 받는 인자.\n
        className: 선택한 클래스 이름을 받는 인자\n
        modelName: 모델 이름 선택\n
    :return:
        JSON 파일로 리턴을 받는데, \n
        형식은 OutputSchema 형태와 유사하지만, 약간 변형해서 batch 별로 데이터를 업로드 했을때,\n
        그 데이터 갯수 만큼 결과값을 반환하도록 수정하였다.
    """
    print("fileName : {},  y_labels : {}  threshold : {}, className : {}, modelName : {}".format(fileName,y_labels,threshold,className,modelName))

    dir_file_path = make_folder(flag=False) ## 파라미터 상관없이 사진을 저장할 폴더를 생성한다.
    #
    # ## 업로드한 파일 읽고 dir_file_path 경로의 폴더에 저장한다.
    # contents = await file.read()
    file_bytes = file.file.read()
    image = Image.open(io.BytesIO(file_bytes))
    filename1 = os.path.join(dir_file_path, fileName)
    image.save(filename1)

    # return {"Good":"OK"}

    args = config()
    args.setParam(dir_file_path=dir_file_path, filename=fileName, className=className, y_labels=y_labels, threshold=threshold, modelName=modelName)
    #
    anomalyScore = inference_one(args)
    # filename = os.path.join(dir_file_path,fileName)
    result_filename = os.path.join(dir_file_path,"result_"+ fileName)
    image_results = make_one_image_of_json(fileName, result_filename, anomalyScore, y_labels)
    #
    # print(f"FileName : {file.filename}  y_label : {y_labels}  threshold : {threshold} className : {className}  modelName:{modelName}")
    return image_results

# -------------------------------------------------------------------------------------------------------#

@app.post('/somefile')
async def get_some_result(file:UploadFile = File(...), threshold:str = 0.38, className:str='bottle', modelName:str="wide_resnet50_2"):
    """
        files: Zip을 업로드를 받는 인자.\n
        y_labels: 이미지의 정상 비정상 여부의 라벨을 받는 인자\n
        threshold: threshold 값을 받는 인자.\n
        className: 선택한 클래스 이름을 받는 인자\n
        modelName: 모델 이름 선택\n
    :return:
        JSON 파일로 리턴을 받는데, \n
        형식은 OutputSchema 형태와 유사하지만, 약간 변형해서 batch 별로 데이터를 업로드 했을때,\n
        그 데이터 갯수 만큼 결과값을 반환하도록 수정하였다.
    """
    dir_file_path, dir_image_path, save_folder_name = make_folder(flag=True)

    ## 업로드한 파일 읽고 dir_file_path 경로의 폴더에 저장한다.
    contents = await file.read()
    with open(os.path.join(dir_file_path, file.filename), "wb") as fp:
        fp.write(contents)

    suvey_zip = zipfile.ZipFile(os.path.join(dir_file_path, file.filename))
    suvey_zip.extractall(dir_image_path)
    suvey_zip.close()

    args = config2()
    args.setParam(dir_image_path=dir_image_path, className=className, threshold=threshold, modelName=modelName,save_folder_name=save_folder_name)

    img_roc_auc,pre_score, rec_score,f_score,save_dir = inference_some(args)

    image_results = make_some_image_of_json(filename=file.filename, result_filename="result.zip", img_roc_auc=img_roc_auc, pre_score=pre_score, rec_score= rec_score, f_score=f_score)

    print("SAVE DIR : ",save_dir+"/"+"result.zip")
    return image_results, FileResponse(save_dir+"/"+"result.zip", media_type='application/x-zip-compressed', filename="result.zip")

#    return image_results

# -------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    ## 두가지 기능이 되도록 설계
    ## 첫번째 기능은 한개의 이미지를 입력하고 그에 대한 결과를 보여주는 기능
    ## 두번째 기능은 여러개의 이미지를 ZIP 파일로 넣고 그에 대한 성능과 결과를 ZIP 파일로 내보낸다.
    print("start API  Service")
    uvicorn.run(app, host="0.0.0.0", port=3677)