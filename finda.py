import os

# print(os.listdir("./MVTec/"))
# MvTec_AD_Class_Name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill',
#                        'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# name = 'bottle'
# if(name in MvTec_AD_Class_Name):
#     print("있습")
# else:
#     print("압습")

names = ["../archive/experiment/casting product/train\\cast_ok_0_7609.jpeg"]

# print(name.split("_")[1])

# print([0 if (name.split("_")[1] =="ok") else 1 for name in names])
#
# import matplotlib.pyplot as plt
#
# fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
# ax_img[0].title.set_text('Image')
# ax_img[1].title.set_text('Predicted heat map')
# ax_img[2].title.set_text('Predicted mask')
# ax_img[3].title.set_text('Segmentation result')
#
# plt.suptitle(f"{cl}")
# plt.show()
#
# import datetime,os
# today = datetime.datetime.today()
# save_folder_name = str(today).split(".")[0].replace(" ","_")
# print(today)
# print(today.year)
# print(today.month)
# print(today.day)
# print(today.time().minute)
# print(str(today).split(".")[0].replace(" ","_"))
#
# dir_image_path = "./upload_image"
#     os.makedirs(dir_file_path,exist_ok=True)
#     os.makedirs(dir_image_path,exist_ok=True)
#
#     today = datetime.datetime.today()
#     save_folder_name = str(today).split(".")[0].replace(" ", "_").replace(":", "_")
#     dir_file_path = dir_file_path+"/"+save_folder_name+"/"
#     dir_image_path = dir_image_path+"/"+save_folder_name+"/"
#     os.makedirs(dir_file_path, exist_ok=True)
#     os.makedirs(dir_image_path, exist_ok=True)
#
#     contents = await file.read()
#     with open(os.path.join(dir_file_path, file.filename), "wb") as fp:
#         fp.write(contents)
#
#     suvey_zip = zipfile.ZipFile(os.path.join(dir_file_path, file.filename))
#     suvey_zip.extractall(dir_image_path)
#     suvey_zip.close()

# class config():
#     def __init__(self):
#         self.MvTec_AD_Class_Name = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut',
#                                'pill',
#                                'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
#         self.train_data_path = ""
#         self.test_data_path = ""
#         self.y_labels = ""
#         self.threshold = ""
#         self.class_name = ""
#         self.method = ""
#
#     def setParam(self,dir_file_path, filename, className,y_labels,threshold,modelName):
#         if (className in self.MvTec_AD_Class_Name):
#             self.train_data_path = './MVTec'
#             self.test_data_path = os.path.join(dir_file_path, filename)
#
#         elif (className in ["casting"]):
#             self.train_data_path = './archive/experiment/casting/train'
#             self.test_data_path = os.path.join(dir_file_path, filename)
#         else:
#             print("클래스를 잘못 입력했습니다.")
#
#         self.y_labels = y_labels
#         self.threshold = threshold
#         self.class_name = className
#         self.method = modelName
#
# args = config()
# args.setParam("d","d","casting","d","d","d")
# print(args.train_data_path)

"""
import csv
import zipfile
from io import BytesIO, StringIO
from fastapi.responses import StreamingResponse

zipped_file = BytesIO()
with zipfile.ZipFile(zipped_file, 'a', zipfile.ZIP_DEFLATED) as zipped:
    csv_data = StringIO()
    writer = csv.writer(csv_data, delimiter=',')
    writer.writerow(["test", "data"])
    csv_data.seek(0)
    csv_buffer = csv_data.read()
    zipped.writestr(f"test_data.csv", csv_buffer)
zipped_file.seek(0)
response = StreamingResponse(zipped_file, media_type="application/x-zip-compressed")
response.headers["Content-Disposition"] = "attachment; filename=test.zip"
return response
"""

# image = BytesIO()
# result_Image.save(image, format='JPEG', quality=85)
# image.seek(0)
# return StreamingResponse(image.read(), media_type="image/jpeg")
# return FileResponse(result_filename)

# fantasy_zip = zipfile.ZipFile('C:\\Stories\\Fantasy\\archive.zip')
# fantasy_zip.extractall('C:\\Library\\Stories\\Fantasy')
#
# fantasy_zip.close()

dataset_path = "./upload_file"
M = os.listdir(dataset_path)
dt = {}
for m in M:
    dt[m] = m.upper()

print(dt)