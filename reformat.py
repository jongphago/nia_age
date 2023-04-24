from glob import glob
import os
from shutil import copyfile

for i in glob('./dlib_face_val/*.jpg'):
    filename = str(i)
    #print(filename)
    filename = filename.split('/')[-1]
    #print(filename)
    ids = filename.split('_')[0]
    gender = filename.split('_')[2]
    age = filename.split('_')[3] 
    img_num = filename.split('_')[5] 

    if gender== 'GM' or gender =='D' or gender == 'M' or gender == 'D2' or  gender == 'D3' or gender == 'GMG' or gender == 'MGM' or gender == 'FGM':
        gender = '1'
    else:
        gender = '0'
    print("filename", filename)
    target = './age_db_val/' + age + '_' + gender + '_' + ids + '_' + img_num
    print("target", target)
    copyfile(i, target)


