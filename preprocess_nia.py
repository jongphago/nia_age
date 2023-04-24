import json
import os
import shutil

import cv2
import numpy as np
from glob import glob
from insightface.utils.face_align import norm_crop
from tqdm import tqdm


def remove_hangul(data_root):
    hangul_paths = glob(data_root + '*/*/')
    hangul_paths.sort()
    for hp in hangul_paths:
        target = hp[:hp[:-1].rfind('/') + 2] + '/'
        shutil.move(hp, target)


def crop_and_save_face_images(data_root, output_root):
    print('preprocessing the dataset (face crop)...')
    os.system('rm -rf ' + output_root)
    json_list = glob(data_root + '*/*/2.*/*.json') + glob(data_root + '*/*/3.*/*.json')

    # crop the images
    nia_kp_to_insight = [2, 0, 1, 3, 4]
    meta_list = []
    for i, jpath in enumerate(tqdm(json_list)):
        try:
            with open(jpath, 'r') as json_fid:
                temp_data = json.load(json_fid)
                json_dir = jpath[:jpath.rfind('/') + 1]
                if 'filename' not in temp_data.keys():
                    temp_data = temp_data[list(temp_data.keys())[0]]
                temp_data['filename'] = temp_data['filename'].replace('MMGM', 'MGM')
                img = cv2.imread(json_dir + temp_data['filename'])
                if isinstance(img, type(None)):
                    print('img file does not exist : {}'.format(jpath))
                    continue
                landmarks = np.zeros([5, 2])
                for kp in temp_data['member'][0]['regions'][0]['keypoint']:
                    kp_id = int(kp['idx'])
                    if kp_id >= 5:
                        continue
                    kp_id = nia_kp_to_insight[kp_id]
                    landmarks[kp_id, 0] = kp['x']
                    landmarks[kp_id, 1] = kp['y']

                warped_img = norm_crop(img, landmarks)
                output_dir = json_dir.replace(data_root, output_root)
                os.makedirs(output_dir, exist_ok=True)
                output_path = output_dir + temp_data['filename']
                cv2.imwrite(output_dir + temp_data['filename'], warped_img)

                img_dict = dict()
                if jpath.split('/')[3][0] == '2':
                    try:
                        img_dict['age'] = int(temp_data['member'][0]['age'])
                    except:
                        img_dict['age'] = int(temp_data['filename'].split('_')[3])

                    img_dict['age_class'] = None
                else:
                    filename = os.path.basename(jpath)
                    img_dict['age'] = None
                    img_dict['age_class'] = filename.split('_')[-1][0]
                img_dict['personal_id'] = temp_data['member'][0]['personal_id']
                img_dict['family_id'] = temp_data['family_id']
                img_dict['img_path'] = output_path
                meta_list.append(img_dict)
        except:
            print('error at {}'.format(jpath))

    np.save(output_root + 'all_meta.npy', meta_list)
    print(len(meta_list))


if __name__ == '__main__':
    raw_data_root = 'nia/'
    cropped_root = 'nia_cropped/'
    if not os.path.isdir(cropped_root):
        remove_hangul(raw_data_root)
        crop_and_save_face_images(raw_data_root, cropped_root)
