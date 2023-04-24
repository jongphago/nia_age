import sys

sys.path.append("nets/")
from main_ae import END_AGE, START_AGE, AgeModel, NUM_AGE_GROUPS
import torchvision
import cv2
import numpy as np
from insightface import model_zoo
from insightface.utils.face_align import norm_crop
import torch
from torch import nn


class AgeEstimator(object):
    def __init__(self,
                 detection_model_path: str,
                 age_model_path: str,
                 detector_threshold=0.5,
                 detector_scale=0.5):
        det_model_root = detection_model_path[:detection_model_path.rfind('/') + 1]
        det_model_name = detection_model_path[detection_model_path.rfind('/') + 1:]
        self.detector = model_zoo.get_model(det_model_name, root=det_model_root)
        self.detector.prepare(ctx_id=-1, nms=0.4)
        self.detector_threshold = detector_threshold
        self.detector_scale = detector_scale

        model = AgeModel(END_AGE - START_AGE + 1, NUM_AGE_GROUPS)
        model.cuda()
        model.load_state_dict(torch.load(age_model_path))
        model.eval()
        self.age_model = model

        # transform
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])

    def detect_face(self, img_path):
        img = cv2.imread(str(img_path))
        bbox, landmarks = self.detector.detect(img, threshold=self.detector_threshold,
                                               scale=self.detector_scale)  # 0.5 is original
        if len(landmarks) < 1:
            print(f'cannot find a face from {img_path}')
            return False, None
        warped_img = norm_crop(img, landmarks[0])
        return True, warped_img

    def inference(self, image_file_path: str):
        success, warped = self.detect_face(image_file_path)
        if not success:
            return False, None, None
        warped = self.transform(warped)
        warped = warped.unsqueeze(0).cuda()
        output, _ = self.age_model(warped)
        # print(output)
        m = nn.Softmax(dim=1)
        output_softmax = m(output)
        # print(output_softmax)
        a = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
        mean = (output_softmax * a).sum(1, keepdim=True).cpu().data.numpy()
        age = np.around(mean)[0][0]

        return success, age


if __name__ == "__main__":
    ae = AgeEstimator(detection_model_path='models/retinaface_r50_v1',
                      age_model_path='result_model/model_0')
    success, age = ae.inference('nia_cropped/F0001/B/2.Individuals/F0001_IND_D_18_45_CAM.jpg')
    print(success, age)
