import torch
import torchvision
from torchvision.ops import nms
import torchvision.transforms.functional as F
import timm
import torch


class CoinExtractor:
    def __init__(self, model_path):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=25)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
    def extract_bboxes(self,image,return_class=False):
        image = F.to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(image).pop()

            scores = prediction['scores'].to('cpu').numpy()
            bboxs = prediction['boxes'].to('cpu').numpy()
            indices = nms(prediction['boxes'], prediction['scores'], iou_threshold=0.3).to('cpu').numpy()
            mask = [i in indices for i in range(len(scores))]
            mask = mask & (scores > 0.4)
            filtered_bboxs = bboxs[mask]
            if return_class:
                class_names = open('data/frcnn_class_names.txt').read().split('\n')
                classes = prediction['labels'].to('cpu').numpy()
                filtered_classes = classes[mask]
                filtered_class_names = [class_names[c] for c in filtered_classes]
                filtered_scores = scores[mask]
                return filtered_class_names,filtered_bboxs,filtered_scores

            return filtered_bboxs
    def extract_coins(self, image, bboxes, size=None):
        coins = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if size is None:
                coin = image.crop((x1, y1, x2, y2))
            else:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                coin = image.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))
            coins.append(coin)
        return coins
    def extract(self, image):
        bboxes = self.extract_bboxes(image)
        coins = self.extract_coins(image, bboxes, size=900)
        return coins
    

## This class is used to classify the coins using the model trained in the previous step
class CoinClassifier:
    def __init__(self, model_path,class_names_path):

        with open(class_names_path) as f:
            self.class_names = f.read().splitlines()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # initialize the model
        self.model = timm.create_model('fastvit_ma36.apple_in1k', pretrained=False, num_classes=len(self.class_names))
        # load the model
        self.model.load_state_dict(torch.load(model_path,map_location=self.device))
        self.model = self.model.to(self.device)
        # get the data configuration
        data_config = timm.data.resolve_model_data_config(self.model)
        # get the transforms needed for the model
        self.transforms = timm.data.create_transform(**data_config, is_training=False)
        self.model.eval()

    def classify(self, image):
        # apply transforms to the image before passing it to the model
        image = self.transforms(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(image)
            score,index = torch.topk(prediction.softmax(dim=1) * 100, k=1)
            score = score.squeeze().item()
            index = index.squeeze().item()
            prediction = self.class_names[index]
            return prediction, score



