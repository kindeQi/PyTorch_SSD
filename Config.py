class Config(object):
    def __init__(self, status):
        self.batch_size = 32
        self.learning_rate = 1e-3
        
        if status == 'remote':
            self.voc2007_root = '/home/kindeqi/PyTorch_SSD/dataset/VOCdevkit/VOC2007/JPEGImages/'
            self.voc2012_root = '/home/kindeqi/PyTorch_SSD/dataset/VOCdevkit/VOC2012/JPEGImages/'

            self.voc2012_trn_anno = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_train2012.json'
            self.voc2012_val_anno = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_val2012.json'

            self.voc2007_trn_anno = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_train2007.json'
            self.voc2007_val_anno = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_val2007.json'
            self.voc2007_test_anno = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_test2007.json'

            self.vgg_weight_path = '/home/kindeqi/.torch/models/vgg16-397923af.pth'
            self.vgg_reduced_weight_path = '/home/kindeqi/PyTorch_SSD/weights/vgg16_reducedfc.pth'
            self.trained_path = '/home/kindeqi/PyTorch_SSD/weights/ssd300_mAP_77.43_v2.pth'
            
            
        if status == 'local':
            self.batch_size = 4
            self.voc2007_root = 'C:\\datasets\\pascal\\VOC2007\\JPEGImages\\'
            self.voc2012_root = 'C:\\datasets\\pascal\\VOC2012\\JPEGImages\\'

            self.voc2007_trn_anno = 'C:\\datasets\\pascal\\PASCAL_VOC\\pascal_train2007.json'
            self.voc2007_val_anno = 'C:\\datasets\\pascal\\PASCAL_VOC\\pascal_val2007.json'

            self.voc2012_trn_anno = 'C:\\datasets\\pascal\\PASCAL_VOC\\pascal_train2012.json'
            self.voc2012_val_anno = 'C:\\datasets\\pascal\\PASCAL_VOC\\pascal_val2012.json'
            self.voc2007_test_anno = 'C:\\datasets\\pascal\\PASCAL_VOC\\pascal_test2007.json'

            self.vgg_weight_path = 'C:\\Users\\ruifr\\.torch\\models\\vgg16-397923af.pth'
            self.vgg_reduced_weight_path = 'D:\\1-usc\\SSD_3rd party implementation\\ssd.pytorch\\weights\\vgg16_reducedfc.pth'
            self.trained_path = 'D:\\1-usc\\SSD_3rd party implementation\\ssd.pytorch\\weights\\ssd300_mAP_77.43_v2.pth'
        
        self.anno_path = [self.voc2012_trn_anno, self.voc2012_val_anno, self.voc2007_trn_anno, self.voc2007_val_anno]
            
catagory_idx = {'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20}       

idx_catagory = {1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'}   