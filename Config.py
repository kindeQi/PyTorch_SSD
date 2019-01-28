class Config(object):
    def __init__(self, status):
        self.batch_size = 16
        self.learning_rate = 1e-3
        
        if status == 'remote':
            self.voc2007_root = '/home/kindeqi/PyTorch_SSD/dataset/VOCdevkit/VOC2007/JPEGImages/'
            self.voc2007_trn_anno = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_train2007.json'
            self.voc2007_val_anno = '/home/kindeqi/PyTorch_SSD/annotation/PASCAL_VOC/pascal_val2007.json'
            self.vgg_weight_path = '/home/kindeqi/.torch/models/vgg16-397923af.pth'
            self.vgg_reduced_weight_path = '/home/kindeqi/PyTorch_SSD/weights/vgg16_reducedfc.pth'
            
        if status == 'local':
            self.voc2007_root = 'C:\\datasets\\pascal\\VOC2007\\JPEGImages\\'
            self.voc2007_trn_anno = 'C:\\datasets\\pascal\\PASCAL_VOC\\pascal_train2007.json'
            self.voc2007_val_anno = 'C:\\datasets\\pascal\\PASCAL_VOC\\pascal_val2007.json'
            self.vgg_weight_path = 'C:\\Users\\ruifr\\.torch\\models\\vgg16-397923af.pth'
            self.vgg_reduced_weight_path = 'C:\\Users\\ruifr\\.torch\\models\\vgg16-397923af.pth'