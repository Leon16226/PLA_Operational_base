from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *
from utils import torch_utils
import torch.nn as nn

def create_modules(module_defs, img_size, cfg):
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    _ = module_defs.pop(0)
    output_filters = [3]
    module_list = nn.ModuleList()
    routs = []
    yolo_index =-1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']
            stride = mdef['stride']
            if isinstance(k, int):
                modules.add_module('Conv2d',nn.Conv2d(in_channels=output_filters[-1],
                                                      output_filters=filters,
                                                      kernel_size=k,
                                                      stride=stride,
                                                      padding=k // 2 if mdef['pad'] else 0,
                                                      groups=mdef['groups'] if 'groups' in mdef else 1,
                                                      bias=not bn
                                                      ))
            else:

                ## modules.add_module('MixConv2d', Mix)
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1E-4))
            else:
                routs.append(i)





class Darknet(nn.Module):

    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super().__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg)
        self.yolo_layers = get_yolo_layers(self)

        self.version = np.array([0,2,5],dtype=np.int32)
        self.seen = np.array([0], dtype=np.in64)

    def forward(self, x, augment=False, verbose=False):
        if not augment:
            return self.forear_once(x)

    def forward_once(self, x, augment=False, verbose=False):
        img_size = x.shape[-2:]
        yolo_out, out =[] ,[]

        for i,module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in []:
                x = module(x, out)
            elif name == 'YOLOLayer':
                yolo_out.append(module(x, out))
            else:
                x = module(x)

        if self.training:
            return yolo_out
        else:
            x, p = zip(*yolo_out)
            x = torch.cat(x, 1)
            if augment:
                x = torch.split(x, nb, dim=0)

            return x, p










def get_yolo_layers(model):
    return [i for i,m in enumerate(model.module_list) if m.__class__.__name__ == 'YOLOLayer']

if __name__ == "__main__":
    print([1234] * 2)