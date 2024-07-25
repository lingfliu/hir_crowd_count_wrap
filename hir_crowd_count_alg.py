from ucs_alg_node import Alg, AlgTask, cli, AlgResult
from PIL import Image
import torch
from torchvision import transforms
from models import vgg19
import os
import numpy as np
import time

import requests

alg_name = 'hir_crowd_count'
alg_mode = 'batch'

class HirCrowdCountAlg(Alg):
    def __init__(self, model, id):
        super().__init__(alg_mode, model)
        self.name = alg_name
        self.mode = 'batch'
        self.model = model
        self.id = id

        self.model = vgg19()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(os.path.join('./teedy/vgg/0106-190713', 'best_model.pth'), self.device))
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.fcli = cli.MinioCli('62.234.16.239', 9090, 'ucs-alg', 'admin', 'admin1234')

    def infer_batch(self, task):
        task_id = task.id
        input = task.sources[0]
        # download the image
        img_path = task_id + '.jpg'
        self.fcli.download(input, img_path)

        img = Image.open(img_path).convert('RGB')

        img = self.trans(img)
        img = img.unsqueeze(0)
        input = img.to(self.device)
        assert input.size(0) == 1, 'the batch size should equal to 1'

        with torch.set_grad_enabled(False):
            output = self.model(input)
            count = torch.sum(output).item()

            scalex = input.size(2) / (output.size(2) + 0.0)
            scaley = input.size(3) / (output.size(3) + 0.0)
            input = input[0].cpu().detach().numpy()
            output = output[0].cpu().detach().numpy()

            if scalex >= scaley:
                output_scale = np.kron(output, np.ones((int(scaley), int(scaley))))

            else:
                output_scale = np.kron(output, np.ones((int(scalex), int(scalex))))

            xlim = min(output_scale.shape[1], input.shape[1])
            ylim = min(output_scale.shape[2], input.shape[2])
            # input_crop = input[:, 0:xlim, 0:ylim]
            # output_scale = output_scale[:, 0:xlim, 0:ylim]

            toc = int(time.time_ns()/1000/1000)
            result_filename = task_id + '_res_' + str(toc) + '.dat'


            with open(result_filename, 'wb') as f:
                np.savetxt(f, output[0])
                f.close()

            self.fcli.upload(result_filename, result_filename)

            # print('done with task', task_id, 'result:', result_filename)
            os.remove(img_path)
            os.remove(result_filename)

            # output_rgb = np.zeros(input_crop.shape)
            # output_rgb[0, :, :] = output_scale
            # output_rgb = output_rgb / np.max(output_rgb) * 255
            #
            # input_crop_rgb = input_crop / np.max(input_crop) * 255

            return AlgResult(task.id, toc, [count, result_filename], '')
