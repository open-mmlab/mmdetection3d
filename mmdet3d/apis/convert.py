import os
import torch
from mmdet3d.apis.inference import init_model
from mmdet3d.apis.utils import find_config


def convert_to_onnx(config, checkpoint=None, device='cuda:0', export_path='.', verbose=False):
    """ Function to convert model to ONNX format """
    model = init_model(config, checkpoint, device=device)
    sub_model = model.get_sub_model_for_conversion()
    model_name = model.cfg['model']['type']
    trace_input_shapes = model.cfg['conversion']['trace_input_shapes']
    trace_input = [torch.randn(shape).to(device) for shape in trace_input_shapes]
    torch.onnx.export(sub_model, tuple(trace_input), os.path.join(export_path, f'{model_name}.onnx'),
                      opset_version=11, verbose=verbose)


if  __name__ == '__main__':
    config = find_config('centerpoint/centerpoint_03pillar_kitti_lum.py')
    checkpoint = '/home/mark/checkpoints/centrpoint/centerpoint_lum_iris.pth'
    convert_to_onnx(config, checkpoint=checkpoint, verbose=True)