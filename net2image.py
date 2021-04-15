import torch
import onnx
from onnx import shape_inference
from model import Discriminator

# MODEL_NAME = 'G_epoch_4_100_osrgan.pth'
# model = Generator_OSRGAN(4).eval()
# model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
# x = torch.randn(64, 3, 9, 9)
# file = 'epochs/' + 'ONNX_' + MODEL_NAME + '_OSRGAN.onnx'
# torch.onnx.export(model, x, file)
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load(file)), file)
MODEL_NAME = 'D_epoch_4_100.pth'
model = Discriminator().eval()
model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
x = torch.randn(64, 3, 3, 3)
file = 'epochs/' + 'ONNX_' + MODEL_NAME + '.onnx'
torch.onnx.export(model, x, file)
onnx.save(onnx.shape_inference.infer_shapes(onnx.load(file)), file)
