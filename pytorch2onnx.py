import torch
MODEL_NAME = "dinov3_vits16"
device = torch.device('cpu')
weights = '/media/inno/Pretrained/DINOV3/ViT_models_pretrained_on_web_dataset_(LVD-1689M)/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'

# 1. load model
model = torch.hub.load(
    repo_or_dir='/home/inno/code/dinov3/',
    model=MODEL_NAME,
    source="local",
    weights=weights,
).to(device)
model.eval()

# Generate some input data
input_data = torch.randn(1, 3, 512, 512).to('cpu')
# Pass the input data through the model
output = model(input_data)

torch.onnx.export(model, input_data, 'model.onnx',opset_version=17)
