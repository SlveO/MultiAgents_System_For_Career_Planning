import torch
print(torch.__version__)        # 应显示 2.4.0+cu118 或 2.3.0+cu118
print(torch.cuda.is_available()) # 应显示 True