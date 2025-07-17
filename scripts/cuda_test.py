import torch
print(f'Check whether torch cuda is avaiable, expect: true. Actual: {torch.cuda.is_available()}')
print(f'Check torch.version.cuda, expect 12.0. Acutal: {torch.version.cuda}')
print(f'Check torch.cuda.device, expect your GPU mode. Actual: {torch.cuda.get_device_name()}') 
