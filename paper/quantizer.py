import torch
from paper.quantizer import Quantizer

class Controller:
    def __init__(self, model, save_bit_path=None, load_bit_path=None, byte_per_group=16):
        self.model = model
        
        self.quantizer = Quantizer(byte_per_group=byte_per_group)
        # does not quantize model parameters
        self.quantizer.filter_tensors(model.named_parameters())
        self.byte_per_group = byte_per_group
        self.iter = 0

    def __del__(self):
        self.uninstall_hook()

    def print(self):
        self.quantizer.print_result()

    def iterate(self, get_grad):
        self.quantizer.iterate()
        self.iter += 1
        self.quantizer.seed_iter = self.iter

    def quantize(self, input):
        return self.quantizer.quantize(input)

    def dequantize(self, input):
        return self.quantizer.dequantize(input)

    def install_hook(self):
        def pack_hook(x):
            r = self.quantize(x)
            del x
            return r

        def unpack_hook(x):
            r = self.dequantize(x)
            del x
            return r

        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._register_saved_tensors_default_hooks(
                pack_hook, unpack_hook)
        else:
            torch._C._autograd._push_saved_tensors_default_hooks(
                pack_hook, unpack_hook)

    def uninstall_hook(self):
        if torch.__version__ < torch.torch_version.Version('1.10'):
            print("[Error] Please install PyTorch with version >= 1.10")
        elif torch.__version__ < torch.torch_version.Version('1.11'):
            torch._C._autograd._reset_saved_tensors_default_hooks()
        else:
            torch._C._autograd._pop_saved_tensors_default_hooks()
    
