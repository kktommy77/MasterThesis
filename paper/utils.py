import torch
import numpy as np
from compression_cuda import compress, decompress
from compression_cuda import trunc_compress, trunc_decompress
from compression_cuda import facet_compress, facet_decompress
from compression_cuda import compress_mask, decompress_mask
from compression_cuda import profile, facet_profile
from paper.utils import AverageMeter

class Quantizer:
    def __init__(self, byte_per_group):
        self.unrelated_tensors = set()  # record the tensors that should not be quantized
        self.byte_per_group = byte_per_group

        #self.compute_stream = torch.cuda.current_stream()
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.sum=0
        self.start_bwd = True
        self.budget=AverageMeter("budget")
        self.budget1=AverageMeter("budget1")
        self.data0= {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
        # data collected for auto precision
        #self.seeds = {}
        #self.bits = {}
        #self.dims = {}

        self.iter = 0  # total number of iterations, including the extra inter for auto precision
        # iteration for seed, share the same seed_iter for the same auto precision adaptive step
        self.seed_iter = 0

    def filter_tensors(self, pairs):
        for _, v in pairs:
            self.unrelated_tensors.add(v.data_ptr())
    
    def print_result(self):
        print(f"Average Budget0 is {self.budget.get_value()}")
        print(f"Average Budget1 is {self.budget1.get_value()}")
        print(f"Data0 is {self.data0}")
        

    def update_counts(self, new_tensor):
        # 1. Compute bincount
        counts = torch.bincount(new_tensor)
        
        # 2. Find indices where count > 0 (optimization for sparse data)
        # indices are the "values", values are the "counts"
        non_zero_indices = counts.nonzero().squeeze()
        
        # Handle edge case where only one unique value exists
        if non_zero_indices.ndim == 0:
            non_zero_indices = non_zero_indices.unsqueeze(0)

        # 3. Update the dictionary
        for idx in non_zero_indices:
            key = idx.item()
            count = counts[idx].item()
            
            # Standard dictionary update (cumulative addition)
            if key in self.data0:
                self.data0[key] += count
            else:
                self.data0[key] = count      



    # return should_be_quantized, is_dropout_mask
    # treat dropout mask differently because it can be quantized with 1 bit with a specialized kernel
    def check_quantize(self, input_tensor, tid=1):
        # does not quantize parameters
        if input_tensor.numel() < 16:
            return False, False, False
        if input_tensor.data_ptr() in self.unrelated_tensors:
            return False, False, False
        # special check for saved mask
        if input_tensor.numel() > 0 and input_tensor.dtype == torch.uint8:
            if (input_tensor.max() == 1) and (input_tensor.min() == 0):
                return True, True, False
            return False, False, False
        if input_tensor.dtype == torch.float32:
            a= input_tensor.min().item()
            if a< (-1.7e38):
                return True, False, True  
        # only quantize float16 and float32 amd bfloat16
        if input_tensor.dtype not in [torch.float32]:
            return False, False, False
        # only quantize activation that requires gradient
        # for example: BN statistics (running mean/var) should not be quantized
        if input_tensor.requires_grad is False:
            return False, False, False
        # only quantize 2/3/4D tensors for now
        if ((len(input_tensor.shape) != 2)
            and (len(input_tensor.shape) != 3)
            and (len(input_tensor.shape) != 4)
            ):
            return False, False, False
        return True, False, False

    def __del__(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        del self.unrelated_tensors

    def iterate(self):
        del self.ptr_qtensor_map
        del self.layer_key_map
        self.ptr_qtensor_map = {}
        self.layer_key_map = {}
        self.tid = 0
        self.start_bwd = True
        self.iter += 1

    def generate_tensor_key(self, t, tid):
        if config.check_dup:
            # sample 100 elements data pointer + tensor.sum() as the key
            sample_cnt = min(100, t.numel())
            key = uniform_sample(t, sample_cnt, add_dataptr=True)
            key.append(t.sum().item())
            return tuple(key)
        else:
            return (tid)

    def quantize(self, input):
        M= input.numel()//16
        origtype= input.dtype
        quantize, is_dropout_mask, has_254 = self.check_quantize(input)
        #has_254=False
        if not quantize:
            return False, input, origtype
        # special case: use 1 bit to quantize dropout mask
        if is_dropout_mask:
            M= input.numel()//8
            c_input= torch.zeros(M, device=input.device, dtype=torch.uint8)
            compress_dropout(input, c_input)
            c_input= [c_input, input.shape]
        #    mask_254= torch.zeros_like(input, device=device, dtype=torch.uint8)
        #    input.__
            return True, True, False, c_input, origtype
        tid = self.tid
        self.tid += 1
        input_shape = input.shape
        key = (tid)
        self.layer_key_map[tid] = key
        skip_quantize = key in self.ptr_qtensor_map
        if not skip_quantize:
            #if self.iter == 0:
            #    bit = self.default_bit
                #self.bits[tid] = bit
                #self.dims[tid] = input.numel()
                #self.seeds[tid] = tid
            #else:
            #    bit = self.bits[tid]
            # quantize
            if input.is_contiguous():
                #print("Input is contiguous")
                input= input.view(-1, 16)
            else:    
                #print("Input is not contiguous")
                #print(f"Input shape is {input.shape}")
                input=input.contiguous().view(-1, 16)
            if not input.is_cuda:
                print("Not Cuda")
            c_input= torch.zeros(M*self.byte_per_group, device=input.device, dtype=torch.uint8)
            #c_input= input.clone()
            if has_254:
                mask_254= torch.zeros(M*2, device=input.device, dtype=torch.uint8)
                compress_mask(input, mask_254)
                #facet_compress(input, c_input, self.byte_per_group)
                compress(input, c_input, self.byte_per_group)
                self.ptr_qtensor_map[key] = [c_input, 1, tid, mask_254]
            else:
                #facet_compress(input, c_input, self.byte_per_group)
                compress(input, c_input, self.byte_per_group)
                #print(f"input size is {input.nbytes/ 1024:.2f} KB and c_input size is{c_input.nbytes/1024:.2f} KB")
                #self.sum+= c_input.nbytes
                #del input
                self.ptr_qtensor_map[key] = [c_input, 1, tid]
                #print("C_input sum %d MB" %(self.sum/(1024*1024)))
        else:
            # increase the ref count
            self.ptr_qtensor_map[key][1] += 1
        
        return True, False, has_254, key, input_shape, tid, origtype

    def dequantize(self, input):
        quantized = input[0]
        if not quantized:
            if input[1].dtype != input[2]:
                input = input.to(input[2])
            return input[1]
       
        is_dropout_mask= input[1]
        has_254 = input[2]
        if is_dropout_mask:
            _, _, _, q_inputs, origtype = input
            qq_inputs, input_shape= q_inputs
            ret= torch.zeros(np.prod(input_shape), dtype=torch.uint8, device=input.device)
            decompress_dropout(ret, qq_inputs)
            ret=ret.view(input_shape)
            if ret.dtype != origtype:
                ret=ret.to(origtype)
            return ret
        _, _, _, key, input_shape, tid, origtype = input
        
        if has_254:
            q_inputs, ref_cnt, key_tid, mask_254 = self.ptr_qtensor_map[key]
            M=q_inputs.numel()//(self.byte_per_group)
            data0 = torch.zeros(M, device=q_inputs.device, dtype=torch.uint8)
            #budget0 = torch.zeros(M, device=q_inputs.device, dtype=torch.int32)
            #budget1 = torch.zeros(M, device=q_inputs.device, dtype=torch.int32)
            
            if not q_inputs.is_cuda:
                q_inputs = q_inputs.cuda(non_blocking=False)
            if not mask_254.is_cuda:
                mask_254 = mask_254.cuda(non_blocking=False)
            ret=torch.zeros(M*16, device=q_inputs.device, dtype=torch.float32)
            
            #facet_profile(q_inputs, data0, budget0, self.byte_per_group)
            #profile(q_inputs, data0, budget0, budget1, self.byte_per_group)
            #self.update_counts(data0)
            #counts=torch.bincount(data0)

            #if counts[0].item()!=0:
            # self.budget.update((budget0.float().sum().item())/(counts[0].item()), counts[0].item())
            
            #if counts[1].item()!=0:
            # self.budget1.update((budget1.float().sum().item())/(counts[1].item()), counts[1].item())



            decompress(ret, q_inputs, self.byte_per_group)
            #facet_decompress(ret, q_inputs, self.byte_per_group)
            decompress_mask(ret, mask_254)
            ret= ret.view(*input_shape).contiguous()
        
        else:
            q_inputs, ref_cnt, key_tid = self.ptr_qtensor_map[key]
            #ret=q_inputs
            M=q_inputs.numel()//(self.byte_per_group)
            if not q_inputs.is_cuda:
                q_inputs = q_inputs.cuda(non_blocking=False)
            
            data0 = torch.zeros(M, device=q_inputs.device, dtype=torch.uint8)
            #budget0 = torch.zeros(M, device=q_inputs.device, dtype=torch.int32)
            #budget1 = torch.zeros(M, device=q_inputs.device, dtype=torch.int32)
            #facet_profile(q_inputs, data0, budget0, self.byte_per_group)
            #profile(q_inputs, data0, budget0, budget1, self.byte_per_group)
            #self.budget.update(budget0.float().mean().item(), M)
            #self.update_counts(data0)
            #print(f"budget's sum is {budget0.float().sum().item()}")
            #counts = torch.bincount(data0)
            
            #if counts[0].item()!=0:
            # self.budget.update((budget0.float().sum().item())/(counts[0].item()), counts[0].item())
            
            #if counts[1].item()!=0:
            # self.budget1.update((budget1.float().sum().item())/(counts[1].item()), counts[1].item())

 

            ret=torch.zeros(M*16, device=q_inputs.device, dtype=torch.float32)
            #ret=q_inputs
            #facet_decompress(ret, q_inputs, self.byte_per_group)

            decompress(ret, q_inputs, self.byte_per_group)


            #del q_inputs
            ret=ret.view(*input_shape).contiguous()
        #ret = op_dequantize(q_inputs, input_shape)
        if ret.dtype != origtype:
           ret=ret.to(origtype)
        ref_cnt -= 1
        if ref_cnt < 0:
            print("[Error] Ref count < 0", key, ref_cnt)
            exit(-1)
        elif ref_cnt == 0:
            del self.ptr_qtensor_map[key]
        else:
            self.ptr_qtensor_map[key] = [q_inputs, ref_cnt, key_tid]
        return ret
