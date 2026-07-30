#include <torch/extension.h>
#include "compression.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define NSTREAMS 4

static cudaStream_t streams[NSTREAMS];

void Compress(torch::Tensor data, torch::Tensor c_data, int byte_per_group)
{
  int M= data.numel()/16;
  CompressCuda((int32_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M, byte_per_group);
  return;
}

void Decompress(torch::Tensor data, torch::Tensor c_data, int byte_per_group)
{
  int M= data.numel()/16;
  DecompressCuda((int32_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M, byte_per_group);
  return;
}

void TruncCompress(torch::Tensor data, torch::Tensor c_data, int byte_per_group)
{
  int M= data.numel()/16;
  TruncCompressCuda((int32_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M, byte_per_group);
  return;
}

void TruncDecompress(torch::Tensor data, torch::Tensor c_data, int byte_per_group)
{
  int M= data.numel()/16;
  TruncDecompressCuda((int32_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M, byte_per_group);
  return;
}

void FacetCompress(torch::Tensor data, torch::Tensor c_data, int byte_per_group)
{
  int M= data.numel()/16;
  FacetCompressCuda((int32_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M, byte_per_group);
  return;
}

void FacetDecompress(torch::Tensor data, torch::Tensor c_data, int byte_per_group)
{
  int M= data.numel()/16;
  FacetDecompressCuda((int32_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M, byte_per_group);
  return;
}

void CompressMask(torch::Tensor data, torch::Tensor mask)
{
  int M= data.numel()/16;
  CompressMaskCuda((int32_t*)data.data_ptr(), (uint8_t*)mask.data_ptr(), M);
  return;
}

void DecompressMask(torch::Tensor data, torch::Tensor mask)
{
  int M= data.numel()/16;
  DecompressMaskCuda((int32_t*)data.data_ptr(), (uint8_t*)mask.data_ptr(), M);
  return;
}

void CompressDropout(torch::Tensor data, torch::Tensor c_data)
{
  int M= (data.numel()+7)/8;
  CompressDropoutCuda((uint8_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M);
  return;
}

void DecompressDropout(torch::Tensor data, torch::Tensor c_data)
{
  int M= (data.numel()+7)/8;
  DecompressDropoutCuda((uint8_t*)data.data_ptr(), (uint8_t*)c_data.data_ptr(), M);
  return;
}

void Profile(torch::Tensor c_data, torch::Tensor data, torch::Tensor budget, torch::Tensor budget1, int byte_per_group)
{
  int M= data.numel();
  ProfileCuda((uint8_t*)c_data.data_ptr(), (uint8_t*)data.data_ptr(), (int32_t*)budget.data_ptr(), (int32_t*)budget1.data_ptr(), M, byte_per_group);
  return;
}

void FacetProfile(torch::Tensor c_data, torch::Tensor data, torch::Tensor budget, int byte_per_group)
{
  int M= data.numel();
  FacetProfileCuda((uint8_t*)c_data.data_ptr(), (uint8_t*)data.data_ptr(), (int32_t*)budget.data_ptr(), M, byte_per_group);
  return;
}






PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("compress", &Compress, "Compress");	
  m.def("decompress", &Decompress, "Decompress");	
  m.def("trunc_compress", &TruncCompress, "TruncCompress");	
  m.def("trunc_decompress", &TruncDecompress, "TruncDecompress");	
  m.def("facet_compress", &FacetCompress, "FacetCompress");	
  m.def("facet_decompress", &FacetDecompress, "FacetDecompress");	
  m.def("compress_mask", &CompressMask, "CompressMask");	
  m.def("decompress_mask", &DecompressMask, "DecompressMask");	
  m.def("compress_dropout", &CompressDropout, "CompressDropout");	
  m.def("decompress_dropout", &DecompressDropout, "DecompressDropout");	
  m.def("profile", &Profile, "Profile");	
  m.def("facet_profile", &FacetProfile, "FacetProfile");	

}



