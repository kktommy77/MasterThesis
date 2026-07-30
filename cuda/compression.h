#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>


__inline__ __device__ void Write8Bit(uint8_t* c_data, int& current_byte, int& current_idx, int idx, int byte_per_group, uint8_t bit_to_write, uint8_t val)
{
  while(bit_to_write>0)
  {
     if(current_byte-(byte_per_group*(idx+1))>=0) return;	  
     if(bit_to_write>=(current_idx+1))
     {
       c_data[current_byte] |= (val>>(bit_to_write-current_idx-1));	     
       bit_to_write-=(current_idx+1);     
       current_byte++;
       current_idx= 7;
     }
     else
     {
       current_idx-= bit_to_write;	     
       c_data[current_byte] |= (val&((1<<bit_to_write)-1))<<(current_idx+1);
       bit_to_write=0;
     }
  
  }
  return;
}

__inline__ __device__ void WriteMultiBit(uint8_t* c_data, int& current_byte, int& current_idx, int idx, int byte_per_group, uint8_t bit_to_write, int val)
{
  while(bit_to_write>0)
  {
    if(current_byte-(byte_per_group*(idx+1))>=0) return;	  
    if(bit_to_write>=(current_idx+1))
    {
       c_data[current_byte] |=  (val>>(22-current_idx));
       bit_to_write-=(current_idx+1);
       val =((val<<(current_idx+1))&0x7FFFFF);
       current_byte++;
       current_idx=7;       
    }
    else
    {
       current_idx-= bit_to_write;
       c_data[current_byte] |= (((val>>(23-bit_to_write))&((1<<bit_to_write)-1))<<(current_idx+1));      
       bit_to_write=0;
    }

  }
  return;
}

__inline__ __device__ void GetEachBudget(int current_byte, int current_idx, int idx, int byte_per_group, uint8_t num, uint16_t mask, int* delta)
{
  int budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));
  //if(idx==0) printf("budget is %d!\n", budget);
  int sum=0;	  
  int k=0;
  int th=0;
  int s=0;
  float f=0.0;
  for(int i=0; i<16; i++)
  {
    if(((mask>>(15-i))&0x1)==0) sum+= (delta[i]&0xFF);
  }  
  th=(sum/16)*2;
  if(budget>=sum)
  {
    //do nothing
  }
  else
  {
    for(int i=0; i<16; i++)
    {
      if(((mask>>(15-i))&0x1)==0)
      {
         if((delta[i]&0xFF)>th)
         {		 
	    f= ((float)th)/(delta[i]&0xFF);
	    if(f<=0.5) 
	    {
	      s= roundf((1.0)/(1.0-f));
	    }
            else
	    {
	      s= roundf((1.0-powf(f,((float)(delta[i]&0xFF))))/(1.0-f));
	    }
	    delta[i]&= ~0xFF;
            delta[i]|= (s);
	 }
      }	      
    }	    
    while(sum>budget)
    {
      k++;
      sum=0;
      for(int i=0; i<16; i++)
      {
        if(((mask>>(15-i))&0x1)==0) sum+= ((delta[i]&0xFF)>>k);
      }
    }
  }
  //////
  //while(sum>budget)
  //{
  //
  // }
  //////
  s=0;
  for(int i=0; i<16; i++)
  {
    th=0;	  
    if(((mask>>(15-i))&0x1)==0) 
    { 	    
      th=((delta[i]&0xFF)>>k)+(budget-sum)/(16-num)+ (s<((budget-sum)%(16-num)));
      s++;
    } 
    th= (th>23)?23:th;    
    delta[i]&= ~0xFF;
    delta[i]|=(th&0xFF);	    
  }
 /*
 if(idx==0)
 {
    for(int i=0; i<16; i++)
    {	    
      printf("delta of index %d is %d\n", i, delta[i]&0xFF);
    } 
 }
 */
  return;
}

__inline__ __device__ void Read8Bit(uint8_t* c_data, int& current_byte, int& current_idx, int idx, int byte_per_group, int bit_to_read, uint8_t& val0)
{
  	
  val0=0;
  uint8_t temp=0;
  while(bit_to_read>0)
  {
    if(current_byte-(byte_per_group*(idx+1))>=0) return;	  
    if(bit_to_read>=(current_idx+1))
    {
       temp= c_data[current_byte];
       bit_to_read-=(current_idx+1);
       val0|= (temp&((1<<(current_idx+1))-1));
       val0= val0<<(bit_to_read);
       current_byte++;
       current_idx=7; 
    }
    else
    {
      temp= c_data[current_byte];
      current_idx-=bit_to_read;
      val0 |= ((temp>>(current_idx+1))&((1<<(bit_to_read))-1));
      bit_to_read=0;
    }
  }
  return;
}

__inline__ __device__ void ReadMultiBit(uint8_t* c_data, int& current_byte, int& current_idx, int idx, int byte_per_group, int bit_to_read, int& val0)
{
  	
  val0=0;
  uint8_t temp=0;
  while(bit_to_read>0)
  {
    if(current_byte-(byte_per_group*(idx+1))>=0) return;	  
    if(bit_to_read>=(current_idx+1))
    {
       val0= (val0<<(current_idx+1));
       temp= c_data[current_byte];
       bit_to_read-=(current_idx+1);
       val0|= ((temp&((1<<(current_idx+1))-1)));
       current_byte++;
       current_idx=7; 
    }
    else
    {
      temp= c_data[current_byte];
      current_idx-=bit_to_read;
      val0= (val0<<(bit_to_read));
      val0 |= ((temp>>(current_idx+1))&((1<<(bit_to_read))-1));
      bit_to_read=0;
    }
  }
  return;
}

__inline__ __device__ void printb(uint8_t* c_data, int byte_per_group)
{
   for(int i=0; i< byte_per_group; i++)
   {
      for(int j=0; j<8; j++)
      {
        printf("%d",(c_data[i]>>(7-j))&0x1);
      }	      
      printf(" ");
   }
   printf("\n");
   return;

}

__inline__ __device__ void GetEachBudget2(int current_byte, int current_idx, int idx, int byte_per_group, int* delta)
{
  int budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));	  
  for(int i=0; i<16; i++)
  {
     uint8_t temp= (budget/16)+(i<(budget%16));
     delta[i]&= ~0xFF;
     delta[i]|=((temp)&0xFF);
  }
  /*
  if(idx==0) printf("budget is %d!!\n", budget);
  if(idx==0)
  {
    for(int i=0; i<16; i++)
    {
      printf("delta of index %d is %d\n", i, delta[i]&0xFF);
    }
  }
  */
  return;
}



__global__ void CompressCudaKernel(int32_t* data, uint8_t* c_data, int M, int byte_per_group, int mode)
{ 
  int idx= blockDim.x*blockIdx.x+threadIdx.x;
  if(idx>=M) return;
  
  int current_byte= idx*byte_per_group;
  int current_idx=7;
  int delta[16]= {0};
  
  uint8_t base=255;
  uint8_t tag=0;
  
  uint8_t temp=0;
  uint16_t mask=0;
  uint8_t val0=0;
  uint8_t val1=0;
  uint8_t val2=0;
  int max_diff=0;
  int meet=0;

  for(int i=0; i<16; i++)
  {
    temp = (data[16*idx+i]>>23)&0xFF;

    delta[i]=255;
    if(temp==0)
    {
       tag=1;
    }
    else
    {
      if(base>temp) base=temp;
    }
  }
  Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
  //if(idx==0) printf("Base is %d!\n", base);
  Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);

  if(tag==1)
  {
    tag=0;
    for(int i=0; i<16; i++)
    {
      temp = (data[16*idx+i]>>23)&0xFF;
      if((temp==0)&(meet==0)) {meet=1; tag++;}
      if((temp!=0)&(meet==1)) meet=0;
      
      if(temp!=0)
      {
	delta[i]=(temp-base);      
        if(max_diff<(temp-base)) max_diff=(temp-base);
      }


    }
    val0= 8-(__clz(max_diff)-24);

    if(tag>1) { tag=1;  Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag); }
    else { tag=0; Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag); }
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, val0);
    
    if(tag==1)
    {
      max_diff=0;	    
      for(int i=0; i<16; i++)
      {
	 max_diff= max_diff<<1;     
         if(delta[i]==255)
	 {
           max_diff|= 0x1;		 
	 }
      }
      Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, max_diff>>8);
      Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, max_diff&0xFF);
    }
    else
    {
      meet=0;
      val1=0;      
      for(int i=0; i<16; i++)
      {
        if((delta[i]==255)&(meet==0)) { meet=1; val1|=i; val1= (val1<<4);}
	if((delta[i]!=255)&(meet==1)) { meet=0; val1|=(i-1); }
      //if(idx==0) printf("val1 is %d!!\n", val1);
	if((meet==1)&(i==15)) { val1|=0xF; }
      //if(idx==0) printf("val1 is %d!!\n", val1);
      }
      Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, val1);   
    }

    for(int i=0; i<16; i++)
    {
      if(delta[i]!=255) Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, val0, delta[i]);
    }

  }
  else
  {
     tag=0;
     meet=0;
     max_diff=0;
     val1=255;
     for(int i=0; i<16; i++)
     {
	temp= (data[16*idx+i]>>23)&0xFF;     
        if(temp!=255)
	{
	   delta[i]=(temp-base);
	   if(max_diff<(temp-base)) max_diff=(temp-base);
           if((val1>(temp-base))&((temp-base)!=0)) val1= (temp-base);
	}
     }
     temp=0;
  //if(idx==0) {for(int i=0;i<16;i++) {printf("Before0 delta of index %d is %d!\n", i, delta[i]); }}
     for(int i=0; i<16; i++)
     {
        if(delta[i]==0) 
	{
	  meet++;
	  if (meet==1) { temp|= ((i)&0xF);}
          if (meet==2) {temp=temp<<4; temp|= ((i)&0xF);}
	}
     }

     val2= 8-(__clz(max_diff)-24);
     val0= 8-(__clz(max_diff-(int)val1)-24);
     tag= (((val2-val0)>0)&(val2-val0<5))&(meet<=2);
     Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);

     if(tag==1)
     {
	   
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, val0);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 2, val2-val0-1);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, val2, val1);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, (meet==2));
        //if(idx==0) printf("val0 is %d!\n", temp);
        if(meet==2) { Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, temp); }
	else{ Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, temp&0xF); }
         

	for(int i=0; i<16; i++)
	{
	  if(delta[i]!=0) Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, val0, delta[i]-val1);
          	
	}

     }
     else
     {
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, val2);
        for(int i=0; i<16; i++)
	{
         //if(idx==0) printf("Diff of Index %d is %d\n", i, delta[i]);		
	  Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, val2, delta[i]);
	}

     }

  }

  //Compress Sign
  tag=1;
  val0=((data[16*idx])>>31)&0x1;
  max_diff=val0;
  for(int i=1; i<16; i++)
  {
    val1= (data[16*idx+i]>>31)&0x1;	  
    max_diff= (max_diff<<1);
    max_diff|= (val1&0x1);    
    if(val1!=val0)
    {
      tag=0;
    }
  }
  if(tag==0)
  {
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, (max_diff>>8)&0xFF);
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, max_diff&0xFF);
  }
  else
  {
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 2, tag*2+val0); 
  }

  //Compress Mantissa
  val0=0;
  for(int i=0; i<16; i++)
  {
    mask= mask<<1; 
    if(delta[i]==255)
    {
       val0++;
       mask|=0x1;
    }
  }

  //if(idx==0) {for(int i=0;i<16;i++) {printf("Before delta of index %d is %d!\n", i, delta[i]); }}
  if (mode==16) {GetEachBudget2(current_byte, current_idx, idx, byte_per_group, delta);} 
  else{ GetEachBudget(current_byte, current_idx, idx, byte_per_group, val0, mask,  delta); } 
  //if(idx==0) {for(int i=0;i<16;i++) {printf("After delta of index %d is %d!\n", i, delta[i]); } printf("delta_bw is %d", val2);}
  for(int i=0; i<16; i++)
  {
     WriteMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, delta[i], data[16*idx+i]&0x7FFFFF); 
  }
  //if(idx==0) printb(c_data, byte_per_group);
  return;
}

__global__ void DecompressCudaKernel(int32_t* data, uint8_t* c_data, int M, int byte_per_group, int mode)
{
  int idx= blockDim.x*blockIdx.x+threadIdx.x;
  if(idx>=M) return;
  
  int32_t delta[16]= {0};
  int current_byte= idx*byte_per_group;
  int current_idx=7;
  uint8_t base;
  uint8_t tag;
  uint8_t delta_bw0=0; 

  int mask=0;
  int sign=0;
  uint8_t temp=0;
  uint8_t val0=0;
  uint8_t val1=0;
  uint8_t val2=0;
  int temp_int=0;
	  
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  
  if(tag==1)
  {
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag); 
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, delta_bw0);
    if(tag==1)
    {
      ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
      //if(idx==0) printf("MASK is %d", mask);
    }
    else
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val1);
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val2);

      for(int i=0; i<16; i++)
      {
	mask= mask<<1;      
        if((i>=val1)&(i<=val2))
	{
	  mask|=0x1;
	}
      
      }
    }
    val1=0;
    for(int i=0; i<16; i++)
    {
       if(((mask>>(15-i))&0x1)==0)
       {
	 Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, delta_bw0, val0);
         delta[i]= (base+val0)<<8;
	 delta[i]|=(val0);
	 val1++;
       } 
    } 
  }
  else
  {
    tag=0;	  
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, delta_bw0);

    if(tag==1)
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 2, val1);
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, val1+delta_bw0+1, val2); //exp_bias
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
      if(tag==1)
      {
         Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, val0);      
      }
      else
      {
         Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val0);      
      }
      //if(idx==0) printf("val0 is %d!\n", val0);
      for(int i=0; i<16; i++)
      {
         if(((tag==1)&((i!=(val0&0xF))&(i!=((val0>>4)&0xF))))|((tag==0)&(i!=(val0&0xF))))
	 {
	    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, delta_bw0, temp);     
	    delta[i] = (base+val2+temp)<<8;
	    delta[i] |= (val2+temp); 
	 }
         else
	 {
	    delta[i] = base<<8;
	 }		 
      }
      //if(idx==0) {for(int i=0; i<16; i++){ printf("exp stored of idx %d is %d!\n", i, delta[i]>>8);} }
    }
    else
    {
      for(int i=0; i<16; i++)
      {
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, delta_bw0, val0);
	delta[i]= (base+val0)<<8;
	delta[i]|= val0;
      }
      //if(idx==0) {for(int i=0;i<16;i++) {printf("Before in Decomp delta of index %d is %d!\n", i, (delta[i]>>8)&0xFF); }}
    }
    val1=16;
  }
  
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  if(tag==1) { Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, val0); sign= (val0==1)?((1<<16)-1):0; }
  else
  {
    ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, sign);	  
  }
  //if(idx==0) printf("Sign is %d!\n",sign);
  
  //if(idx==0) {for(int i=0;i<16;i++) {printf("Before1 in Decomp delta of index %d is %d!\n", i, (delta[i])&0xFF); }}
  if(mode==16) {GetEachBudget2(current_byte, current_idx, idx, byte_per_group, delta);}
  else{GetEachBudget(current_byte, current_idx,idx,  byte_per_group, 16-val1, mask&(0xFFFF), delta); }
  //if(idx==0) {for(int i=0; i<16; i++){ printf("exp stored of idx %d is %d!\n", i, delta[i]>>8);} }
  //if(idx==0) {for(int i=0;i<16;i++) {printf("After in Decomp delta of index %d is %d!\n", i, (delta[i])&0xFF); }}
  for(int i=0; i<16; i++)
  {
    ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, delta[i]&0xFF, temp_int);	  
    data[16*idx+i] = (((sign>>(15-i))&0x1)<<31) | (((delta[i]>>8)&0xFF)<<23) | (temp_int<<(23-(delta[i]&0xFF)));
  }
  return;

}



void CompressCuda(int32_t* data, uint8_t* c_data, int M, int byte_per_group, int mode)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  CompressCudaKernel<<<gridDim, blockDim>>>(data, c_data, M, byte_per_group, mode);
  cudaDeviceSynchronize();
  return;
}

void DecompressCuda(int32_t* data, uint8_t* c_data, int M, int byte_per_group, int mode)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  DecompressCudaKernel<<<gridDim, blockDim>>>(data, c_data, M, byte_per_group, mode);
  cudaDeviceSynchronize();
  return;
}

__global__ void TruncCompressCudaKernel(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
   int idx= blockIdx.x*blockDim.x+threadIdx.x;
   if(idx>=M) return;
   int current_byte= idx*byte_per_group;
   int current_idx=7;
   uint16_t mask=0;
   int budget=0;
   int temp=0;
   for(int i=0; i<16; i++)
   {
     mask= mask<<1;	   
     if(((data[16*idx+i]>>31)&0x1)==1) mask|=0x1;	   
     Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, (data[16*idx+i]>>23)&0xFF);
   }
   Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask>>8);
   Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask&0xFF);
   budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));
   for(int i=0; i<16; i++)
   {
     temp=budget/16+ (i<(budget%16));
     WriteMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, temp, data[16*idx+i]&0x7FFFFF);	     
   }
   return;
}

__global__ void TruncDecompressCudaKernel(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
   int idx= blockIdx.x*blockDim.x+threadIdx.x;
   if(idx>=M) return;
   int current_byte= idx*byte_per_group;
   int current_idx=7;
   int budget=0;
   uint8_t val0=0;
   int val1=0;
   int val2=0;
   int temp=0;
   int temp1[16]= {0};
   for(int i=0; i<16; i++)
   {
     Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, val0);
     temp1[i]|=(val0<<23);
   }
   ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, val2);
   budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));
   for(int i=0; i<16; i++)
   {
     temp=budget/16+ (i<(budget%16));
     ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, temp, val1);
     data[16*idx+i]= (((val2>>(15-i))&0x1)<<31) | temp1[i] | (val1<<(23-temp));     
   }
   return;
}


void TruncCompressCuda(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  TruncCompressCudaKernel<<<gridDim, blockDim>>>(data, c_data, M, byte_per_group);
  cudaDeviceSynchronize();
  return;
}

void TruncDecompressCuda(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  TruncDecompressCudaKernel<<<gridDim, blockDim>>>(data, c_data, M, byte_per_group);
  cudaDeviceSynchronize();
  return;
}


__global__ void FacetCompressCudaKernel(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
  int idx= blockDim.x*blockIdx.x+threadIdx.x;
  if(idx>=M) return;
  int current_byte= idx*byte_per_group;
  int current_idx=7;
  uint8_t tag=0;
  uint8_t temp=0;
  uint8_t base=0;
  uint8_t meet=0;
  uint8_t val0=0;
  uint8_t local=0;
  int status=0;
  int mask=0;
  int budget=0;
  int a=0;
  int delta[16]={0};
  // status=0 --> uncompressible
  // status=1 --> local-base
  // status=2 --> local-base with outlier
  // status=3 --> zero-stream
  // status=4 --> global-base

  for(int i=0; i<16; i++)
  {
    temp= ((data[16*idx+i]>>23)&0xFF);
    delta[i]=temp;
    if((temp<0x78)||(temp>0x87))
    {
       tag=1;
    }   
  }
  if(tag==0)//global-base
  {
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, 1);
    for(int i=0; i<16; i++)
    {
      temp= (delta[i]>127)?1:0;	    
      tag=(temp<<3) | (delta[i]&0x7);	    
      Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, tag);
    }
  
  }
  else
  {
    tag=0;
    local=1;
    val0=0;
    for(int i=0; i<16; i++)
    {
      if((delta[i]==0)&(meet==0))
      {
        meet=1;
	tag++;
        val0|= (i<<4);
      }
      if((delta[i]!=0)&(meet==1)) 
      {
         meet=0;
	 val0|=(i-1)&0xF;
      }
      if((meet==1)&(i==15))
      {
         val0|=0xF;
      }
    }
    for(int i=0; i<16; i++)
    {
      if(delta[i]!=0)
      {
        if(status==0)
	{
	  base=delta[i];
	  status++;
	}
        else
	{
	  a=delta[i]-base;
	  if((a>15)|(a<(-16)))
	  {
	    local=0;
	  }
	}
      }
    }

    if((tag==1)&(local==1))// zero-stream
    {
       Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, 1);
       Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, val0);
       Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
       status=0;
       for(int i=0; i<16; i++)
       {
         if(delta[i]!=0)
	 {
	    if(status==0)
	    {
	      status++;
	    }
	    else
	    {
	      a=delta[i]-base;
	      val0= a&0x1F;
	      Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0);
	    }
	 }
       }
       
    }
    else
    {
      tag=0;
      status=0;      
      for(int i=0; i<16; i++)
      {
        if(status==0)
	{
          status++;
          base=delta[i];	  
	}
	else
	{
          a=delta[i]-base;
          if((a>15)|(a<(-16)))
	  {
	    tag++;
	    temp=i;
	  }	  
	}
      }
      if(tag==0)//local-base without outlier
      {
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, 2);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	status=0;
	for(int i=0; i<16; i++)
	{
      	   if(status==0)
	   {
	      status++;
	   }
	   else
	   {
	      a=delta[i]-base;
              val0= a&0x1F;	      
	      Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0); 
	   }
	}

      }
      else if(tag==1)//local-base with outlier
      {
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, 3);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, temp);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, delta[temp]&0xFF);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	status=0;
	for(int i=0; i<16; i++)
	{
	  if(i!=temp)
	  {
	     if(status==0)
	     {
	       status++;
	     }
	     else
	     {
	       a=delta[i]-base;
	       val0= a&0x1F;
	       Write8Bit(c_data,current_byte, current_idx, idx, byte_per_group, 5, val0);
	     }
	  }
	}
      }
      else// uncompressible
      {
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, 1);
        for(int i=0; i<16; i++)
	{
	   Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, delta[i]&0xFF);
	}
      }
    }

  }
 
  
  //Compress Sign and Mantissa
  tag=1;
  mask=0;
  temp= (data[16*idx]>>31)&0x1;
  mask|=temp;
  for(int i=1; i<16; i++)
  {
    val0= (data[16*idx+i]>>31)&0x1;	  
    mask= mask<<1;	  
    if(temp!=val0)
    {
      tag=0;  
    }
    if(val0==1) mask|=0x1;
  }
  if(tag==1)
  {
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, temp);
  }
  else
  {
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask>>8);
    Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask&0xFF);
  }

  budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));	  
  a=0;
  while(budget>0)
  {
    a=(a<23)?a:22;
    mask=0;    
    if(a<3)
    {
      tag=1;
      temp= (data[16*idx]>>(22-a))&0x1;
      mask|=temp; 
      for(int i=1; i<16; i++)
      {
        val0= (data[16*idx+i]>>(22-a))&0x1;	  
        mask= mask<<1;	  
        if(temp!=val0)
        {
          tag=0;  
        }
        if(val0==1) mask|=0x1;
      }
      if(tag==1)
      {
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, temp);
        budget-=2;
      }
      else
      {
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask>>8);
        Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask&0xFF);
        budget-=17;
      }
      a++;
    }
    else
    {
       for(int i=0; i<16; i++)
       {
	 mask=mask<<1;
	 val0= (data[16*idx+i]>>(22-a))&0x1;
         mask|= (val0);	 
       }	       
       Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask>>8);
       Write8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, mask&0xFF);
       budget-=16;
       a++;   
    }
     
  }
  //if(idx==0) printb(c_data, byte_per_group);  
  return;
}	

__global__ void FacetDecompressCudaKernel(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
  
  int idx= blockDim.x*blockIdx.x+threadIdx.x;
  if(idx>=M) return;
  int current_byte= idx*byte_per_group;
  int current_idx= 7;
  int delta[16]= {0};
    
  uint8_t base=0;
  uint8_t tag=0;
  uint8_t val0=0;
  int mask=0;
  int budget=0;
  int k =0;
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  if(tag==1)// Uncompressible
  {
    for(int i=0; i<16; i++)
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, tag);
      delta[i]= ((tag)<<23);
    }
  }
  else
  {
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    if(tag==1)//local-base
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
      if(tag==1)//with-outlier
      {
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val0);
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, tag);
	delta[val0]= ((tag)<<23);
        mask=(1<<(val0));
	Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	
	for(int i=0; i<16; i++)
	{
          if(((mask>>i)&0x1)==0)
          {
	    if(k==0)
	    {
	      delta[i]= (base<<23);
	      k++;
	    }
	    else
	    {
              Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0);
	      if(((val0>>4)&0x1)==1) { val0= (~val0&0xF)+1;  delta[i]=((base-(val0))<<23);}
	      else { delta[i]= ((base+val0)<<23);  }
	    }
	  }		  
	}

      }
      else//without outlier
      {
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	for(int i=0; i<16; i++)
	{
	  if(k==0)
	  {
	    delta[i]= (base<<23);	  
            k++;  		  
	  }
	  else
	  {
	    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0);
	    if(((val0>>4)&0x1)==1) { val0= (~val0&0xF)+1;  delta[i]=((base-(val0))<<23);}
	    else { delta[i]= ((base+val0)<<23);  }
	  }  
	}
      }
    }
    else
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
      if(tag==1)// zero-stream
      {
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, tag);
        for(int i=0; i<16; i++)
	{
           mask=(mask<<1);
	   if((i>=((tag>>4)&0xF))&(i<=((tag)&0xF)))
	   {
	      mask|=0x1;
	   }
	}
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	for(int i=0; i<16; i++)
	{
	   if(((mask>>(15-i))&0x1)==0)
	   {
	      if(k==0)
	      {
		delta[i]= (base<<23);      
	        k++;
	      }
	      else
	      {
		Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0);
	        if(((val0>>4)&0x1)==1) { val0= (~val0&0xF)+1; delta[i]= ((base-val0)<<23);}	
		else{ delta[i]= ((base+val0)<<23);}
	      }
	   }

	}

      }
      else // global-base
      {
	      
         Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
         for(int i=0; i<16; i++)
	 {
	   Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val0);
	   if(((val0>>3)&0x1)==1) {base=0x80;}
           else {base=0x78;}
           delta[i]= ((base+(val0&0x7))<<23);
	 }
      }
    }
  }
   
  mask=0;
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  if(tag==1)
  { 
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    mask=(tag==1)?((1<<16)-1):0; 
  }
  else 
  { 
    ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
  }
  for(int i=0; i<16; i++)
  {
     if(((mask>>(15-i))&0x1)==1)
     {
       delta[i]|=(1<<31); 
     }
  }

  budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));	  
  k=0;
  while(budget>0)
  {
    
    mask=0;
    k=(k<23)?k:22;
    if(k<3)
    {	    
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);

      if(tag==1)
      {
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
        mask=(tag==1)?((1<<16)-1):0;
        budget-=2; 
      }
      else
      {
        ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
        budget-=17;
      }
    
      for(int i=0; i<16; i++)
      {
        if(((mask>>(15-i))&0x1)==1)
        {
          delta[i]|=(1<<(22-k)); 
        }    
      }
      k++;
    }
    else
    {
      ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
      budget-=16;
      for(int i=0; i<16; i++)
      {
        if(((mask>>(15-i))&0x1)==1)
        {
          delta[i]|=(1<<(22-k));
        }
      }
      k++;
    }
  }    
  for(int i=0; i<16; i++)
  {
    data[16*idx+i]=delta[i];
  }
  return;

}

void FacetCompressCuda(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  FacetCompressCudaKernel<<<gridDim, blockDim>>>(data, c_data, M, byte_per_group);
  cudaDeviceSynchronize();
  return;
}

void FacetDecompressCuda(int32_t* data, uint8_t* c_data, int M, int byte_per_group)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  FacetDecompressCudaKernel<<<gridDim, blockDim>>>(data, c_data, M, byte_per_group);
  cudaDeviceSynchronize();
  return;
}


__global__ void CompressMaskCudaKernel(int32_t* data, uint8_t* mask, int M)
{
   int idx= blockDim.x*blockIdx.x+threadIdx.x;
   if(idx>=M) return;
   uint8_t temp=0;
   for(int i=0; i<16; i++) 
   {
      temp=temp<<1;	   
      if(((data[16*idx+i]>>23)&0xFF)>=254)
      {
        data[16*idx+i]=0;
	temp|=0x1;
      }
      if((i%8)==7)
      {
        mask[2*idx+(i/8)]=temp;
	temp=0;
      }
   }
   return;
}

void CompressMaskCuda(int32_t* data, uint8_t* mask, int M)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  CompressMaskCudaKernel<<<gridDim, blockDim>>>(data, mask, M); 
  cudaDeviceSynchronize();
  return;
}

__global__ void DecompressMaskCudaKernel(int32_t* data, uint8_t* mask, int M)
{
   int idx= blockDim.x*blockIdx.x+threadIdx.x;
   if(idx>=M) return;
   
   for(int i=0; i<16; i++) 
   {
      if(((mask[2*idx+(i/8)]>>(7-(i%8)))&0x1)==1)
      {
        data[16*idx+i]= (1<<31) | ((254&0xFF)<<23);
      }
   }
   return;
}
void DecompressMaskCuda(int32_t* data, uint8_t* mask, int M)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  DecompressMaskCudaKernel<<<gridDim, blockDim>>>(data, mask, M); 
  cudaDeviceSynchronize();
  return;
}

__global__ void CompressDropoutCudaKernel(uint8_t* data, uint8_t* c_data, int M)
{
  int idx= blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=M) return;
  
  for(int i=0; i<8;i++)
  {
    if(((data[8*idx+i])&0x1)==1) 
    {
      c_data[idx] |= 1<<(7-i); 
    }
  
  }
  return;
}

void CompressDropoutCuda(uint8_t* data, uint8_t* c_data, int M)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  CompressDropoutCudaKernel<<<gridDim, blockDim>>>(data, c_data, M);
  cudaDeviceSynchronize();
  return;
}

__global__ void DecompressDropoutCudaKernel(uint8_t* data, uint8_t* c_data, int M)
{
  int idx= blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>=M) return;

  for(int i=0; i<8;i++)
  {
    if(((data[idx]>>(7-i))&0x1)==1)
    {
      data[8*idx+i] = 1;
    }

  }
  return;
}

void DecompressDropoutCuda(uint8_t* data, uint8_t* c_data, int M)
{
  cudaSetDevice(0);
  dim3 gridDim((M+256-1)/256);
  dim3 blockDim(256);
  DecompressDropoutCudaKernel<<<gridDim, blockDim>>>(data, c_data, M);
  cudaDeviceSynchronize();
  return;
}



__global__ void ProfileCudaKernel(uint8_t* c_data, uint8_t* data, int* budget0, int* budget1, int M, int byte_per_group)
{
int idx= blockDim.x*blockIdx.x+threadIdx.x;
  if(idx>=M) return;
  
  int32_t delta[16]= {0};
  int current_byte= idx*byte_per_group;
  int current_idx=7;
  uint8_t base;
  uint8_t tag;
  uint8_t delta_bw0=0; 

  int mask=0;
  int sign=0;
  uint8_t temp=0;
  uint8_t val0=0;
  uint8_t val1=0;
  uint8_t val2=0;
  int temp_int=0;
  int q0=0;
    
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  
  if(tag==1)
  {
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag); 
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, delta_bw0);
    if(tag==1)
    {
      ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
      //if(idx==0) printf("MASK is %d", mask);
      data[idx]=3;
    }
    else
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val1);
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val2);
      data[idx]=2;
      for(int i=0; i<16; i++)
      {
	mask= mask<<1;      
        if((i>=val1)&(i<=val2))
	{
	  mask|=0x1;
	}
      
      }
    }
    val1=0;
    for(int i=0; i<16; i++)
    {
       if(((mask>>(15-i))&0x1)==0)
       {
	 Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, delta_bw0, val0);
         delta[i]= (base+val0)<<8;
	 delta[i]|=(val0);
	 val1++;
       } 
    } 
  }
  else
  {
    tag=0;	  
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 3, delta_bw0);

    if(tag==1)
    {
      //data[idx]=delta_bw0+val1+1;
      q0=2;	    
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 2, val1);
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, val1+delta_bw0+1, val2); //exp_bias
      data[idx]=1;	    
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
      if(tag==1)
      {
         Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, val0);      
      }
      else
      {
         Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val0);      
      }
      //if(idx==0) printf("val0 is %d!\n", val0);
      for(int i=0; i<16; i++)
      {
         if(((tag==1)&((i!=(val0&0xF))&(i!=((val0>>4)&0xF))))|((tag==0)&(i!=(val0&0xF))))
	 {
	    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, delta_bw0, temp);     
	    delta[i] = (base+val2+temp)<<8;
	    delta[i] |= (val2+temp); 
	 }
         else
	 {
	    delta[i] = base<<8;
	 }		 
      }
      //if(idx==0) {for(int i=0; i<16; i++){ printf("exp stored of idx %d is %d!\n", i, delta[i]>>8);} }
    }
    else
    {
      data[idx]=0;
      q0=1;      
      for(int i=0; i<16; i++)
      {
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, delta_bw0, val0);
	delta[i]= (base+val0)<<8;
	delta[i]|= val0;
      }
      //if(idx==0) {for(int i=0;i<16;i++) {printf("Before in Decomp delta of index %d is %d!\n", i, (delta[i]>>8)&0xFF); }}
    }
    val1=16;
  }
  if(q0==1)
  {	  
   int budget= (8*(current_byte-idx*byte_per_group)+(7-current_idx));
   budget0[idx]=budget;
  }

  if(q0==2)
  {	  
   int budget= (8*(current_byte-idx*byte_per_group)+(7-current_idx));
   budget1[idx]=budget;
  } 
 


  //int budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));
  //budget0[idx]=budget;
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  if(tag==1) { Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, val0); sign= (val0==1)?((1<<16)-1):0; }//budget0[idx]=2; }
  else
  {
    //budget0[idx]=17;  
    ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, sign);	  
  }
    //if(idx==0) printf("Sign is %d!\n",sign);
  
  //if(idx==0) {for(int i=0;i<16;i++) {printf("Before1 in Decomp delta of index %d is %d!\n", i, (delta[i])&0xFF); }}
  GetEachBudget(current_byte, current_idx, idx, byte_per_group, 16-val1, mask&(0xFFFF), delta);
  //if(idx==0) {for(int i=0; i<16; i++){ printf("exp stored of idx %d is %d!\n", i, delta[i]>>8);} }
  //if(idx==0) {for(int i=0;i<16;i++) {printf("After in Decomp delta of index %d is %d!\n", i, (delta[i])&0xFF); }}
  for(int i=0; i<16; i++)
  {
    ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, delta[i]&0xFF, temp_int);	  
    //data[16*idx+i] = (((sign>>(15-i))&0x1)<<31) | (((delta[i]>>8)&0xFF)<<23) | (temp_int<<(23-(delta[i]&0xFF)));
  }
  return;

}

__global__ void FacetProfileCudaKernel(uint8_t* c_data, uint8_t* data, int* budget0, int M, int byte_per_group)
{
  int idx= blockDim.x*blockIdx.x+threadIdx.x;
  if(idx>=M) return;
  int current_byte= idx*byte_per_group;
  int current_idx= 7;
  int delta[16]= {0};
    
  uint8_t base=0;
  uint8_t tag=0;
  uint8_t val0=0;
  int mask=0;
  int budget=0;
  int k =0;
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  if(tag==1)// Uncompressible
  {
    data[idx]=0; 	  
    for(int i=0; i<16; i++)
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, tag);
      delta[i]= ((tag)<<23);
    }
  }
  else
  {
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    if(tag==1)//local-base
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
      if(tag==1)//with-outlier
      {
	      
        data[idx]=1; 	  
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val0);
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, tag);
	delta[val0]= ((tag)<<23);
        mask=(1<<(val0));
	Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	
	for(int i=0; i<16; i++)
	{
          if(((mask>>i)&0x1)==0)
          {
	    if(k==0)
	    {
	      delta[i]= (base<<23);
	      k++;
	    }
	    else
	    {
              Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0);
	      if(((val0>>4)&0x1)==1) { val0= (~val0&0xF)+1;  delta[i]=((base-(val0))<<23);}
	      else { delta[i]= ((base+val0)<<23);  }
	    }
	  }		  
	}

      }
      else//without outlier
      {
	data[idx]=2;      
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	for(int i=0; i<16; i++)
	{
	  if(k==0)
	  {
	    delta[i]= (base<<23);	  
            k++;  		  
	  }
	  else
	  {
	    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0);
	    if(((val0>>4)&0x1)==1) { val0= (~val0&0xF)+1;  delta[i]=((base-(val0))<<23);}
	    else { delta[i]= ((base+val0)<<23);  }
	  }  
	}
      }
    }
    else
    {
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
      if(tag==1)// zero-stream
      {
	data[idx]=3;      
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, tag);
        for(int i=0; i<16; i++)
	{
           mask=(mask<<1);
	   if((i>=((tag>>4)&0xF))&(i<=((tag)&0xF)))
	   {
	      mask|=0x1;
	   }
	}
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 8, base);
	for(int i=0; i<16; i++)
	{
	   if(((mask>>(15-i))&0x1)==0)
	   {
	      if(k==0)
	      {
		delta[i]= (base<<23);      
	        k++;
	      }
	      else
	      {
		Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 5, val0);
	        if(((val0>>4)&0x1)==1) { val0= (~val0&0xF)+1; delta[i]= ((base-val0)<<23);}	
		else{ delta[i]= ((base+val0)<<23);}
	      }
	   }

	}

      }
      else // global-base
      {
	 data[idx]=4;     
         Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
         for(int i=0; i<16; i++)
	 {
	   Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 4, val0);
	   if(((val0>>3)&0x1)==1) {base=0x80;}
           else {base=0x78;}
           delta[i]= ((base+(val0&0x7))<<23);
	 }
      }
    }
  }
  int ab=0; 
  budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));	  
  budget0[idx]=budget;
  mask=0;
  Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
  if(tag==1)
  { 
    Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
    mask=(tag==1)?((1<<16)-1):0; 

  }
  else 
  { 
    ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
  }
  for(int i=0; i<16; i++)
  {
     if(((mask>>(15-i))&0x1)==1)
     {
       delta[i]|=(1<<31); 
     }
  }

  budget= byte_per_group*8-(8*(current_byte-idx*byte_per_group)+(7-current_idx));	  
  k=0;
  while(budget>0)
  {
    
    mask=0;
    k=(k<23)?k:22;
    if(k<3)
    {	    
      Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);

      if(tag==1)
      {
        Read8Bit(c_data, current_byte, current_idx, idx, byte_per_group, 1, tag);
        mask=(tag==1)?((1<<16)-1):0;
        budget-=2;
        ab+=2;	
      }
      else
      {
        ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
        budget-=17;
	ab+=17;
      }
    
      for(int i=0; i<16; i++)
      {
        if(((mask>>(15-i))&0x1)==1)
        {
          delta[i]|=(1<<(22-k)); 
        }    
      }
      k++;
    }
    else
    {
      ReadMultiBit(c_data, current_byte, current_idx, idx, byte_per_group, 16, mask);
      budget-=16;
      for(int i=0; i<16; i++)
      {
        if(((mask>>(15-i))&0x1)==1)
        {
          delta[i]|=(1<<(22-k));
        }
      }
      k++;
    }
  } 
  budget0[idx]=ab;  
  for(int i=0; i<16; i++)
  {
     //data[16*idx+i]=delta[i];
  }
  return;


}



void ProfileCuda(uint8_t* c_data, uint8_t* data, int* budget, int* budget1, int M, int byte_per_group)
{
   cudaSetDevice(0);
   dim3 gridDim((M+256-1)/256);
   dim3 blockDim(256);
   ProfileCudaKernel<<<gridDim, blockDim>>>(c_data, data, budget, budget1, M, byte_per_group);
   cudaDeviceSynchronize();
   return;
}

void FacetProfileCuda(uint8_t* c_data, uint8_t* data, int* budget, int M, int byte_per_group)
{
   cudaSetDevice(0);
   dim3 gridDim((M+256-1)/256);
   dim3 blockDim(256);
   FacetProfileCudaKernel<<<gridDim, blockDim>>>(c_data, data, budget, M, byte_per_group);
   cudaDeviceSynchronize();
   return;
}









