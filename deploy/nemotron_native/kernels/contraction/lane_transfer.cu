

#include <cuda_runtime.h>
#include "mk2_abi.cuh"
#include <cstdint>

struct LaneArgs {
 const uint16_t *embedding; uint16_t *packed_embedding;
 const uint32_t *noise; uint32_t *packed_noise;
 const int *rows, *slot; int *packed_slot;
 int *seq, *packed_seq;
 const int *tables[3]; int *packed_tables[3];
 int64_t *codes, *packed_codes; int *phones, *packed_phones;
 int full_width, narrow_width, blocks;
};
static LaneArgs arguments(const MK2Stage0Bindings &full, const MK2Stage0Bindings &small,
                          const int *rows, int full_width, int narrow_width) {
 LaneArgs a{};
 a.embedding=(const uint16_t*)full.embedding; a.packed_embedding=(uint16_t*)small.embedding;
 a.noise=(const uint32_t*)full.gumbel; a.packed_noise=(uint32_t*)small.gumbel;
 a.rows=rows; a.slot=(const int*)full.slot_map; a.packed_slot=(int*)small.slot_map;
 a.seq=(int*)full.seq_len; a.packed_seq=(int*)small.seq_len;
 for(int l=0;l<3;l++){a.tables[l]=(const int*)full.kv_block_table[l];a.packed_tables[l]=(int*)small.kv_block_table[l];}
 a.codes=(int64_t*)full.codes_out; a.packed_codes=(int64_t*)small.codes_out;
 a.phones=(int*)full.phoneme_out;a.packed_phones=(int*)small.phoneme_out;
 a.full_width=full_width;a.narrow_width=narrow_width;a.blocks=full.kv_max_blocks_per_seq;
 return a;
}
__global__ void gather_lanes(LaneArgs a) {
 int r=blockIdx.x, original=a.rows[r];
 for(int j=threadIdx.x;j<1536;j+=blockDim.x)
  a.packed_embedding[r*1536+j]=a.embedding[original*1536+j];
 for(int j=threadIdx.x;j<16*80;j+=blockDim.x)
  a.packed_noise[r*16*80+j]=a.noise[original*16*80+j];
 for(int l=0;l<3;l++) for(int j=threadIdx.x;j<2*a.blocks;j+=blockDim.x)
  a.packed_tables[l][r*2*a.blocks+j]=a.tables[l][original*2*a.blocks+j];
 if(threadIdx.x==0){a.packed_slot[r]=a.slot[original];a.packed_seq[r]=a.seq[original];}
}
__global__ void scatter_lanes(LaneArgs a) {
 int original=blockIdx.x, packed=-1;
 for(int r=0;r<a.narrow_width;r++) if(a.rows[r]==original){packed=r;break;}
 if(threadIdx.x<16) a.codes[original*16+threadIdx.x]=packed>=0?a.packed_codes[packed*16+threadIdx.x]:0;
 if(threadIdx.x==0){
  a.phones[original]=packed>=0?a.packed_phones[packed]:0;


  if(packed>=0) a.seq[original]=a.packed_seq[packed];
 }
}
static bool valid(const MK2Stage0Bindings *f,const MK2Stage0Bindings *s,const int *rows,int fw,int nw){
 return f&&s&&rows&&((fw==32&&nw==16)||(fw==64&&(nw==16||nw==32)))&&
 f->kv_max_blocks_per_seq==s->kv_max_blocks_per_seq&&f->kv_max_blocks_per_seq>0&&
 f->kv_tokens_per_block==32&&s->kv_tokens_per_block==32;
}
extern "C" size_t native_lanes_bindings_size(){return sizeof(MK2Stage0Bindings);}
extern "C" int native_lanes_gather(const MK2Stage0Bindings *f,const MK2Stage0Bindings *s,
 const int *rows,int fw,int nw,void *stream){
 if(!valid(f,s,rows,fw,nw))return (int)cudaErrorInvalidValue;
 gather_lanes<<<nw,256,0,(cudaStream_t)stream>>>(arguments(*f,*s,rows,fw,nw));
 return (int)cudaGetLastError();
}
extern "C" int native_lanes_scatter(const MK2Stage0Bindings *f,const MK2Stage0Bindings *s,
 const int *rows,int fw,int nw,void *stream){
 if(!valid(f,s,rows,fw,nw))return (int)cudaErrorInvalidValue;
 scatter_lanes<<<fw,32,0,(cudaStream_t)stream>>>(arguments(*f,*s,rows,fw,nw));
 return (int)cudaGetLastError();
}
