#include "native_stage1.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
#include <new>
#include <string>
#include <vector>
#ifndef NATIVE_CODEC_PREFETCH
#define NATIVE_CODEC_PREFETCH 1
#endif
#ifndef NATIVE_CODEC_PACK_SUBCTAS
#define NATIVE_CODEC_PACK_SUBCTAS 1
#endif
namespace cg=cooperative_groups;
static_assert(SMALL_NATIVE_BATCH == 2 || SMALL_NATIVE_BATCH == 4 || SMALL_NATIVE_BATCH == 8);
constexpr int PHYSICAL_THREADS=512,MAIN_ARENA=227840,AUX_BYTES=4096,META_OFFSET=231936,TOTAL_SHARED=232192;
constexpr int SLICE_BYTES=MAIN_ARENA+AUX_BYTES;
constexpr int MAX_SUBCTA=8;
struct NativeMeta {
    dim3 grid,threads;
    uint3 block[MAX_SUBCTA];
    NativeStage1Status *status;const int32_t *lengths;int padded_frames;
    int submask,tshift,tmask,slice;
    int resident;
};
static_assert(sizeof(NativeMeta)<=TOTAL_SHARED-META_OFFSET);
__device__ __forceinline__ char *native_base(){extern __shared__ __align__(16) char arena[];return arena;}
__device__ __forceinline__ NativeMeta *native_meta(){return reinterpret_cast<NativeMeta*>(native_base()+META_OFFSET);}
__device__ __forceinline__ int native_subid(){const NativeMeta *m=native_meta();return (int)((threadIdx.x>>m->tshift)&(unsigned)m->submask);}
__device__ __forceinline__ char *native_arena(){const NativeMeta *m=native_meta();return native_base()+native_subid()*m->slice;}
__device__ __forceinline__ char *native_aux(){const NativeMeta *m=native_meta();return native_base()+native_subid()*m->slice+(m->slice-AUX_BYTES);}
__device__ __forceinline__ uint3 native_tid(){uint3 r;r.x=threadIdx.x&(unsigned)native_meta()->tmask;r.y=0u;r.z=0u;return r;}
__device__ __forceinline__ dim3 native_block(){const uint3 b=native_meta()->block[native_subid()];return dim3(b.x,b.y,b.z);}
__device__ __forceinline__ dim3 native_grid(){return native_meta()->grid;}
__device__ __forceinline__ dim3 native_threads(){return native_meta()->threads;}
__device__ __forceinline__ int native_resident(){return native_meta()->resident;}
__device__ __forceinline__ void native_local_sync(){
    const NativeMeta *m=native_meta();unsigned n=m->threads.x,id=15u-(unsigned)native_subid();
    asm volatile("bar.sync %0, %1;"::"r"(id),"r"(n):"memory");
}
__device__ __forceinline__ dim3 &native_config_grid(cudaLaunchConfig_t &c){return c.gridDim;}
__device__ __forceinline__ dim3 &native_config_threads(cudaLaunchConfig_t &c){return c.blockDim;}
__device__ __forceinline__ int native_true_tokens(int row,int padded_tokens){
    const NativeMeta *m=native_meta();
    return (padded_tokens/m->padded_frames)*m->lengths[row];
}
__device__ __forceinline__ void native_warm(const char *p,size_t n,int k,int nk,int me){
    if(p==nullptr||n<16||nk<=0)return;
    const size_t a=(size_t)p&15u;p-=a;n=(n+a)&~(size_t)15;
    const uint4 *q=reinterpret_cast<const uint4*>(p);
    const size_t cnt=n>>4,step=(size_t)nk*PHYSICAL_THREADS;
    uint32_t acc=0u;
    for(size_t i=(size_t)k*PHYSICAL_THREADS+(size_t)me;i<cnt;i+=step){
        const uint4 v=__ldg(q+i);acc^=v.x^v.y^v.z^v.w;
    }
    asm volatile(""::"r"(acc));
}
template<class F,class... A> __device__ __forceinline__ void native_invoke_pf(F fn,dim3 g,dim3 t,size_t shared,const void *pf,size_t pfb,A&&... args){
    const int me=threadIdx.x,owner=blockIdx.x,owners=gridDim.x;
    const cg::grid_group all=cg::this_grid();NativeMeta *meta=native_meta();
    if(t.x>512||t.y!=1||t.z!=1||t.x%32||shared>MAIN_ARENA){
        if(me==0&&owner==0)meta->status->code=10;
        all.sync();return;
    }
    int sub=1,tshift=9;
    const unsigned tx=t.x;
    const int vtotal=(int)(g.x*g.y*g.z);
    if(NATIVE_CODEC_PACK_SUBCTAS && (tx&(tx-1))==0u){
        int s=0;while((1u<<s)<tx)++s;tshift=s;
        int cand=(int)(512u/tx);if(cand>MAX_SUBCTA)cand=MAX_SUBCTA;
        const int want=(vtotal+owners-1)/owners;
        while(cand>1&&cand>want)cand>>=1;
        while(cand>1&&shared>(size_t)(SLICE_BYTES/cand-AUX_BYTES))cand>>=1;
        sub=cand;
    }
    const int slice=(SLICE_BYTES/sub)&~127;
    if(me==0){meta->grid=g;meta->threads=t;meta->submask=sub-1;meta->tshift=tshift;
              meta->tmask=(int)((1u<<tshift)-1u);meta->slice=slice;}
    const int total=vtotal,grp=me>>tshift;
    const bool holder=me<sub*(int)t.x,lead=holder&&((me&((1<<tshift)-1))==0);
    const int busy=(total+sub-1)/sub;
    if(NATIVE_CODEC_PREFETCH && owner>=busy&&pf!=nullptr)native_warm((const char*)pf,pfb,owner-busy,owners-busy,me);
    for(int base=owner*sub;base<total;base+=owners*sub){
        const int i=base+grp;
        if(lead&&i<total)meta->block[grp]=make_uint3(i%g.x,(i/g.x)%g.y,i/(g.x*g.y));
        __syncthreads();
        if(holder&&i<total){
            fn(static_cast<A&&>(args)...);
            asm volatile("cp.async.wait_all;":::"memory");
            if(t.x>=128)asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
        }
        __syncthreads();
    }
    all.sync();
}
template<class F,class... A> __device__ __forceinline__ void native_invoke(F fn,dim3 g,dim3 t,size_t shared,A&&... args){
    native_invoke_pf(fn,g,t,shared,nullptr,(size_t)0,static_cast<A&&>(args)...);
}
template<class F,class... A> __device__ __forceinline__ void native_cudaLaunchKernelEx(cudaLaunchConfig_t *c,F fn,A... args){
    native_invoke(fn,c->gridDim,c->blockDim,c->dynamicSmemBytes,args...);
}
template<class F,class... A> __device__ __forceinline__ void native_cudaLaunchKernelEx_pf(cudaLaunchConfig_t *c,F fn,const void *pf,size_t pfb,A... args){
    native_invoke_pf(fn,c->gridDim,c->blockDim,c->dynamicSmemBytes,pf,pfb,args...);
}
#define blockIdx native_block()
#define gridDim native_grid()
#define blockDim native_threads()
#define threadIdx native_tid()
#define __syncthreads() native_local_sync()
#include "legacy_codec.cuh"
#undef __syncthreads
#undef threadIdx
#undef blockDim
#undef gridDim
#undef blockIdx
namespace {
struct FastArgs {
    CodecArgs codec;Plan plan;RepackSet repack;int repack_grid;int input_frames;int blocks;
    const int64_t *codes;const int32_t *lengths,*slots;
    const __half *state[26];__half *next[26];float *pcm;int32_t *samples;
    NativeStage1Status *status;int nslots;
    int channels[26],histories[26];
};
static_assert(sizeof(FastArgs)<=4096,"precomputed kernel argument must fit the conservative parameter budget");
struct Handle {FastArgs args{};Plan plans[17]{};RepackSet repacks[17]{};int repack_grids[17]{};void *scratch=nullptr;size_t bytes=0;void *immutable=nullptr;size_t immutable_bytes=0;int setup_launches=0;std::vector<void*> packed;int max_frames=0,blocks=0,device=-1;};
thread_local std::string error;
bool check(cudaError_t e,const char *where){if(e==cudaSuccess)return true;error=std::string(where)+": "+cudaGetErrorString(e);return false;}
__device__ void reject(NativeStage1Status *s,int code,int row,int frame,int detail){
    if(atomicCAS(&s->code,0,code)==0){s->row=row;s->frame=frame;s->detail=detail;}
}
__global__ __launch_bounds__(512,1) void stage1_fast_persistent(const __grid_constant__ FastArgs a){
    const auto all=cg::this_grid();size_t me=(size_t)blockIdx.x*512+threadIdx.x,step=(size_t)gridDim.x*512;
    if(threadIdx.x==0){native_meta()->status=a.status;native_meta()->lengths=a.lengths;native_meta()->padded_frames=a.codec.F;native_meta()->resident=a.blocks;}
    if(me==0)*a.status=NativeStage1Status{0,-1,-1,0};
    all.sync();const int B=a.codec.B,F=a.codec.F,inputF=a.input_frames;
    for(int row=me;row<B;row+=step){
        int n=a.lengths[row],slot=a.slots[row];
        if(n<0||n>inputF)reject(a.status,8,row,-1,n);
        if(slot<0||slot>=a.nslots)reject(a.status,2,row,-1,slot);
        for(int j=0;j<row;++j)if(slot==a.slots[j])reject(a.status,3,row,-1,slot);
        a.samples[row]=(n>=0&&n<=inputF)?n*1764:0;
    }
    for(size_t i=me;i<(size_t)B*inputF;i+=step){
        int row=i/inputF,t=i%inputF;bool any=false;
        for(int c=0;c<16;++c){int64_t v=a.codes[i*16+c];if(v<0||v>=1024)reject(a.status,4,row,t,c);any|=v!=0;}
        if(t<a.lengths[row]&&!any)reject(a.status,5,row,t,0);
        if(t>=a.lengths[row]&&any)reject(a.status,6,row,t,0);
    }
    for(int p=0;p<26;++p)for(size_t i=me;i<(size_t)a.nslots*a.channels[p]*a.histories[p];i+=step)a.next[p][i]=a.state[p][i];
    for(size_t i=me;i<(size_t)B*inputF*1764;i+=step)a.pcm[i]=0.f;
    all.sync();if(a.status->code)return;
    for(int p=0;p<26;++p){
        const size_t stride=(size_t)a.channels[p]*a.histories[p];
        __half *dest=(__half*)a.codec.st[p];
        for(size_t i=me;i<(size_t)B*stride;i+=step)dest[i]=a.state[p][(size_t)a.slots[i/stride]*stride+i%stride];
    }
    __half *latent=(__half*)a.codec.inp[0];int T=2*F;
    for(size_t i=me;i<(size_t)B*40*T;i+=step){
        int t=i%T,c=(i/T)%40,row=i/((size_t)40*T);float v=0.f;
        if(t/2<a.lengths[row]){int code=a.codes[((size_t)row*inputF+t/2)*16+2*(c/5)+t%2];v=(((code>>(2*(c%5)))&3)-2)*.5f;}
        latent[i]=__float2half_rn(v);
    }
    all.sync();
    codec_program(a.codec,a.plan,a.repack,a.repack_grid);
    if(a.status->code)return;
    for(int p=0;p<26;++p){
        const size_t stride=(size_t)a.channels[p]*a.histories[p];const __half *src=(const __half*)a.codec.ns[p];
        for(size_t i=me;i<(size_t)B*stride;i+=step){int row=i/stride;if(a.lengths[row])a.next[p][(size_t)a.slots[row]*stride+i%stride]=src[i];}
    }
    const __half *wave=(const __half*)a.codec.waveform;
    for(size_t i=me;i<(size_t)B*inputF*1764;i+=step){
        const int row=i/(inputF*1764),t=i%(inputF*1764);
        if(t<a.lengths[row]*1764)a.pcm[i]=__half2float(wave[(size_t)row*F*1764+t]);
    }
}
__global__ __launch_bounds__(512,1) void stage1_prepare_immutable(const __grid_constant__ FastArgs a){
    const auto all=cg::this_grid();
    if(threadIdx.x==0){native_meta()->status=a.status;native_meta()->resident=a.blocks;}
    if(blockIdx.x==0&&threadIdx.x==0)*a.status=NativeStage1Status{0,-1,-1,0};
    all.sync();
    native_invoke(repack_rs_kernel<7>,dim3(a.repack_grid),dim3(128),MAIN_ARENA,
        a.repack,H16(a.codec.inp[0]),H16(a.codec.st[0]),a.plan.LB,
        40,a.plan.s0.T,a.plan.s0.L,a.plan.s0.SC,2,a.codec.B);
    native_invoke(repack3_rs_kernel,dim3(1152,12),dim3(128),MAIN_ARENA,
        H16(a.codec.inp[3]),H16(a.codec.inp[6]),a.plan.W3i,a.plan.W3s,
        a.plan.A,a.plan.Ac,a.plan.s1.SC,a.plan.s1.SC,DIVUP(a.plan.s1.SC,128));
}
bool stack(Handle *h,int slot,const NativeStage1Weights *w,int first,int field,size_t elements){
    const int ids[7]={first,first+3,first+5,first+7,first+9,first+11,first+13};
    const size_t stride=elements*(field==0?sizeof(__half):sizeof(float));void *ptr=nullptr;
    if(!check(cudaMalloc(&ptr,7*stride),"packed-weight allocation"))return false;
    h->packed.push_back(ptr);h->args.codec.inp[slot]=ptr;
    for(int i=0;i<7;++i){const void *src=field==0?w[ids[i]].weight:field==1?(const void*)w[ids[i]].bias:(const void*)w[ids[i]].alpha;
        if(!check(cudaMemcpy((char*)ptr+i*stride,src,stride,cudaMemcpyDeviceToDevice),"immutable weight stack"))return false;}
    return true;
}
void release(Handle *h){if(h->immutable)cudaFree(h->immutable);if(h->scratch)cudaFree(h->scratch);for(void *p:h->packed)cudaFree(p);delete h;}
}
extern "C" void *native_stage1_create(int b,int maxf,int slots,int blocks,const NativeStage1Weights *w){
    error.clear();if((b!=SMALL_NATIVE_BATCH)||maxf<1||maxf>16||slots<b||slots>4096||blocks<0||!w){error="invalid create arguments";return nullptr;}
    for(int p=0;p<26;++p)if(!w[p].weight||!w[p].bias||(p!=0&&p!=25&&!w[p].alpha)){error="missing weight pointer";return nullptr;}
    Handle *h=new(std::nothrow)Handle;if(!h){error="host allocation";return nullptr;}
    h->max_frames=maxf;h->args.nslots=slots;h->args.codec.B=b;
    const int cs[26]={40,768,1536,768,768,1536,768,1536,768,1536,768,1536,768,1536,768,1536,768,384,768,384,128,256,128,32,64,32};
    const int hs[26]={2,2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,1,6,6,1,6,6,1,6,6,6};
    cudaDeviceProp prop{};int occupancy=0,coop=0;
    if(!check(cudaGetDevice(&h->device),"device")||!check(cudaGetDeviceProperties(&prop,h->device),"properties")||
       !check(cudaDeviceGetAttribute(&coop,cudaDevAttrCooperativeLaunch,h->device),"cooperative support")||
       !check(cudaFuncSetAttribute(stage1_fast_persistent,cudaFuncAttributeMaxDynamicSharedMemorySize,TOTAL_SHARED),"shared-memory optin")||
       !check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,stage1_fast_persistent,512,TOTAL_SHARED),"occupancy")) {release(h);return nullptr;}
    if(prop.major!=9||prop.minor!=0||!coop||occupancy<1){error="requires resident SM90 cooperative kernel";release(h);return nullptr;}
    h->blocks=blocks?blocks:occupancy*prop.multiProcessorCount;
    if(h->blocks>occupancy*prop.multiProcessorCount){error="grid exceeds cooperative occupancy";release(h);return nullptr;}
    h->args.blocks=h->blocks;
    const int capacityF=maxf<=3?3:(maxf<=5?5:std::max(8,maxf));
    size_t decoder=codec_scratch_bytes(b,capacityF),halfs=(size_t)b*40*2*capacityF+(size_t)b*1764*capacityF;
    for(int p=0;p<26;++p){h->args.channels[p]=cs[p];h->args.histories[p]=hs[p];halfs+=2*(size_t)b*cs[p]*hs[p];}
    h->bytes=decoder+halfs*sizeof(__half);
    if(!check(cudaMalloc(&h->scratch,h->bytes),"scratch allocation")){release(h);return nullptr;}
    if(!check(cudaMemset(h->scratch,0,h->bytes),"initial scratch padding")){release(h);return nullptr;}
    h->args.codec.scratch=h->scratch;__half *q=(__half*)((char*)h->scratch+decoder);
    h->args.codec.inp[0]=q;q+=(size_t)b*40*2*capacityF;h->args.codec.waveform=q;q+=(size_t)b*1764*capacityF;
    for(int p=0;p<26;++p){h->args.codec.st[p]=q;q+=(size_t)b*cs[p]*hs[p];h->args.codec.ns[p]=q;q+=(size_t)b*cs[p]*hs[p];}
    const void **in=h->args.codec.inp;
    in[1]=w[0].weight;in[2]=w[0].bias;
    if(!stack(h,3,w,1,0,(size_t)1536*768*3)||!stack(h,4,w,1,1,1536)||!stack(h,5,w,1,2,768)||
       !stack(h,6,w,2,0,(size_t)768*1536*3)||!stack(h,7,w,2,1,768)||!stack(h,8,w,2,2,384)){release(h);return nullptr;}
    const int upops[4]={3,16,19,22};for(int i=0;i<4;++i){in[9+3*i]=w[upops[i]].weight;in[10+3*i]=w[upops[i]].bias;in[11+3*i]=w[upops[i]].alpha;}
    for(int i=0;i<3;++i){int p=17+3*i;in[21+6*i]=w[p].weight;in[22+6*i]=w[p].bias;in[23+6*i]=w[p].alpha;
        in[24+6*i]=w[p+1].weight;in[25+6*i]=w[p+1].bias;in[26+6*i]=w[p+1].alpha;}
    in[39]=w[25].weight;in[40]=w[25].bias;
    __half *immutable_w7[6]{};__half *immutable_w3i=nullptr,*immutable_w3s=nullptr;
    if(capacityF>=3){
        size_t wh=0;for(int j=0;j<6;++j)wh+=(size_t)native_RP_COUTP(j)*native_RP_CIN(j)*7;
        const size_t IW=(size_t)3*1536*768;wh+=14*IW;
        h->immutable_bytes=wh*sizeof(__half);
        if(!check(cudaMalloc(&h->immutable,h->immutable_bytes+sizeof(NativeStage1Status)),"immutable weight allocation")){release(h);return nullptr;}
        if(!check(cudaMemset(h->immutable,0,h->immutable_bytes+sizeof(NativeStage1Status)),"immutable padding initialization")){release(h);return nullptr;}
        __half *q=(__half*)h->immutable;
        for(int j=0;j<6;++j){immutable_w7[j]=q;q+=(size_t)native_RP_COUTP(j)*native_RP_CIN(j)*7;}
        immutable_w3i=q+IW;q+=7*IW;immutable_w3s=q+IW;q+=7*IW;
        if((char*)q!=(char*)h->immutable+h->immutable_bytes){error="immutable layout size mismatch";release(h);return nullptr;}
    }
    for(int f:{3,5,8}) if(f<=capacityF){
        Plan p=build_plan(b,f,static_cast<__half*>(h->scratch));
        for(int j=0;j<6;++j)p.W7[j]=immutable_w7[j];p.W3i=immutable_w3i;p.W3s=immutable_w3s;
        const void *const *W=h->args.codec.inp;
        RepackSet js{};
        const void *SRC[RP_JOBS] = {W[21], W[24], W[27], W[30], W[33], W[36]};
        for (int j = 0; j < RP_JOBS; ++j) {
            js.src[j] = H16(SRC[j]);
            js.dst[j] = p.W7[j];
            js.Cout[j] = native_RP_COUT(j);
            js.CoutP[j] = native_RP_COUTP(j);
            js.Cin[j] = native_RP_CIN(j);
        }
        int acc = 0;
        for (int j = 0; j < RP_JOBS; ++j) {
            js.base[j] = acc;
            js.nch[j] = native_RP_CIN(j) / 16;
            js.rs[j] = (j < 4) ? 1 : 0;
            acc += (native_RP_COUTP(j) / 16) * DIVUP(native_RP_CIN(j), 64);
        }
        js.k3base = acc;
        js.k3src[0] = H16(W[3]);
        js.k3src[1] = H16(W[6]);
        js.k3dst[0] = p.W3i ? p.W3i - (size_t)3 * 1536 * 768 : nullptr;
        js.k3dst[1] = p.W3s ? p.W3s - (size_t)3 * 768 * 1536 : nullptr;
        const int k3n = p.W3i ? 2304 : 0;
        h->plans[f]=p;h->repacks[f]=js;h->repack_grids[f]=acc+k3n;
    }
    if(capacityF>=3){
        FastArgs a=h->args;a.codec.F=3;a.plan=h->plans[3];a.repack=h->repacks[3];a.repack_grid=h->repack_grids[3];
        a.status=(NativeStage1Status*)((char*)h->immutable+h->immutable_bytes);
        int setup_occupancy=0;
        if(!check(cudaFuncSetAttribute(stage1_prepare_immutable,cudaFuncAttributeMaxDynamicSharedMemorySize,TOTAL_SHARED),"setup shared-memory optin")||
           !check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&setup_occupancy,stage1_prepare_immutable,512,TOTAL_SHARED),"setup occupancy")||
           setup_occupancy<1||h->blocks>setup_occupancy*prop.multiProcessorCount){
            if(error.empty())error="setup grid exceeds cooperative occupancy";release(h);return nullptr;}
        void *params[]={&a};
        if(!check(cudaLaunchCooperativeKernel((void*)stage1_prepare_immutable,dim3(h->blocks),dim3(512),params,TOTAL_SHARED,nullptr),"immutable setup launch")||
           !check(cudaStreamSynchronize(nullptr),"immutable setup completion")){release(h);return nullptr;}
        NativeStage1Status status{};
        if(!check(cudaMemcpy(&status,a.status,sizeof(status),cudaMemcpyDeviceToHost),"immutable setup status")){release(h);return nullptr;}
        if(status.code){error="immutable setup device status "+std::to_string(status.code);release(h);return nullptr;}
        h->setup_launches=1;
    }
    return h;
}
extern "C" int native_stage1_run(void *handle,const int64_t *codes,const int32_t *lengths,const int32_t *slots,
    const void *const *states,void *const *next,float *pcm,int32_t *samples,NativeStage1Status *status,int f,void *stream){
    error.clear();Handle *h=(Handle*)handle;
    if(!h||!codes||!lengths||!slots||!states||!next||!pcm||!samples||!status){error="null run argument";return 1;}
    if(f<1||f>8||f>h->max_frames){error="true-state variant supports public F1..8";return 1;}
    const int paddedF=f<=3?3:(f<=5?5:8);
    int device=-1;if(!check(cudaGetDevice(&device),"device")||device!=h->device){error="device mismatch";return 1;}
    FastArgs a=h->args;a.input_frames=f;a.plan=h->plans[paddedF];a.repack=h->repacks[paddedF];a.repack_grid=h->repack_grids[paddedF];a.codec.F=paddedF;a.codes=codes;a.lengths=lengths;a.slots=slots;a.pcm=pcm;a.samples=samples;a.status=status;
    for(int p=0;p<26;++p){if(!states[p]||!next[p]||states[p]==next[p]){error="invalid/aliased state";return 1;}a.state[p]=(const __half*)states[p];a.next[p]=(__half*)next[p];}
    void *params[]={&a};if(!check(cudaLaunchCooperativeKernel((void*)stage1_fast_persistent,dim3(h->blocks),dim3(512),params,TOTAL_SHARED,(cudaStream_t)stream),"fast Stage1 launch"))return 1;
    return check(cudaPeekAtLastError(),"immediate launch status")?0:1;
}
extern "C" int native_stage1_destroy(void *p){error.clear();if(!p)return 0;release((Handle*)p);return check(cudaPeekAtLastError(),"destroy status")?0:1;}
extern "C" const char *native_stage1_last_error(){return error.c_str();}
extern "C" size_t native_stage1_scratch_bytes(void *p){return p?((Handle*)p)->bytes:0;}
extern "C" int native_stage1_resident_blocks(void *p){return p?((Handle*)p)->blocks:0;}
extern "C" int native_stage1_setup_kernel_launches(void *p){return p?((Handle*)p)->setup_launches:0;}
extern "C" size_t native_stage1_immutable_weight_bytes(void *p){return p?((Handle*)p)->immutable_bytes:0;}
