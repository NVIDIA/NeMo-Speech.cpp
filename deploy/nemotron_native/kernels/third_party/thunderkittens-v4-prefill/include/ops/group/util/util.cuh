




#include "sync.cuh"




#if defined(KITTENS_SM10X) || defined(KITTENS_SM120)

struct clc {








__device__ static inline void schedule(kittens::clc::handle &h, kittens::semaphore &sem) {
    if (laneid() == 0) {
        asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value))), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem)))
            : "memory"
        );
    }
}





__device__ static inline kittens::clc::result query(kittens::clc::handle &h) {
    kittens::clc::result r;
    asm volatile(
        "{\n"
        ".reg .pred SUCCESS;\n"
        ".reg .b128 CLC_HANDLE;\n"
        "ld.shared.b128 CLC_HANDLE, [%4];\n"
        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 SUCCESS, CLC_HANDLE;\n"
        "selp.u32 %0, 1, 0, SUCCESS;\n"
        "@!SUCCESS bra.uni DONE;\n"
        "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, CLC_HANDLE;\n"
        "fence.proxy.async.shared::cta;\n"
        "DONE:\n"
        "}"
        : "=r"(r.success), "=r"(r.x), "=r"(r.y), "=r"(r.z)
        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value)))
        : "memory"
    );
    return r;
}

};

#endif

#if defined(KITTENS_SM90) || defined(KITTENS_SM10X) || defined(KITTENS_SM120)









struct pdl {








__device__ static inline void arrive() {
    if (laneid() == 0) {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
}




__device__ static inline void wait() {
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

};

#endif
