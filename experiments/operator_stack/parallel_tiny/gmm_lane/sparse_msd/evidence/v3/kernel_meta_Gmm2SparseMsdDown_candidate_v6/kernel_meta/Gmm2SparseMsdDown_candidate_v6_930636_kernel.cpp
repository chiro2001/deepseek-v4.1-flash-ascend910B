#ifndef __GROUPED_MATMUL__KERNEL_FUN_H__
#define __GROUPED_MATMUL__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#include "/work/gmm_lane_sparse_msd_build_20261011_v3_repro/private_impl/op_kernel/grouped_matmul.cpp"
#include "kernel_common.h"
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#define TEMPLATE_PARAMS -1
#define TEMPLATE_PARAMS_LEN 0

#if TILING_KEY_VAR == 69325233410UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 27, 0, 0, 1, 0, 1, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 69323529474UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 1, 0, 0, 1, 0, 1, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 34965495042UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 27, 0, 0, 1, 0, 1, 0, 1, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 34963791106UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 1, 0, 0, 1, 0, 1, 0, 1, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 69862104322UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 27, 0, 0, 1, 0, 2, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 69860400386UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 1, 0, 0, 1, 0, 2, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 36039236866UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 27, 0, 0, 1, 0, 3, 0, 1, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 36037532930UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 1, 0, 0, 1, 0, 3, 0, 1, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 70935846146UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 27, 0, 0, 1, 0, 4, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 70934142210UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 1, 0, 0, 1, 0, 4, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 71472717058UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 27, 0, 0, 1, 0, 5, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 71471013122UL
#undef TEMPLATE_PARAMS
#define TEMPLATE_PARAMS 2, 29, 1, 0, 0, 1, 0, 5, 0, 2, 0
#undef TEMPLATE_PARAMS_LEN
#define TEMPLATE_PARAMS_LEN 11
#endif

#if TILING_KEY_VAR == 69325233410UL && (defined(__DAV_CUBE__) && __NPU_ARCH__ == 2201)
    __gm__ struct OpSystemRunCfg g_opSystemRunCfg = {0};
#else
    extern __gm__ struct OpSystemRunCfg g_opSystemRunCfg;
#endif

__aicore__ inline __attribute__((always_inline)) void ascendc_auto_gen_gmm2_sparse_msd_kernel(GM_ADDR ffts_addr, GM_ADDR x_in__, GM_ADDR weight_in__, GM_ADDR bias_in__, GM_ADDR scale_in__, GM_ADDR group_list_in__, GM_ADDR per_token_scale_in__, GM_ADDR y_out_, GM_ADDR workspace) {
    AscendC::SetSysWorkspaceForce(workspace);
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_MC2_CTX));
#endif
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    icache_preload(1);
    if (ffts_addr != nullptr) {
        set_ffts_base_addr((uint64_t)ffts_addr);
    }
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_FFTS_ADDR));
#endif
#ifdef __MIX_CORE_MACRO__ 
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(workspace);
#ifdef ASCENDC_TIME_STAMP_ON
        AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_CLEAR_WK_SPAC));
#endif
    }
#endif
#if defined(TEMPLATE_PARAMS_LEN) && TEMPLATE_PARAMS_LEN != 0
    gmm2_sparse_msd<TEMPLATE_PARAMS>(x_in__, weight_in__, bias_in__, scale_in__,  nullptr,  nullptr,  nullptr, group_list_in__, per_token_scale_in__, y_out_, usrWorkspace, nullptr);
#else
    gmm2_sparse_msd(x_in__, weight_in__, bias_in__, scale_in__,  nullptr,  nullptr,  nullptr, group_list_in__, per_token_scale_in__, y_out_, usrWorkspace, nullptr);
#endif
}

extern "C" __global__ [aicore] void auto_gen_gmm2_sparse_msd_kernel(GM_ADDR ffts_addr, GM_ADDR x_in__, GM_ADDR weight_in__, GM_ADDR bias_in__, GM_ADDR scale_in__, GM_ADDR group_list_in__, GM_ADDR per_token_scale_in__, GM_ADDR y_out_, GM_ADDR workspace) {
    ascendc_auto_gen_gmm2_sparse_msd_kernel(ffts_addr, x_in__, weight_in__, bias_in__, scale_in__, group_list_in__, per_token_scale_in__, y_out_, workspace);
}


#if TILING_KEY_VAR == 69325233410UL
#if (defined(__DAV_CUBE__) && __NPU_ARCH__ == 2201)
static const struct FunLevelMixCoreType Gmm2SparseMsdDown_candidate_v6_mix_aic_section __attribute__ ((used, section (".ascend.meta.Gmm2SparseMsdDown_candidate_v6_mix_aic"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},    {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };
#endif
#endif
// generate dfx section for tiling_key:69325233410
struct AscendCInfoMetaDFX {
    BaseTlv head;
    uint8_t value[244];
};




#if TILING_KEY_VAR == 69325233410UL && (defined(__DAV_CUBE__) && __NPU_ARCH__ == 2201)
static const struct AscendCInfoMetaDFX Gmm2SparseMsdDown_candidate_v6_mix_aic_dfx_section __attribute__ ((used, section (".ascend.meta.Gmm2SparseMsdDown_candidate_v6_mix_aic"))) = {{4, 244}, { 0, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0, 11, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 0, 4, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 128, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 48, 0, 5, 0, 4, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 7, 0, 0, 0, } };
#endif


#if TILING_KEY_VAR == 69325233410UL
#if (defined(__DAV_VEC__) && __NPU_ARCH__ == 2201)
static const struct FunLevelMixCoreType Gmm2SparseMsdDown_candidate_v6_mix_aiv_section __attribute__ ((used, section (".ascend.meta.Gmm2SparseMsdDown_candidate_v6_mix_aiv"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN},    {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };
#endif
#endif
// generate dfx section for tiling_key:69325233410
#if TILING_KEY_VAR == 69325233410UL && (defined(__DAV_VEC__) && __NPU_ARCH__ == 2201)
static const struct AscendCInfoMetaDFX Gmm2SparseMsdDown_candidate_v6_mix_aiv_dfx_section __attribute__ ((used, section (".ascend.meta.Gmm2SparseMsdDown_candidate_v6_mix_aiv"))) = {{4, 244}, { 0, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0, 11, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 4, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 2, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 8, 0, 5, 0, 4, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 128, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 48, 0, 5, 0, 4, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 6, 0, 5, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 0, 2, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 0, 0, 7, 0, 0, 0, } };
#endif

#endif
