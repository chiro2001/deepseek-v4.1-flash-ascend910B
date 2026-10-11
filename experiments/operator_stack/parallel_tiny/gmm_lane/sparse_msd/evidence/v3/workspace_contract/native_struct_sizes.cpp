#include <cstdint>
#include <cstdio>
using GM_ADDR=uint8_t*;
struct TilingInfo {
    GM_ADDR tilingAddr;  // the GM address of the tiling.
};
struct MatmulOrgShape {
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
};
struct MatmulConfigParams {
    // offset->8
    uint32_t enAtomic : 8;
    uint32_t enSequentialWrite : 1;
    uint32_t isTransA : 1;
    uint32_t isTransB : 1;
    uint32_t enPartialSum : 1;
    uint32_t setTail : 1;
    uint32_t setTensorA : 1;
    uint32_t setTensorB : 1;
    uint32_t setTensorBias : 1;
    uint32_t setClearBias : 1;
    uint32_t cIsTscm : 1;
    uint32_t isFirstIter : 1;
    uint32_t sync : 1;
    uint32_t enHF32 : 1;
    uint32_t hf32TransMode : 1;
    uint32_t setQuant : 1;
    uint32_t setBatch : 1;
    uint32_t waitIterateAll : 1;
    uint32_t waitIterateBatch : 1;
    uint32_t iterateFakeMsg : 1;

    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    uint32_t sizeAmatrix;
    uint32_t sizeBmatrix;

    uint64_t aAddr;
    uint64_t bAddr;
    uint64_t cAddr;
    uint64_t biasAddr;  // 56 bytes, but access to the maximum message size (64 bytes)
    uint64_t quantAddr; // 64 bytes
    uint32_t quantSize;
    uint32_t quantMode;
    uint64_t quantScalar;        // 80 bytes
    uint32_t batchA;
    uint32_t batchB;
    uint32_t matrixStrideA;
    uint32_t matrixStrideB;
    uint32_t matrixStrideC;
    uint32_t batchLoop;
    uint32_t counterId; // query mm task process
    uint32_t reserved0;
    uint64_t dataPtr; // 120 bytes
};
struct MatmulUserDefInfo {
    uint64_t tilingPtr;
};
struct Conv3DBpInputConfigParams {
    uint32_t enAtomic : 8;
    uint32_t enSequentialWrite : 1;
    uint32_t setTensorWeight : 1;
    uint32_t sync : 1;
    uint32_t isFirstIter : 1;
    uint32_t setTensorFmap : 1;
    uint32_t setTensorOutBackprop : 1;
    uint32_t setSingleShape : 1;
    uint32_t setStartIdx : 1;
    uint32_t enPartialSum : 1;

    uint32_t singleShapeN;
    uint32_t singleShapeD;
    uint32_t curDinStartIdx;

    uint32_t singleShapeM;
    uint32_t singleShapeK;

    uint64_t weightAddr;
    uint64_t outBackpropAddr;
    uint64_t outputAddr;

    int32_t curHoStartIdx;
    uint32_t res;
    uint32_t res1[16];
};
struct Conv3DBpFilterConfigParams {
    uint32_t enAtomic : 8;
    uint32_t enSequentialWrite : 1;
    uint32_t setTensorFmap : 1;
    uint32_t setTensorOutBackprop : 1;
    uint32_t setSingleShape : 1;
    uint32_t sync : 1;
    uint32_t isFirstIter : 1;
    uint32_t waitIterateAll : 1;
    uint32_t enPartialSum : 1;

    uint32_t singleShapeM;
    uint32_t singleShapeN;
    uint32_t singleShapeK;

    uint64_t fmapAddr;
    uint64_t outBackpropAddr;
    uint64_t outputAddr;

    uint32_t curHoStartIdx;
    uint32_t fmapSize;
    uint32_t outBackpropSize;
    uint32_t res;

    uint32_t res1[16];
};
struct Conv3DForwardConfigParams {
    uint32_t enAtomic: 8; // for enable atomic add
    uint32_t enSequentialWrite: 1;
    uint32_t enSetTensorFmap: 1; // for enable SetTensorFmap intf
    uint32_t enSetTensorWeight: 1; // for enable SetTensorWeight intf
    uint32_t enSetTensorBias: 1; // for enable SetTensorBias intf
    uint32_t sync: 1; // for control sync aic/aiv
    uint32_t enSetSingleOutputShape: 1; // for enable SetSingleOutputShape intf
    uint32_t enSetFmapStartPosition: 1; // for enable SetFmapStartPosition intf
    uint32_t waitIterateAll: 1;
    uint32_t enPartialSum: 1; // for part of results of L0C write to gm
    uint32_t fmapSize;

    uint64_t fmapAddr;
    uint64_t weightAddr;
    uint64_t biasAddr;
    uint64_t outputAddr;

    uint32_t weightSize;
    uint32_t biasSize;
    uint32_t singleCoreBatch;
    uint32_t singleCo;
    uint32_t singleDo;
    uint32_t singleCoreM;
    uint32_t singleGroupOpt;
    uint32_t diStartPos;
    uint32_t mStartPos;
    uint32_t ciStartPos;
    uint32_t res[10]; // Reserved��for 120Byte ailgn
};
struct KfcMsg {
    uint32_t head = 0;
    int32_t ubAddr = -1;
    union {
        uint8_t buffer[120];  // Used for customizing and formatting messages
        TilingInfo tilingInfo;
        MatmulConfigParams body;
        MatmulOrgShape orgShape;
        MatmulUserDefInfo userDefInfo;
        Conv3DBpInputConfigParams convBpInputBody;
        Conv3DBpFilterConfigParams convBpFilterBody;
        Conv3DForwardConfigParams convForwardBody;
    };
};
struct MsgUBAvailable {
    uint32_t head;
    uint32_t res;
    uint8_t buffer[56];
};
struct MsgMatmulCnt {
    uint32_t head;
    uint32_t res;
    uint8_t buffer[56];
};
int main() {
    static_assert(sizeof(void*)==8);
    const uint64_t msg=(sizeof(KfcMsg)+31)/32*32;
    const uint64_t ub=(sizeof(MsgUBAvailable)+31)/32*32;
    const uint64_t cnt=(sizeof(MsgMatmulCnt)+31)/32*32;
    const uint64_t stride=2*64*msg*2;
    const uint64_t ub_offset=50*2*64*msg + 50*8*cnt;
    const uint64_t msg_end=24*stride;
    const uint64_t ub_end=ub_offset+24*2*ub;
    std::printf("{\"KfcMsg_bytes\":%llu,\"MsgUBAvailable_bytes\":%llu,\"MsgMatmulCnt_bytes\":%llu,"
        "\"msg_clear_end_exclusive\":%llu,\"ub_clear_end_exclusive\":%llu}\n",
        (unsigned long long)sizeof(KfcMsg),(unsigned long long)sizeof(MsgUBAvailable),
        (unsigned long long)sizeof(MsgMatmulCnt),(unsigned long long)msg_end,(unsigned long long)ub_end);
}
