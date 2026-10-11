// Compile the actual bridge bind/submit/finish against CPU-only intercepted APIs.
// The independent expected payloads are 27 successful v5 runtime receipts.
#ifndef PERF_BRIDGE_SOURCE
#define PERF_BRIDGE_SOURCE "perf_bridge.cpp"
#endif
#include PERF_BRIDGE_SOURCE
#include "v5_actual_args_fixture.h"
#include <map>
#include <vector>
#include <stdexcept>
#include <cstddef>

static_assert(sizeof(Gmm2MsdArgsV3)==72 && sizeof(Gmm2PlannerArgsV3)==16);
static_assert(offsetof(Gmm2MsdArgsV3,ffts)==0);
static_assert(offsetof(Gmm2MsdArgsV3,x_list)==8);
static_assert(offsetof(Gmm2MsdArgsV3,weight_list)==16);
static_assert(offsetof(Gmm2MsdArgsV3,assist_list)==24);
static_assert(offsetof(Gmm2MsdArgsV3,scale_list)==32);
static_assert(offsetof(Gmm2MsdArgsV3,counts)==40);
static_assert(offsetof(Gmm2MsdArgsV3,token_scale)==48);
static_assert(offsetof(Gmm2MsdArgsV3,y_list)==56);
static_assert(offsetof(Gmm2MsdArgsV3,workspace_system_base)==64);
static_assert(offsetof(Gmm2PlannerArgsV3,counts)==0 && offsetof(Gmm2PlannerArgsV3,frame)==8);

namespace {
const V5RecordedLaunch *fixture{};
Probe *active_owner{};
std::map<uint64_t,std::array<uint64_t,4>> descriptors;
std::vector<std::string> calls;
uint64_t checks=0, rejected_output_gm=0;
void require(bool ok,const char *reason){++checks;if(!ok)throw std::runtime_error(reason);}
void compare_payload(const Gmm2MsdArgsV3 &args){
    const uint64_t *actual=reinterpret_cast<const uint64_t *>(&args);
    for(unsigned i=0;i<9;++i){
        if(actual[i]!=fixture->args[i]){
            std::cerr<<"field mismatch: index="<<i<<" offset="<<8*i<<" actual="<<actual[i]<<" v5="<<fixture->args[i]<<'\n';
            require(false,"actual compiled bind differs from independent v5 successful payload");
        }
        require(true,"field");
    }
    require(std::memcmp(&args,fixture->args.data(),72)==0,"72B payload differs");
}
}

extern "C" const char *aclGetRecentErrMsg(){return "CPU mock only";}
extern "C" aclError aclrtGetCurrentContext(aclrtContext *context){*context=active_owner->borrowed_context;return 0;}
extern "C" aclError aclrtMemcpy(void *dst,size_t max,const void *src,size_t bytes,aclrtMemcpyKind kind){
    require(max>=bytes,"copy extent");
    calls.push_back(kind==ACL_MEMCPY_HOST_TO_DEVICE?"H2D":"D2H");
    if(kind==ACL_MEMCPY_HOST_TO_DEVICE){
        require(bytes==32,"bind writes a 32B GM descriptor");
        std::array<uint64_t,4> data{};std::memcpy(data.data(),src,32);descriptors[addr(dst)]=data;
    }else{
        auto it=descriptors.find(addr(const_cast<void *>(src)));
        if(it!=descriptors.end())std::memcpy(dst,it->second.data(),bytes);
        else std::memset(dst,0xa5,bytes); // CPU canary fixture; no numeric claim.
    }
    return 0;
}
extern "C" aclError aclrtMemset(void *,size_t max,int32_t,size_t bytes){
    require(max>=bytes,"memset extent");calls.push_back("clear");return 0;
}
extern "C" aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream,int32_t){
    require(stream==active_owner->borrowed_stream,"stream owner changed");calls.push_back("sync");return 0;
}
extern "C" int gmm2_launch_v3_host_args(void *function,void *stream,void *host,size_t bytes,unsigned blocks,unsigned schedule){
    require(stream==active_owner->borrowed_stream,"launch stream differs from borrowed SDK stream");
    if(function==active_owner->functions.planner){
        calls.push_back("planner");
        require(host==&active_owner->perf_plan,"planner args lack persistent owner");
        require(bytes==16&&blocks==1&&schedule==0,"planner launch profile");
        const uint64_t expected[2]={fixture->args[5],fixture->args[8]+SYSTEM};
        require(std::memcmp(host,expected,16)==0,"planner 16B payload differs");
        return 0;
    }
    calls.push_back(function==active_owner->functions.native?"native":"candidate");
    require(function==active_owner->functions.native||function==active_owner->functions.candidate,"function owner mismatch");
    require(host==&active_owner->perf_args,"GMM args lack persistent owner");
    require(bytes==72&&blocks==24&&schedule==1,"GMM launch profile");
    // Reject malformed field 8 from independent v5 data, including output GM.
    if(std::memcmp(host,fixture->args.data(),72)!=0){
        const auto *words=static_cast<const uint64_t *>(host);
        if(words[7]==fixture->gm[6])++rejected_output_gm;
        return GMM2_LAUNCH_V3_INVALID_ARGUMENT;
    }
    for(unsigned i=0;i<5;++i){
        auto it=descriptors.find(fixture->args[i==4?7:i+1]);
        require(it!=descriptors.end(),"submitted GM descriptor missing");
        require(std::memcmp(it->second.data(),fixture->lists.data()+4*i,32)==0,"descriptor bytes differ from v5 runtime receipt");
    }
    return 0;
}

int main(){
    try{
        for(const auto &record:V5_RECORDED_LAUNCHES){
            fixture=&record;Probe p{};active_owner=&p;descriptors.clear();calls.clear();
            p.borrowed_context=reinterpret_cast<void *>(0x7100);
            p.borrowed_stream=reinterpret_cast<void *>(0x7200);
            p.functions.planner=reinterpret_cast<void *>(0x7300);
            p.functions.native=reinterpret_cast<void *>(0x7400);
            p.functions.candidate=reinterpret_cast<void *>(0x7500);
            p.ffts=record.args[0];p.workspace_base=record.args[8];
            p.workspace_allocation=reinterpret_cast<void *>(record.args[8]-512);
            p.y_allocation=reinterpret_cast<void *>(record.gm[6]-128);
            for(unsigned i=0;i<5;++i)p.lists[i]=reinterpret_cast<void *>(record.args[i==4?7:i+1]);
            require(p.workspace_base%512==0,"v5 workspace alignment");
            require(record.args[7]!=record.gm[6],"v5 output descriptor and output tensor unexpectedly alias");
            require(m1_perf_bind(&p,record.gm[0],record.gm[1],record.gm[2],record.gm[3],record.gm[4],record.gm[5])==0,"bind failed");
            compare_payload(p.perf_args);
            for(unsigned i=0;i<5;++i)require(std::memcmp(descriptors[addr(p.lists[i])].data(),record.lists.data()+4*i,32)==0,"compiled pack mismatch");
            auto before=descriptors;auto *stable_args=&p.perf_args;calls.clear();
            require(m1_perf_submit(&p,record.sparse)==0,"submit failed");
            require(calls==(record.sparse?std::vector<std::string>{"planner","candidate"}:std::vector<std::string>{"native"}),"submit includes allocation/copy/clear/sync or wrong ordering");
            require(stable_args==&p.perf_args,"args address changed before stop");
            auto submit_calls=calls;
            require(m1_perf_bind(&p,record.gm[0],record.gm[1],record.gm[2],record.gm[3],record.gm[4],record.gm[5])==-910901,"in-flight bind was allowed");
            require(calls==submit_calls&&descriptors==before,"in-flight rejected bind changed descriptors");
            std::array<uint16_t,6*5120> output{};std::array<uint32_t,32> frame{};
            require(m1_perf_finish(&p,output.data(),frame.data())==0&&p.synchronized,"finish failed");
            require(calls[submit_calls.size()]=="sync","finish did not synchronize before readback");
            require(descriptors==before,"finish overwrote a descriptor");
            // Exactly reproduce the v1 bug in the actual owner payload.
            p.perf_args.y_list=record.gm[6];calls.clear();
            const auto accepted_before=p.perf_native_accepted;
            require(m1_perf_submit(&p,0)==GMM2_LAUNCH_V3_INVALID_ARGUMENT,"output GM masquerading as descriptor was accepted");
            require(p.perf_native_accepted==accepted_before,"rejected API counted as accepted");
            p.synchronized=true; // CPU model termination; never a device operation.
        }
        require(rejected_output_gm==27,"negative examples not exercised");
        std::cout<<"{\"CPU_only\":true,\"NPU_calls\":0,\"SDK_runtime_linked\":false,\"v5_actual_payloads_checked\":27,\"output_GM_rejected_examples\":"<<rejected_output_gm<<",\"checks\":"<<checks<<",\"passed\":true}"<<std::endl;
        return 0;
    }catch(const std::exception &e){std::cerr<<"CPU ABI gate failure: "<<e.what()<<std::endl;return 1;}
}
