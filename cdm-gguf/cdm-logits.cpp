#include "llama.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
int main(int argc,char **argv) {
    if(argc<3 || (argc<4 && !getenv("CDM_PROMPT"))){fprintf(stderr,"usage: cdm-logits model.gguf logits.f32 token_id... [CDM_THREADS=4, CDM_INCREMENTAL=1]\n");return 2;}
    ggml_backend_load_all(); llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=getenv("CDM_GPU_LAYERS")?atoi(getenv("CDM_GPU_LAYERS")):0;
    if(mp.n_gpu_layers>0) {
        bool gpu=false;for(size_t i=0;i<ggml_backend_dev_count();++i) gpu|=ggml_backend_dev_type(ggml_backend_dev_get(i))==GGML_BACKEND_DEVICE_TYPE_GPU;
        if(!gpu){fprintf(stderr,"CDM: GPU offload requested but no GPU backend is available\n");return 2;}
    }
    auto *m=llama_model_load_from_file(argv[1],mp); if(!m)return 1;
    std::vector<llama_token> ids;for(int i=3;i<argc;i++)ids.push_back(std::stoi(argv[i]));
    if(const char *prompt=getenv("CDM_PROMPT")) {
        auto *vocab=llama_model_get_vocab(m);
        int n=-llama_tokenize(vocab,prompt,std::string(prompt).size(),nullptr,0,true,true);
        ids.resize(n);n=llama_tokenize(vocab,prompt,std::string(prompt).size(),ids.data(),ids.size(),true,true);
        if(n<0)return 2;ids.resize(n);
        fprintf(stderr,"CDM_IDS:");for(auto id:ids)fprintf(stderr," %d",id);fprintf(stderr,"\n");
    }
    int nv=llama_vocab_n_tokens(llama_model_get_vocab(m));
    if(ids.empty() || ids.size() > llama_model_n_ctx_train(m)){fprintf(stderr,"input empty or exceeds trained context\n");return 2;}
    for(auto id:ids) if(id<0 || id>=nv){fprintf(stderr,"invalid token ID\n");return 2;}
    auto cp=llama_context_default_params();cp.n_ctx=llama_model_n_ctx_train(m);cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.n_batch=cp.n_ctx;cp.n_ubatch=cp.n_ctx;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads=cp.n_threads_batch=getenv("CDM_THREADS")?atoi(getenv("CDM_THREADS")):4;
    auto *c=llama_init_from_model(m,cp);if(!c)return 1;
    auto batch=llama_batch_init(ids.size(),0,1);
    size_t vocab=llama_vocab_n_tokens(llama_model_get_vocab(m));std::vector<float> logits(ids.size()*vocab);
    bool incremental=getenv("CDM_INCREMENTAL") && std::string(getenv("CDM_INCREMENTAL"))=="1";
    int repeats=getenv("CDM_REPEATS")?atoi(getenv("CDM_REPEATS")):1;
    if(repeats<1 || repeats>10000)return 2;
    std::vector<double> timings, decode_times, prefill_times;
    int prefill=getenv("CDM_PREFILL_COUNT")?atoi(getenv("CDM_PREFILL_COUNT")):1;
    if(prefill<1 || prefill>int(ids.size()))return 2;
    for(int run=(repeats>1?-1:0);run<repeats;++run) {
        llama_memory_clear(llama_get_memory(c),true);
        llama_synchronize(c);
        auto start=std::chrono::steady_clock::now();
        double prefill_seconds=0;
        auto decode_start=start;
        for(size_t offset=0;offset<ids.size();) {
            int count=incremental?(offset==0?prefill:1):ids.size(); batch.n_tokens=count;
            for(int j=0;j<count;j++){batch.token[j]=ids[offset+j];batch.pos[j]=offset+j;batch.n_seq_id[j]=1;batch.seq_id[j][0]=0;batch.logits[j]=1;}
            if(llama_decode(c,batch)){fprintf(stderr,"decode failed\n");return 1;}
            for(int j=0;j<count;j++)std::copy_n(llama_get_logits_ith(c,j),vocab,logits.data()+(offset+j)*vocab);
            offset+=count;
            if(offset==size_t(count)) {decode_start=std::chrono::steady_clock::now();prefill_seconds=std::chrono::duration<double>(decode_start-start).count();}
        }
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(run>=0){timings.push_back(elapsed);prefill_times.push_back(prefill_seconds);decode_times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-decode_start).count());}
    }
    fprintf(stderr,"CDM_TIMES:");for(auto t:timings)fprintf(stderr," %.9f",t);fprintf(stderr,"\n");
    fprintf(stderr,"CDM_DECODE_TIMES:");for(auto t:decode_times)fprintf(stderr," %.9f",t);fprintf(stderr,"\n");
    fprintf(stderr,"CDM_PREFILL_TIMES:");for(auto t:prefill_times)fprintf(stderr," %.9f",t);fprintf(stderr,"\n");
    std::sort(timings.begin(),timings.end());
    double seconds=timings[timings.size()/2];
    auto *f=fopen(argv[2],"wb");if(!f)return 1;fwrite(logits.data(),sizeof(float),logits.size(),f);fclose(f);
    fprintf(stderr,"CDM: %zu tokens, %zu vocab, %.6f s, %.3f tokens/s (%s)\n",ids.size(),vocab,seconds,ids.size()/seconds,incremental?"incremental KV":"prefill");
    llama_batch_free(batch);llama_free(c);llama_model_free(m);llama_backend_free();
}
