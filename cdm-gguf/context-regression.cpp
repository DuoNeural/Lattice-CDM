#include "llama.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    ggml_backend_load_all();llama_backend_init();
    auto mp=llama_model_default_params();mp.n_gpu_layers=getenv("CDM_GPU_LAYERS")?atoi(getenv("CDM_GPU_LAYERS")):0;
    auto *model=llama_model_load_from_file(argv[1],mp);if(!model)return 1;
    unsigned limit=llama_model_n_ctx_train(model);
    auto cp=llama_context_default_params();cp.n_ctx=limit+1;cp.n_seq_max=1;cp.n_batch=1;cp.n_ubatch=1;cp.n_threads=cp.n_threads_batch=4;
    cp.type_k=cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto *bad=llama_init_from_model(model,cp);
    if(bad){fprintf(stderr,"FAIL: oversized context accepted\n");return 1;}
    cp.n_ctx=limit;auto *ctx=llama_init_from_model(model,cp);if(!ctx)return 1;
    if(llama_memory_can_shift(llama_get_memory(ctx))){fprintf(stderr,"FAIL: CDM cache permits invalid context shift\n");return 1;}
    auto batch=llama_batch_init(1,0,1);batch.n_tokens=1;batch.token[0]=2;batch.n_seq_id[0]=1;batch.seq_id[0][0]=0;batch.logits[0]=1;
    for(unsigned i=0;i<limit;++i){batch.pos[0]=i;if(llama_decode(ctx,batch)){fprintf(stderr,"FAIL: valid position rejected\n");return 1;}}
    batch.pos[0]=limit;
    if(llama_decode(ctx,batch)==0){fprintf(stderr,"FAIL: generation beyond learned table accepted\n");return 1;}
    batch.pos[0]=-1;
    if(llama_decode(ctx,batch)==0){fprintf(stderr,"FAIL: negative position accepted\n");return 1;}
    fprintf(stderr,"CDM_CONTEXT_CHECK: oversized context rejected; valid positions accepted; generation overflow and negative position rejected; cache shift disabled\n");
    llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
}
