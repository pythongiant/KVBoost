/*
Streaming Inference CUDA Glue Code

Minimal CUDA utilities:
- cudaStreamWaitEvent helpers for event synchronization
- Optional fused RMSNorm kernel for streaming layers
*/

#include <cuda_runtime.h>

/*
Helper to wait for event on a specific stream.
*/
cudaError_t stream_wait_event(cudaStream_t stream, cudaEvent_t event) {
    return cudaStreamWaitEvent(stream, event, 0);
}

/*
Helper to record event on a specific stream.
*/
cudaError_t stream_record_event(cudaEvent_t event, cudaStream_t stream) {
    return cudaEventRecord(event, stream);
}

/*
Optional fused RMSNorm kernel placeholder.
Could optimize by fusing RMSNorm + residual addition for streaming layers.
*/
__global__ void fused_rms_norm_kernel(float *x, float *out, float *weight, 
                                      int hidden_size, float epsilon) {
    // Placeholder for fused RMSNorm + residual
}
