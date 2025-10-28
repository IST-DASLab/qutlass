#pragma once
#include <common.h>

int backward_qt_bf16_cuda(
    const void*,
    const void*,
    const void*,
    const void*,
    void*,
    void*,
    const int,
    const int,
    const int,
    cudaStream_t
);

int backward_t_bf16_cuda(
    const void*,
    const void*,
    void*,
    void*,
    const int,
    const int,
    const int,
    cudaStream_t
);

int backward_bf16_square_double_mxfp8_cuda(
    const void*,
    const int,
    const int,
    void*,
    void*,
    void*,
    cudaStream_t
);

int mxfp4_transpose_mxfp8_cuda(
    const void*,
    const void*,
    const int,
    const int,
    void*,
    void*,
    cudaStream_t
);