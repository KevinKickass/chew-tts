template <typename T>
__device__ inline void kv_write_rows_impl(
        const T * src,
              T * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    const long long linear = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long) index_count * src_row_width;

    if (linear >= total) {
        return;
    }

    const int row = (int) (linear / src_row_width);
    const int col = (int) (linear % src_row_width);

    if (row >= src_row_count) {
        return;
    }

    const long long dst_row = idxs[row];
    char * dst_bytes = (char *) dst + dst_row * dst_row_stride_bytes;
    T * dst_row_ptr = (T *) dst_bytes;

    if (col < dst_row_width) {
        dst_row_ptr[col] = src[(long long) row * src_row_width + col];
    }
}

extern "C" __global__ void kv_write_k_rows_f16(
        const unsigned short * src,
              unsigned short * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<unsigned short>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_k_rows_bf16(
        const unsigned short * src,
              unsigned short * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<unsigned short>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_k_rows_f32(
        const float * src,
              float * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<float>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_v_rows_f16(
        const unsigned short * src,
              unsigned short * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<unsigned short>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_v_rows_bf16(
        const unsigned short * src,
              unsigned short * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<unsigned short>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_v_rows_f32(
        const float * src,
              float * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<float>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_v_rows_trans_f16(
        const unsigned short * src,
              unsigned short * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<unsigned short>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_v_rows_trans_bf16(
        const unsigned short * src,
              unsigned short * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<unsigned short>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}

extern "C" __global__ void kv_write_v_rows_trans_f32(
        const float * src,
              float * dst,
        const long long * idxs,
        int src_row_width,
        int dst_row_width,
        long long dst_row_stride_bytes,
        int src_row_count,
        int index_count) {
    kv_write_rows_impl<float>(
        src,
        dst,
        idxs,
        src_row_width,
        dst_row_width,
        dst_row_stride_bytes,
        src_row_count,
        index_count);
}
