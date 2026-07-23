use crate::fast_launch::{FastStream, scalar_ptr, slice_ptr, slice_ptr_mut};
use crate::loader::{self, KernelError};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use std::ffi::c_void;
use std::sync::Arc;

const SPEAKER_CU: &str = include_str!("cuda/speaker.cu");

pub struct SpeakerKernels {
    stream: Arc<CudaStream>,
    fast: FastStream,
    _module: Arc<CudaModule>,
    unfold_reflect: CudaFunction,
    unfold_causal_stride: CudaFunction,
    relu: CudaFunction,
    elu: CudaFunction,
    tanh: CudaFunction,
    sigmoid: CudaFunction,
    channel_mean: CudaFunction,
    channel_scale: CudaFunction,
    append_channel_block: CudaFunction,
    append_context: CudaFunction,
    channel_stats: CudaFunction,
    softmax_channels: CudaFunction,
    nearest_codebook: CudaFunction,
    subtract_codebook: CudaFunction,
}

impl SpeakerKernels {
    pub fn load(stream: &Arc<CudaStream>) -> Result<Self, KernelError> {
        let module = loader::load_module_from_source(stream, SPEAKER_CU, "speaker")?;
        Ok(Self {
            stream: Arc::clone(stream),
            fast: FastStream::new(stream),
            unfold_reflect: loader::get_fn(&module, "unfold_reflect_f16")?,
            unfold_causal_stride: loader::get_fn(&module, "unfold_causal_stride_f16")?,
            relu: loader::get_fn(&module, "relu_f16")?,
            elu: loader::get_fn(&module, "elu_f16")?,
            tanh: loader::get_fn(&module, "tanh_f16")?,
            sigmoid: loader::get_fn(&module, "sigmoid_f16")?,
            channel_mean: loader::get_fn(&module, "channel_mean_f16")?,
            channel_scale: loader::get_fn(&module, "channel_scale_f16")?,
            append_channel_block: loader::get_fn(&module, "append_channel_block_f16")?,
            append_context: loader::get_fn(&module, "append_context_f16")?,
            channel_stats: loader::get_fn(&module, "channel_stats_f16")?,
            softmax_channels: loader::get_fn(&module, "softmax_channels_f16")?,
            nearest_codebook: loader::get_fn(&module, "nearest_codebook_f16")?,
            subtract_codebook: loader::get_fn(&module, "subtract_codebook_f16")?,
            _module: module,
        })
    }

    fn elementwise(&self, function: &CudaFunction, values: &mut CudaSlice<half::f16>) {
        let n = values.len() as i32;
        let mut args = [slice_ptr_mut(values), scalar_ptr(&n)];
        unsafe {
            self.fast.fire(
                function,
                ((n as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn relu(&self, values: &mut CudaSlice<half::f16>) {
        self.elementwise(&self.relu, values)
    }

    pub fn elu(&self, values: &mut CudaSlice<half::f16>) {
        self.elementwise(&self.elu, values)
    }

    pub fn tanh(&self, values: &mut CudaSlice<half::f16>) {
        self.elementwise(&self.tanh, values)
    }

    pub fn sigmoid(&self, values: &mut CudaSlice<half::f16>) {
        self.elementwise(&self.sigmoid, values)
    }

    pub fn unfold_reflect(
        &self,
        input: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        channels: u32,
        seq_len: u32,
        kernel_size: u32,
        dilation: u32,
    ) {
        let channels = channels as i32;
        let seq_len = seq_len as i32;
        let kernel_size = kernel_size as i32;
        let dilation = dilation as i32;
        let total = channels as u32 * seq_len as u32 * kernel_size as u32;
        let mut args: [*mut c_void; 6] = [
            slice_ptr(input),
            slice_ptr_mut(output),
            scalar_ptr(&channels),
            scalar_ptr(&seq_len),
            scalar_ptr(&kernel_size),
            scalar_ptr(&dilation),
        ];
        unsafe {
            self.fast.fire(
                &self.unfold_reflect,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn unfold_causal_stride(
        &self,
        input: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        channels: u32,
        input_len: u32,
        output_len: u32,
        kernel_size: u32,
        stride: u32,
        dilation: u32,
    ) {
        let channels_i = channels as i32;
        let input_len_i = input_len as i32;
        let output_len_i = output_len as i32;
        let kernel_size_i = kernel_size as i32;
        let stride_i = stride as i32;
        let dilation_i = dilation as i32;
        let total = output_len * channels * kernel_size;
        let mut args: [*mut c_void; 8] = [
            slice_ptr(input),
            slice_ptr_mut(output),
            scalar_ptr(&channels_i),
            scalar_ptr(&input_len_i),
            scalar_ptr(&output_len_i),
            scalar_ptr(&kernel_size_i),
            scalar_ptr(&stride_i),
            scalar_ptr(&dilation_i),
        ];
        unsafe {
            self.fast.fire(
                &self.unfold_causal_stride,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn channel_mean(
        &self,
        input: &CudaSlice<half::f16>,
        mean: &mut CudaSlice<half::f16>,
        channels: u32,
        seq_len: u32,
    ) {
        self.channel_reduce(&self.channel_mean, input, mean, channels, seq_len)
    }

    fn channel_reduce(
        &self,
        function: &CudaFunction,
        input: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        channels: u32,
        seq_len: u32,
    ) {
        let channels_i = channels as i32;
        let seq_len_i = seq_len as i32;
        let mut args = [
            slice_ptr(input),
            slice_ptr_mut(output),
            scalar_ptr(&channels_i),
            scalar_ptr(&seq_len_i),
        ];
        unsafe {
            self.fast
                .fire(function, (channels, 1, 1), (256, 1, 1), 0, &mut args)
        }
    }

    pub fn channel_scale(
        &self,
        input: &CudaSlice<half::f16>,
        scale: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        channels: u32,
        seq_len: u32,
    ) {
        let channels_i = channels as i32;
        let seq_len_i = seq_len as i32;
        let total = channels * seq_len;
        let mut args = [
            slice_ptr(input),
            slice_ptr(scale),
            slice_ptr_mut(output),
            scalar_ptr(&channels_i),
            scalar_ptr(&seq_len_i),
        ];
        unsafe {
            self.fast.fire(
                &self.channel_scale,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_channel_block(
        &self,
        input: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        input_channels: u32,
        output_channels: u32,
        channel_offset: u32,
        seq_len: u32,
    ) {
        let input_channels_i = input_channels as i32;
        let output_channels_i = output_channels as i32;
        let channel_offset_i = channel_offset as i32;
        let seq_len_i = seq_len as i32;
        let total = input_channels * seq_len;
        let mut args = [
            slice_ptr(input),
            slice_ptr_mut(output),
            scalar_ptr(&input_channels_i),
            scalar_ptr(&output_channels_i),
            scalar_ptr(&channel_offset_i),
            scalar_ptr(&seq_len_i),
        ];
        unsafe {
            self.fast.fire(
                &self.append_channel_block,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn append_context(
        &self,
        input: &CudaSlice<half::f16>,
        mean: &CudaSlice<half::f16>,
        stddev: &CudaSlice<half::f16>,
        output: &mut CudaSlice<half::f16>,
        channels: u32,
        seq_len: u32,
    ) {
        let channels_i = channels as i32;
        let seq_len_i = seq_len as i32;
        let total = channels * seq_len;
        let mut args = [
            slice_ptr(input),
            slice_ptr(mean),
            slice_ptr(stddev),
            slice_ptr_mut(output),
            scalar_ptr(&channels_i),
            scalar_ptr(&seq_len_i),
        ];
        unsafe {
            self.fast.fire(
                &self.append_context,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn channel_stats(
        &self,
        input: &CudaSlice<half::f16>,
        weights: Option<&CudaSlice<half::f16>>,
        mean: &mut CudaSlice<half::f16>,
        stddev: &mut CudaSlice<half::f16>,
        channels: u32,
        seq_len: u32,
    ) {
        let channels_i = channels as i32;
        let seq_len_i = seq_len as i32;
        let weights_ptr = weights.map_or(std::ptr::null_mut(), slice_ptr);
        let mut args = [
            slice_ptr(input),
            weights_ptr,
            slice_ptr_mut(mean),
            slice_ptr_mut(stddev),
            scalar_ptr(&channels_i),
            scalar_ptr(&seq_len_i),
        ];
        unsafe {
            self.fast.fire(
                &self.channel_stats,
                (channels, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn softmax_channels(&self, values: &mut CudaSlice<half::f16>, channels: u32, seq_len: u32) {
        let channels_i = channels as i32;
        let seq_len_i = seq_len as i32;
        let mut args = [
            slice_ptr_mut(values),
            scalar_ptr(&channels_i),
            scalar_ptr(&seq_len_i),
        ];
        unsafe {
            self.fast.fire(
                &self.softmax_channels,
                (channels, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn nearest_codebook(
        &self,
        input: &CudaSlice<half::f16>,
        codebook: &CudaSlice<half::f16>,
        indices: &mut CudaSlice<i32>,
        frames: u32,
        codebook_size: u32,
        dim: u32,
    ) {
        let frames_i = frames as i32;
        let codebook_size_i = codebook_size as i32;
        let dim_i = dim as i32;
        let mut args = [
            slice_ptr(input),
            slice_ptr(codebook),
            slice_ptr_mut(indices),
            scalar_ptr(&frames_i),
            scalar_ptr(&codebook_size_i),
            scalar_ptr(&dim_i),
        ];
        unsafe {
            self.fast.fire(
                &self.nearest_codebook,
                (frames, 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn subtract_codebook(
        &self,
        residual: &mut CudaSlice<half::f16>,
        codebook: &CudaSlice<half::f16>,
        indices: &CudaSlice<i32>,
        frames: u32,
        dim: u32,
    ) {
        let frames_i = frames as i32;
        let dim_i = dim as i32;
        let total = frames * dim;
        let mut args = [
            slice_ptr_mut(residual),
            slice_ptr(codebook),
            slice_ptr(indices),
            scalar_ptr(&frames_i),
            scalar_ptr(&dim_i),
        ];
        unsafe {
            self.fast.fire(
                &self.subtract_codebook,
                (total.div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )
        }
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}
