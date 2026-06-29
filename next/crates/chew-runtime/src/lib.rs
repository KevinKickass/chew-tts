//! Runtime core shared by model backends.
//! Keeps context management, batching, KV ownership and scheduling out of model-specific code.

use std::error::Error;
use std::fmt::{Display, Formatter};

use chew_cuda::{
    execute_prepared_kv_write_bundle_f32, execute_prepared_kv_write_bundle_u16, CudaBackend,
    CudaError, Dim3, KernelLaunchConfig, KvWriteJobF32, KvWriteJobU16, KvWriteKernelKind,
    KvWriteKernelLaunch, KvWriteLaunchPlan, ModuleCache, NvrtcCompileOptions,
    PreparedKvWriteBundle, ScalarType as CudaScalarType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeLimits {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionSlot {
    pub id: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionMode {
    FullCausal,
    SlidingWindow { window: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSharing {
    Dedicated,
    Shared { group: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerKvSpec {
    pub layer_idx: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub attention: AttentionMode,
    pub sharing: KvSharing,
}

impl LayerKvSpec {
    pub fn n_embd_k_gqa(&self) -> u32 {
        self.kv_heads.saturating_mul(self.head_dim)
    }

    pub fn n_embd_v_gqa(&self) -> u32 {
        self.kv_heads.saturating_mul(self.head_dim)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheLayout {
    pub layers: Vec<LayerKvSpec>,
}

impl KvCacheLayout {
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn full_attention_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| matches!(layer.attention, AttentionMode::FullCausal))
            .count()
    }

    pub fn sliding_window_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| matches!(layer.attention, AttentionMode::SlidingWindow { .. }))
            .count()
    }

    pub fn shared_group_count(&self) -> usize {
        let mut groups = std::collections::BTreeSet::new();
        for layer in &self.layers {
            if let KvSharing::Shared { group } = layer.sharing {
                groups.insert(group);
            }
        }
        groups.len()
    }

    pub fn dedicated_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|layer| matches!(layer.sharing, KvSharing::Dedicated))
            .count()
    }

    pub fn layer(&self, layer_idx: u32) -> Option<&LayerKvSpec> {
        self.layers.iter().find(|layer| layer.layer_idx == layer_idx)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum KvStorageKey {
    Dedicated(u32),
    Shared(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvScalarType {
    F16,
    BF16,
    F32,
}

impl KvScalarType {
    pub fn bytes_per_scalar(self) -> u32 {
        match self {
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSwaType {
    None,
    Standard,
    Chunked,
    Symmetric,
}

impl KvSwaType {
    pub fn is_masked(self, n_swa: u32, p0: u32, p1: u32) -> bool {
        match self {
            Self::None => false,
            Self::Standard => p1.saturating_sub(p0) >= n_swa,
            Self::Chunked => {
                let pos_chunk_start = (p1 / n_swa) * n_swa;
                p0 < pos_chunk_start
            }
            Self::Symmetric => {
                let half_n_swa = (n_swa / 2) as i64;
                let pos_diff = p1 as i64 - p0 as i64;
                pos_diff < -half_n_swa || pos_diff > half_n_swa
            }
        }
    }
}

fn pad_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    value.div_ceil(align) * align
}

fn validate_slot_info_shape(ubatch: &KvUbatch, sinfo: &KvSlotInfo) -> Result<(), RuntimeError> {
    if !sinfo.is_rectangular() {
        return Err(RuntimeError::NonRectangularKvSlotInfo);
    }

    let slot_tokens = sinfo.size() * sinfo.n_stream();
    if ubatch.n_tokens() != slot_tokens {
        return Err(RuntimeError::KvIndexTokenMismatch {
            ubatch_tokens: ubatch.n_tokens(),
            slot_tokens,
        });
    }

    Ok(())
}

fn build_input_k_indices_for_slot(
    kv_size: u32,
    ubatch: &KvUbatch,
    sinfo: &KvSlotInfo,
) -> Result<Vec<i64>, RuntimeError> {
    validate_slot_info_shape(ubatch, sinfo)?;

    let per_stream = sinfo.size();
    let mut data = Vec::with_capacity(ubatch.n_tokens());

    for (stream_idx, stream) in sinfo.strm.iter().copied().enumerate() {
        let offs = (stream as i64) * (kv_size as i64);
        for i in 0..per_stream {
            data.push(offs + sinfo.idxs[stream_idx][i] as i64);
        }
    }

    Ok(data)
}

fn build_input_v_indices_for_slot(
    kv_size: u32,
    ubatch: &KvUbatch,
    sinfo: &KvSlotInfo,
    n_embd_v_gqa: u32,
    v_trans: bool,
) -> Result<Vec<i64>, RuntimeError> {
    validate_slot_info_shape(ubatch, sinfo)?;

    if !v_trans {
        return build_input_k_indices_for_slot(kv_size, ubatch, sinfo);
    }

    let per_stream = sinfo.size();
    let n_embd_v_gqa = n_embd_v_gqa as usize;
    let mut data = Vec::with_capacity(ubatch.n_tokens() * n_embd_v_gqa);

    for (stream_idx, stream) in sinfo.strm.iter().copied().enumerate() {
        let offs = (stream as i64) * (kv_size as i64) * (n_embd_v_gqa as i64);
        for i in 0..per_stream {
            let idx = sinfo.idxs[stream_idx][i] as i64;
            for j in 0..n_embd_v_gqa {
                data.push(offs + (j as i64) * (kv_size as i64) + idx);
            }
        }
    }

    Ok(data)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvStorageDescriptor {
    pub storage_idx: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub stride_el: u32,
    pub max_ctx: u32,
    pub n_stream: u32,
    pub scalar_type: KvScalarType,
    pub sharing: KvSharing,
    pub key_bytes_offset: u64,
    pub value_bytes_offset: u64,
    pub bytes_per_stream_cache: u64,
    pub bytes_per_tensor: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvTensorViewDescriptor {
    pub byte_offset: u64,
    pub scalar_type: KvScalarType,
    pub dims: [u32; 4],
    pub strides: [u64; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayerCacheView {
    pub layer_idx: u32,
    pub storage_idx: u32,
    pub n_kv: u32,
    pub stream_start: u32,
    pub stream_count: u32,
    pub v_trans: bool,
    pub key: KvTensorViewDescriptor,
    pub value: KvTensorViewDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCopyTargetDescriptor {
    pub byte_offset: u64,
    pub scalar_type: KvScalarType,
    pub row_width_el: u32,
    pub row_count: u32,
    pub row_stride_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayerCopyPlan {
    pub layer_idx: u32,
    pub storage_idx: u32,
    pub v_trans: bool,
    pub key: KvCopyTargetDescriptor,
    pub value: KvCopyTargetDescriptor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvInputTensorDescriptor {
    pub row_width_el: u32,
    pub row_count: u32,
    pub contiguous: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayerInputCopyPlan {
    pub key: KvInputTensorDescriptor,
    pub value: KvInputTensorDescriptor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayerExecutionPlan {
    pub layer_idx: u32,
    pub n_kv: u32,
    pub cache_view: KvLayerCacheView,
    pub copy_plan: KvLayerCopyPlan,
    pub k_indices: Vec<i64>,
    pub v_indices: Vec<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvBatchExecutionPlan {
    pub n_kv: u32,
    pub layers: Vec<KvLayerExecutionPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvLayerInputU16 {
    pub layer_idx: u32,
    pub key_src: Vec<u16>,
    pub value_src: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvLayerInputF32 {
    pub layer_idx: u32,
    pub key_src: Vec<f32>,
    pub value_src: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvAttentionMask {
    pub n_tokens: u32,
    pub n_kv: u32,
    pub values: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvLayerAttentionPlan {
    pub layer_idx: u32,
    pub attention: AttentionMode,
    pub mask: KvAttentionMask,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvBatchAttentionPlan {
    pub n_kv: u32,
    pub layers: Vec<KvLayerAttentionPlan>,
}

fn kv_scalar_type_to_cuda(scalar_type: KvScalarType) -> CudaScalarType {
    match scalar_type {
        KvScalarType::F16 => CudaScalarType::F16,
        KvScalarType::BF16 => CudaScalarType::BF16,
        KvScalarType::F32 => CudaScalarType::F32,
    }
}

fn default_kv_write_launch(index_count: u32) -> KernelLaunchConfig {
    KernelLaunchConfig {
        grid: Dim3 {
            x: index_count.max(1),
            y: 1,
            z: 1,
        },
        block: Dim3 { x: 128, y: 1, z: 1 },
        shared_mem_bytes: 0,
    }
}

fn expected_input_elements(row_width_el: u32, row_count: u32) -> Result<usize, RuntimeError> {
    let total = (row_width_el as u64).saturating_mul(row_count as u64);
    usize::try_from(total).map_err(|_| RuntimeError::KvInputElementCountOverflow { elements: total })
}

fn find_layer_input_u16<'a>(
    inputs: &'a [KvLayerInputU16],
    layer_idx: u32,
) -> Result<&'a KvLayerInputU16, RuntimeError> {
    inputs
        .iter()
        .find(|input| input.layer_idx == layer_idx)
        .ok_or(RuntimeError::MissingKvLayerInput { layer_idx })
}

fn find_layer_input_f32<'a>(
    inputs: &'a [KvLayerInputF32],
    layer_idx: u32,
) -> Result<&'a KvLayerInputF32, RuntimeError> {
    inputs
        .iter()
        .find(|input| input.layer_idx == layer_idx)
        .ok_or(RuntimeError::MissingKvLayerInput { layer_idx })
}

impl KvLayerExecutionPlan {
    pub fn input_copy_plan(
        &self,
        n_embd_k_gqa: u32,
        n_embd_v_gqa: u32,
    ) -> Result<KvLayerInputCopyPlan, RuntimeError> {
        let k_row_count = u32::try_from(self.k_indices.len())
            .map_err(|_| RuntimeError::KvIndexCountOverflow {
                count: self.k_indices.len(),
            })?;
        let v_row_count = u32::try_from(self.v_indices.len())
            .map_err(|_| RuntimeError::KvIndexCountOverflow {
                count: self.v_indices.len(),
            })?;

        if n_embd_k_gqa != self.copy_plan.key.row_width_el {
            return Err(RuntimeError::KvInputWidthMismatch {
                expected: self.copy_plan.key.row_width_el,
                actual: n_embd_k_gqa,
            });
        }

        if !self.copy_plan.v_trans && n_embd_v_gqa != self.copy_plan.value.row_width_el {
            return Err(RuntimeError::KvInputWidthMismatch {
                expected: self.copy_plan.value.row_width_el,
                actual: n_embd_v_gqa,
            });
        }

        Ok(KvLayerInputCopyPlan {
            key: KvInputTensorDescriptor {
                row_width_el: n_embd_k_gqa,
                row_count: k_row_count,
                contiguous: true,
            },
            value: KvInputTensorDescriptor {
                row_width_el: if self.copy_plan.v_trans { 1 } else { n_embd_v_gqa },
                row_count: v_row_count,
                contiguous: true,
            },
        })
    }

    pub fn cuda_kv_write_launch_plan(
        &self,
        n_embd_k_gqa: u32,
        n_embd_v_gqa: u32,
    ) -> Result<KvWriteLaunchPlan, RuntimeError> {
        let input = self.input_copy_plan(n_embd_k_gqa, n_embd_v_gqa)?;

        Ok(KvWriteLaunchPlan {
            key: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: KvWriteKernelKind::KeyRows,
                scalar_type: kv_scalar_type_to_cuda(self.copy_plan.key.scalar_type),
                dst_byte_offset: self.copy_plan.key.byte_offset,
                src_row_width_el: input.key.row_width_el,
                row_width_el: self.copy_plan.key.row_width_el,
                row_stride_bytes: self.copy_plan.key.row_stride_bytes,
                dst_row_count: self.copy_plan.key.row_count,
                src_row_count: input.key.row_count,
                index_count: input.key.row_count,
                launch: default_kv_write_launch(input.key.row_count),
            },
            value: KvWriteKernelLaunch {
                module_name: "kv_cache",
                kind: if self.copy_plan.v_trans {
                    KvWriteKernelKind::ValueRowsTransposed
                } else {
                    KvWriteKernelKind::ValueRows
                },
                scalar_type: kv_scalar_type_to_cuda(self.copy_plan.value.scalar_type),
                dst_byte_offset: self.copy_plan.value.byte_offset,
                src_row_width_el: input.value.row_width_el,
                row_width_el: self.copy_plan.value.row_width_el,
                row_stride_bytes: self.copy_plan.value.row_stride_bytes,
                dst_row_count: self.copy_plan.value.row_count,
                src_row_count: input.value.row_count,
                index_count: input.value.row_count,
                launch: default_kv_write_launch(input.value.row_count),
            },
        })
    }
}

impl KvBatchExecutionPlan {
    pub fn cuda_kv_write_launch_plans(
        &self,
        layout: &KvCacheLayout,
    ) -> Result<Vec<KvWriteLaunchPlan>, RuntimeError> {
        self.layers
            .iter()
            .map(|layer| {
                let spec = layout
                    .layer(layer.layer_idx)
                    .ok_or(RuntimeError::MissingKvLayoutLayer {
                        layer_idx: layer.layer_idx,
                    })?;
                layer.cuda_kv_write_launch_plan(spec.n_embd_k_gqa(), spec.n_embd_v_gqa())
            })
            .collect()
    }

    pub fn jobs_u16(
        &self,
        layout: &KvCacheLayout,
        inputs: &[KvLayerInputU16],
    ) -> Result<Vec<KvWriteJobU16>, RuntimeError> {
        let mut jobs = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let spec = layout
                .layer(layer.layer_idx)
                .ok_or(RuntimeError::MissingKvLayoutLayer {
                    layer_idx: layer.layer_idx,
                })?;
            let input_plan = layer.input_copy_plan(spec.n_embd_k_gqa(), spec.n_embd_v_gqa())?;
            let input = find_layer_input_u16(inputs, layer.layer_idx)?;
            let expected_key = expected_input_elements(
                input_plan.key.row_width_el,
                input_plan.key.row_count,
            )?;
            let expected_value = expected_input_elements(
                input_plan.value.row_width_el,
                input_plan.value.row_count,
            )?;

            if input.key_src.len() != expected_key {
                return Err(RuntimeError::KvInputBufferLengthMismatch {
                    layer_idx: layer.layer_idx,
                    tensor: "key",
                    expected: expected_key,
                    actual: input.key_src.len(),
                });
            }
            if input.value_src.len() != expected_value {
                return Err(RuntimeError::KvInputBufferLengthMismatch {
                    layer_idx: layer.layer_idx,
                    tensor: "value",
                    expected: expected_value,
                    actual: input.value_src.len(),
                });
            }

            jobs.push(KvWriteJobU16 {
                key_src: input.key_src.clone(),
                key_indices: layer.k_indices.clone(),
                value_src: input.value_src.clone(),
                value_indices: layer.v_indices.clone(),
            });
        }

        Ok(jobs)
    }

    pub fn jobs_f32(
        &self,
        layout: &KvCacheLayout,
        inputs: &[KvLayerInputF32],
    ) -> Result<Vec<KvWriteJobF32>, RuntimeError> {
        let mut jobs = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let spec = layout
                .layer(layer.layer_idx)
                .ok_or(RuntimeError::MissingKvLayoutLayer {
                    layer_idx: layer.layer_idx,
                })?;
            let input_plan = layer.input_copy_plan(spec.n_embd_k_gqa(), spec.n_embd_v_gqa())?;
            let input = find_layer_input_f32(inputs, layer.layer_idx)?;
            let expected_key = expected_input_elements(
                input_plan.key.row_width_el,
                input_plan.key.row_count,
            )?;
            let expected_value = expected_input_elements(
                input_plan.value.row_width_el,
                input_plan.value.row_count,
            )?;

            if input.key_src.len() != expected_key {
                return Err(RuntimeError::KvInputBufferLengthMismatch {
                    layer_idx: layer.layer_idx,
                    tensor: "key",
                    expected: expected_key,
                    actual: input.key_src.len(),
                });
            }
            if input.value_src.len() != expected_value {
                return Err(RuntimeError::KvInputBufferLengthMismatch {
                    layer_idx: layer.layer_idx,
                    tensor: "value",
                    expected: expected_value,
                    actual: input.value_src.len(),
                });
            }

            jobs.push(KvWriteJobF32 {
                key_src: input.key_src.clone(),
                key_indices: layer.k_indices.clone(),
                value_src: input.value_src.clone(),
                value_indices: layer.v_indices.clone(),
            });
        }

        Ok(jobs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvArenaLayout {
    pub scalar_type: KvScalarType,
    pub storages: Vec<KvStorageDescriptor>,
    pub layer_to_storage: Vec<u32>,
    pub total_bytes: u64,
}

impl KvArenaLayout {
    pub fn for_layout(
        layout: &KvCacheLayout,
        max_ctx: u32,
        n_stream: u32,
        scalar_type: KvScalarType,
    ) -> Result<Self, RuntimeError> {
        if n_stream == 0 {
            return Err(RuntimeError::InvalidKvArenaStreams { n_stream });
        }
        let mut storages = Vec::new();
        let mut layer_to_storage = Vec::with_capacity(layout.layers.len());
        let mut storage_by_key = std::collections::BTreeMap::new();
        let mut shared_specs: std::collections::BTreeMap<u32, (u32, u32, u32)> =
            std::collections::BTreeMap::new();
        let mut total_bytes = 0u64;

        for layer in &layout.layers {
            let key = match layer.sharing {
                KvSharing::Dedicated => KvStorageKey::Dedicated(layer.layer_idx),
                KvSharing::Shared { group } => KvStorageKey::Shared(group),
            };

            if let KvSharing::Shared { group } = layer.sharing {
                if let Some((first_layer, kv_heads, head_dim)) = shared_specs.get(&group).copied() {
                    if kv_heads != layer.kv_heads || head_dim != layer.head_dim {
                        return Err(RuntimeError::IncompatibleSharedKvSpec {
                            group,
                            first_layer,
                            conflicting_layer: layer.layer_idx,
                        });
                    }
                } else {
                    shared_specs.insert(group, (layer.layer_idx, layer.kv_heads, layer.head_dim));
                }
            }

            let storage_idx = if let Some(existing) = storage_by_key.get(&key).copied() {
                existing
            } else {
                let bytes_per_scalar = scalar_type.bytes_per_scalar() as u64;
                let stride_el = layer.kv_heads.saturating_mul(layer.head_dim);
                let bytes_per_stream_cache =
                    (max_ctx as u64) * (stride_el as u64) * bytes_per_scalar;
                let bytes_per_tensor = bytes_per_stream_cache * (n_stream as u64);
                let storage_idx = storages.len() as u32;
                let descriptor = KvStorageDescriptor {
                    storage_idx,
                    kv_heads: layer.kv_heads,
                    head_dim: layer.head_dim,
                    stride_el,
                    max_ctx,
                    n_stream,
                    scalar_type,
                    sharing: layer.sharing,
                    key_bytes_offset: total_bytes,
                    value_bytes_offset: total_bytes + bytes_per_tensor,
                    bytes_per_stream_cache,
                    bytes_per_tensor,
                    total_bytes: bytes_per_tensor * 2,
                };
                total_bytes += descriptor.total_bytes;
                storages.push(descriptor);
                storage_by_key.insert(key, storage_idx);
                storage_idx
            };

            layer_to_storage.push(storage_idx);
        }

        Ok(Self {
            scalar_type,
            storages,
            layer_to_storage,
            total_bytes,
        })
    }

    pub fn storage_for_layer(&self, layer_idx: u32) -> Option<&KvStorageDescriptor> {
        let storage_idx = *self.layer_to_storage.get(layer_idx as usize)?;
        self.storages.get(storage_idx as usize)
    }

    pub fn view_layer_kv_cache(
        &self,
        layer_idx: u32,
        sinfo: &KvSlotInfo,
        n_kv: u32,
        v_trans: bool,
    ) -> Result<KvLayerCacheView, RuntimeError> {
        let storage = self
            .storage_for_layer(layer_idx)
            .ok_or(RuntimeError::MissingKvStorageForLayer { layer_idx })?;

        if n_kv > storage.max_ctx {
            return Err(RuntimeError::KvViewExceedsContext {
                n_kv,
                max_ctx: storage.max_ctx,
            });
        }

        let stream_start = sinfo.s0;
        let stream_count = sinfo.s1.saturating_sub(sinfo.s0).saturating_add(1);
        if stream_count == 0 || sinfo.s1 >= storage.n_stream {
            return Err(RuntimeError::KvViewStreamRangeExceedsArena {
                stream_start,
                stream_end: sinfo.s1,
                n_stream: storage.n_stream,
            });
        }

        let bps = storage.scalar_type.bytes_per_scalar() as u64;
        let key_base = storage.key_bytes_offset + storage.bytes_per_stream_cache * (stream_start as u64);
        let value_base =
            storage.value_bytes_offset + storage.bytes_per_stream_cache * (stream_start as u64);

        let key = KvTensorViewDescriptor {
            byte_offset: key_base,
            scalar_type: storage.scalar_type,
            dims: [storage.head_dim, storage.kv_heads, n_kv, stream_count],
            strides: [
                bps,
                bps * (storage.head_dim as u64),
                bps * (storage.stride_el as u64),
                storage.bytes_per_stream_cache,
            ],
        };

        let value = if !v_trans {
            KvTensorViewDescriptor {
                byte_offset: value_base,
                scalar_type: storage.scalar_type,
                dims: [storage.head_dim, storage.kv_heads, n_kv, stream_count],
                strides: [
                    bps,
                    bps * (storage.head_dim as u64),
                    bps * (storage.stride_el as u64),
                    storage.bytes_per_stream_cache,
                ],
            }
        } else {
            KvTensorViewDescriptor {
                byte_offset: value_base,
                scalar_type: storage.scalar_type,
                dims: [n_kv, storage.kv_heads, storage.head_dim, stream_count],
                strides: [
                    bps,
                    bps * (storage.max_ctx as u64) * (storage.head_dim as u64),
                    bps * (storage.max_ctx as u64),
                    storage.bytes_per_stream_cache,
                ],
            }
        };

        Ok(KvLayerCacheView {
            layer_idx,
            storage_idx: storage.storage_idx,
            n_kv,
            stream_start,
            stream_count,
            v_trans,
            key,
            value,
        })
    }

    pub fn bind_session_kv_write(
        &self,
        plan: &SessionKvWritePlan,
    ) -> Result<BoundSessionKvWritePlan, RuntimeError> {
        let mut layers = Vec::with_capacity(plan.layers.len());

        for layer in &plan.layers {
            let storage = self
                .storage_for_layer(layer.layer_idx)
                .ok_or(RuntimeError::MissingKvStorageForLayer {
                    layer_idx: layer.layer_idx,
                })?;
            let bytes_per_scalar = storage.scalar_type.bytes_per_scalar() as u64;
            let write_el_offset = (layer.write_pos as u64) * (storage.stride_el as u64);
            let write_el_count = (layer.token_count as u64) * (storage.stride_el as u64);

            layers.push(BoundKvLayerWrite {
                layer_idx: layer.layer_idx,
                storage_idx: storage.storage_idx,
                key_byte_offset: storage.key_bytes_offset + write_el_offset * bytes_per_scalar,
                value_byte_offset: storage.value_bytes_offset + write_el_offset * bytes_per_scalar,
                byte_len: write_el_count * bytes_per_scalar,
                attention_span: layer.attention_span,
                sharing: layer.sharing,
            });
        }

        Ok(BoundSessionKvWritePlan {
            slot: plan.slot,
            start_pos: plan.start_pos,
            token_count: plan.token_count,
            layers,
        })
    }

    pub fn copy_plan_for_layer(
        &self,
        layer_idx: u32,
        v_trans: bool,
    ) -> Result<KvLayerCopyPlan, RuntimeError> {
        let storage = self
            .storage_for_layer(layer_idx)
            .ok_or(RuntimeError::MissingKvStorageForLayer { layer_idx })?;
        let bps = storage.scalar_type.bytes_per_scalar() as u64;
        let merged_rows = storage.max_ctx.saturating_mul(storage.n_stream);

        let key = KvCopyTargetDescriptor {
            byte_offset: storage.key_bytes_offset,
            scalar_type: storage.scalar_type,
            row_width_el: storage.stride_el,
            row_count: merged_rows,
            row_stride_bytes: bps * (storage.stride_el as u64),
        };

        let value = if !v_trans {
            KvCopyTargetDescriptor {
                byte_offset: storage.value_bytes_offset,
                scalar_type: storage.scalar_type,
                row_width_el: storage.stride_el,
                row_count: merged_rows,
                row_stride_bytes: bps * (storage.stride_el as u64),
            }
        } else {
            KvCopyTargetDescriptor {
                byte_offset: storage.value_bytes_offset,
                scalar_type: storage.scalar_type,
                row_width_el: 1,
                row_count: merged_rows.saturating_mul(storage.stride_el),
                row_stride_bytes: bps,
            }
        };

        Ok(KvLayerCopyPlan {
            layer_idx,
            storage_idx: storage.storage_idx,
            v_trans,
            key,
            value,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvToken {
    pub pos: u32,
    pub seq_ids: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvUbatch {
    pub tokens: Vec<KvToken>,
}

impl KvUbatch {
    pub fn n_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn single_seq(seq_id: u32, positions: &[u32]) -> Self {
        Self {
            tokens: positions
                .iter()
                .copied()
                .map(|pos| KvToken {
                    pos,
                    seq_ids: vec![seq_id],
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvSlotInfo {
    pub s0: u32,
    pub s1: u32,
    pub strm: Vec<u32>,
    pub idxs: Vec<Vec<u32>>,
}

impl KvSlotInfo {
    pub fn size(&self) -> usize {
        self.idxs.first().map_or(0, Vec::len)
    }

    pub fn is_rectangular(&self) -> bool {
        let size = self.size();
        self.idxs.iter().all(|idxs| idxs.len() == size)
    }

    pub fn head(&self) -> Option<u32> {
        self.idxs.first().and_then(|idxs| idxs.first()).copied()
    }

    pub fn n_stream(&self) -> usize {
        self.strm.len()
    }

    pub fn is_contiguous(&self) -> bool {
        if self.idxs.is_empty() || self.idxs[0].is_empty() {
            return true;
        }
        if self.idxs.len() > 1 {
            return false;
        }
        let head = self.idxs[0][0];
        self.idxs[0]
            .iter()
            .enumerate()
            .all(|(i, &idx)| idx == head + i as u32)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KvCell {
    pos: Option<u32>,
    seq_ids: Vec<u32>,
}

impl KvCell {
    fn empty() -> Self {
        Self {
            pos: None,
            seq_ids: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.seq_ids.is_empty()
    }

    fn seq_count(&self) -> usize {
        self.seq_ids.len()
    }

    fn seq_get(&self) -> Option<u32> {
        self.seq_ids.first().copied()
    }

    fn seq_has(&self, seq_id: u32) -> bool {
        self.seq_ids.contains(&seq_id)
    }

    fn seq_rm(&mut self, seq_id: u32) -> bool {
        self.seq_ids.retain(|&id| id != seq_id);
        if self.seq_ids.is_empty() {
            self.pos = None;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KvCellStream {
    head: u32,
    cells: Vec<KvCell>,
}

impl KvCellStream {
    fn size(&self) -> u32 {
        self.cells.len() as u32
    }

    fn get_used(&self) -> u32 {
        self.cells.iter().filter(|cell| !cell.is_empty()).count() as u32
    }

    fn seq_pos_min(&self, seq_id: u32) -> Option<u32> {
        self.cells
            .iter()
            .filter(|cell| cell.seq_has(seq_id))
            .filter_map(|cell| cell.pos)
            .min()
    }

    fn seq_pos_max(&self, seq_id: u32) -> Option<u32> {
        self.cells
            .iter()
            .filter(|cell| cell.seq_has(seq_id))
            .filter_map(|cell| cell.pos)
            .max()
    }

    fn used_max_p1(&self) -> u32 {
        self.cells
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, cell)| (!cell.is_empty()).then_some(idx as u32 + 1))
            .unwrap_or(0)
    }

    fn seq_rm(&mut self, seq_id: u32, p0: u32, p1: u32) {
        for cell in &mut self.cells {
            if !cell.seq_has(seq_id) {
                continue;
            }
            if let Some(pos) = cell.pos {
                if pos >= p0 && pos < p1 {
                    cell.seq_rm(seq_id);
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheState {
    kv_size: u32,
    n_seq_max: u32,
    n_stream: u32,
    n_pad: u32,
    n_swa: u32,
    swa_type: KvSwaType,
    seq_to_stream: Vec<u32>,
    streams: Vec<KvCellStream>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheBatchContext {
    kv_size: u32,
    sinfos: Vec<KvSlotInfo>,
    ubatches: Vec<KvUbatch>,
    i_cur: usize,
    n_kv: u32,
}

impl KvCacheState {
    pub fn new(kv_size: u32, n_seq_max: u32, unified: bool) -> Self {
        Self::new_with_swa(kv_size, n_seq_max, unified, 0, KvSwaType::None)
    }

    pub fn new_with_swa(
        kv_size: u32,
        n_seq_max: u32,
        unified: bool,
        n_swa: u32,
        swa_type: KvSwaType,
    ) -> Self {
        Self::new_with_policy(kv_size, n_seq_max, unified, 1, n_swa, swa_type)
    }

    pub fn new_with_policy(
        kv_size: u32,
        n_seq_max: u32,
        unified: bool,
        n_pad: u32,
        n_swa: u32,
        swa_type: KvSwaType,
    ) -> Self {
        assert!(n_pad > 0, "n_pad must be positive");
        assert!(
            kv_size.is_multiple_of(n_pad),
            "kv_size must be divisible by n_pad"
        );
        let n_stream = if unified { 1 } else { n_seq_max.max(1) };
        let mut seq_to_stream = vec![0; n_seq_max as usize];
        if n_stream > 1 {
            for s in 0..n_stream as usize {
                seq_to_stream[s] = s as u32;
            }
        }

        Self {
            kv_size,
            n_seq_max,
            n_stream,
            n_pad,
            n_swa,
            swa_type,
            seq_to_stream,
            streams: (0..n_stream)
                .map(|_| KvCellStream {
                    head: 0,
                    cells: (0..kv_size).map(|_| KvCell::empty()).collect(),
                })
                .collect(),
        }
    }

    pub fn n_stream(&self) -> u32 {
        self.n_stream
    }

    pub fn prepare(&self, ubatches: &[KvUbatch], cont: bool) -> Option<Vec<KvSlotInfo>> {
        let mut scratch = self.clone();
        let mut slots = Vec::with_capacity(ubatches.len());
        for ubatch in ubatches {
            let sinfo = scratch.find_slot(ubatch, cont)?;
            scratch.apply_ubatch(&sinfo, ubatch);
            slots.push(sinfo);
        }
        Some(slots)
    }

    pub fn prepare_context(
        &self,
        ubatches: Vec<KvUbatch>,
        cont: bool,
    ) -> Result<KvCacheBatchContext, RuntimeError> {
        let sinfos = self
            .prepare(&ubatches, cont)
            .ok_or(RuntimeError::KvSlotUnavailable)?;
        for (ubatch, sinfo) in ubatches.iter().zip(&sinfos) {
            validate_slot_info_shape(ubatch, sinfo)?;
        }
        Ok(KvCacheBatchContext {
            kv_size: self.kv_size,
            sinfos,
            ubatches,
            i_cur: 0,
            n_kv: 0,
        })
    }

    pub fn find_slot(&self, ubatch: &KvUbatch, cont: bool) -> Option<KvSlotInfo> {
        if ubatch.tokens.is_empty() {
            return Some(KvSlotInfo {
                s0: 0,
                s1: 0,
                strm: vec![0],
                idxs: vec![Vec::new()],
            });
        }

        let mut stream_to_tokens: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (token_idx, token) in ubatch.tokens.iter().enumerate() {
            let seq_id = *token.seq_ids.first()?;
            let stream = *self.seq_to_stream.get(seq_id as usize)?;
            stream_to_tokens.entry(stream).or_default().push(token_idx);
        }

        let s0 = *stream_to_tokens.keys().next()?;
        let s1 = *stream_to_tokens.keys().last()?;

        let mut strm = Vec::new();
        let mut idxs = Vec::new();
        for (stream, token_idxs) in stream_to_tokens {
            let stream_state = self.streams.get(stream as usize)?;
            let found = self.find_slot_in_stream(stream_state, token_idxs.len() as u32, cont)?;
            strm.push(stream);
            idxs.push(found);
        }

        Some(KvSlotInfo { s0, s1, strm, idxs })
    }

    pub fn apply_ubatch(&mut self, sinfo: &KvSlotInfo, ubatch: &KvUbatch) {
        let mut seq_pos_max_rm = vec![None; self.n_seq_max as usize];
        let mut stream_to_tokens: std::collections::BTreeMap<u32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (token_idx, token) in ubatch.tokens.iter().enumerate() {
            let seq_id = token.seq_ids[0];
            let stream = self.seq_to_stream[seq_id as usize];
            stream_to_tokens.entry(stream).or_default().push(token_idx);
        }

        for (group_idx, stream) in sinfo.strm.iter().copied().enumerate() {
            let token_idxs = &stream_to_tokens[&stream];
            let cell_idxs = &sinfo.idxs[group_idx];
            let stream_state = &mut self.streams[stream as usize];

            for (local_idx, cell_idx) in cell_idxs.iter().copied().enumerate() {
                let token = &ubatch.tokens[token_idxs[local_idx]];
                let cell = &mut stream_state.cells[cell_idx as usize];

                if !cell.is_empty() && cell.seq_count() == 1 {
                    if let (Some(seq_id), Some(pos)) = (cell.seq_get(), cell.pos) {
                        let entry = &mut seq_pos_max_rm[seq_id as usize];
                        *entry = Some(entry.map_or(pos, |cur: u32| cur.max(pos)));
                    }
                }

                cell.pos = Some(token.pos);
                cell.seq_ids = token.seq_ids.clone();
            }

            if let Some(last_idx) = cell_idxs.last().copied() {
                stream_state.head = (last_idx + 1) % self.kv_size.max(1);
            }
        }

        for (seq_id, pos_max_rm) in seq_pos_max_rm.into_iter().enumerate() {
            let Some(pos_max_rm) = pos_max_rm else {
                continue;
            };
            let stream = self.seq_to_stream[seq_id];
            let stream_state = &mut self.streams[stream as usize];
            if let Some(pos_min) = stream_state.seq_pos_min(seq_id as u32) {
                if pos_min <= pos_max_rm {
                    stream_state.seq_rm(seq_id as u32, pos_min, pos_max_rm + 1);
                }
            }
        }
    }

    pub fn stream_head(&self, stream: u32) -> Option<u32> {
        self.streams.get(stream as usize).map(|state| state.head)
    }

    pub fn get_n_kv(&self, sinfo: &KvSlotInfo) -> u32 {
        let mut result = 0;
        let n_pad_cur = self.n_pad.max(256);

        for &stream in &sinfo.strm {
            let cells = &self.streams[stream as usize];
            let used_max_p1 = cells.used_max_p1();
            let padded = pad_up(used_max_p1, n_pad_cur).max(n_pad_cur);
            result = result.max(padded.min(cells.size()));
        }

        result
    }

    pub fn build_input_k_indices(
        &self,
        ubatch: &KvUbatch,
        sinfo: &KvSlotInfo,
    ) -> Result<Vec<i64>, RuntimeError> {
        build_input_k_indices_for_slot(self.kv_size, ubatch, sinfo)
    }

    pub fn build_input_v_indices(
        &self,
        ubatch: &KvUbatch,
        sinfo: &KvSlotInfo,
        n_embd_v_gqa: u32,
        v_trans: bool,
    ) -> Result<Vec<i64>, RuntimeError> {
        build_input_v_indices_for_slot(self.kv_size, ubatch, sinfo, n_embd_v_gqa, v_trans)
    }

    fn find_slot_in_stream(
        &self,
        stream: &KvCellStream,
        n_tokens: u32,
        cont: bool,
    ) -> Option<Vec<u32>> {
        if n_tokens == 0 {
            return Some(Vec::new());
        }
        if n_tokens > stream.size() {
            return None;
        }

        let mut head_cur = stream.head;
        if head_cur > stream.get_used().saturating_add(2 * n_tokens) {
            head_cur = 0;
        }

        let mut res = Vec::with_capacity(n_tokens as usize);
        let mut n_tested = 0u32;
        let n_test = if cont { n_tokens } else { 1 };

        loop {
            if head_cur + n_test > stream.size() {
                n_tested += stream.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for _ in 0..n_test {
                let idx = head_cur;
                head_cur += 1;
                n_tested += 1;

                let cell = &stream.cells[idx as usize];
                let mut can_use = cell.is_empty();

                if !can_use && cell.seq_count() == 1 {
                    let pos_cell = cell.pos.expect("occupied KV cell must have a position");
                    let seq_id_cell = cell.seq_get().expect("single-seq KV cell must expose seq id");
                    if let Some(pos_max) = stream.seq_pos_max(seq_id_cell) {
                        can_use = self
                            .swa_type
                            .is_masked(self.n_swa, pos_cell, pos_max.saturating_add(1));
                    }
                }

                if can_use {
                    res.push(idx);
                } else if cont {
                    break;
                }
            }

            if res.len() == n_tokens as usize {
                break;
            }

            if cont {
                res.clear();
            }

            if n_tested >= stream.size() {
                return None;
            }
        }

        if res.len() == n_tokens as usize {
            Some(res)
        } else {
            None
        }
    }
}

impl KvCacheBatchContext {
    pub fn len(&self) -> usize {
        self.ubatches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ubatches.is_empty()
    }

    pub fn current_ubatch(&self) -> Option<&KvUbatch> {
        self.ubatches.get(self.i_cur)
    }

    pub fn current_slot(&self) -> Option<&KvSlotInfo> {
        self.sinfos.get(self.i_cur)
    }

    pub fn n_kv(&self) -> u32 {
        self.n_kv
    }

    pub fn next(&mut self) -> bool {
        if self.i_cur + 1 >= self.ubatches.len() {
            return false;
        }
        self.i_cur += 1;
        true
    }

    pub fn apply(&mut self, kv: &mut KvCacheState) -> Result<(), RuntimeError> {
        let sinfo = self.current_slot().ok_or(RuntimeError::EmptyKvBatchContext)?;
        let ubatch = self
            .current_ubatch()
            .ok_or(RuntimeError::EmptyKvBatchContext)?;

        kv.apply_ubatch(sinfo, ubatch);
        self.n_kv = kv.get_n_kv(sinfo);

        Ok(())
    }

    pub fn build_input_k_indices(&self) -> Result<Vec<i64>, RuntimeError> {
        let sinfo = self.current_slot().ok_or(RuntimeError::EmptyKvBatchContext)?;
        let ubatch = self
            .current_ubatch()
            .ok_or(RuntimeError::EmptyKvBatchContext)?;
        build_input_k_indices_for_slot(self.kv_size, ubatch, sinfo)
    }

    pub fn build_input_v_indices(
        &self,
        n_embd_v_gqa: u32,
        v_trans: bool,
    ) -> Result<Vec<i64>, RuntimeError> {
        let sinfo = self.current_slot().ok_or(RuntimeError::EmptyKvBatchContext)?;
        let ubatch = self
            .current_ubatch()
            .ok_or(RuntimeError::EmptyKvBatchContext)?;
        build_input_v_indices_for_slot(self.kv_size, ubatch, sinfo, n_embd_v_gqa, v_trans)
    }

    pub fn layer_execution_plan(
        &self,
        arena: &KvArenaLayout,
        layer_idx: u32,
        n_embd_v_gqa: u32,
        v_trans: bool,
    ) -> Result<KvLayerExecutionPlan, RuntimeError> {
        let sinfo = self.current_slot().ok_or(RuntimeError::EmptyKvBatchContext)?;
        if self.n_kv == 0 {
            return Err(RuntimeError::KvBatchContextNotApplied);
        }

        Ok(KvLayerExecutionPlan {
            layer_idx,
            n_kv: self.n_kv,
            cache_view: arena.view_layer_kv_cache(layer_idx, sinfo, self.n_kv, v_trans)?,
            copy_plan: arena.copy_plan_for_layer(layer_idx, v_trans)?,
            k_indices: self.build_input_k_indices()?,
            v_indices: self.build_input_v_indices(n_embd_v_gqa, v_trans)?,
        })
    }

    pub fn batch_execution_plan(
        &self,
        arena: &KvArenaLayout,
        layout: &KvCacheLayout,
        v_trans: bool,
    ) -> Result<KvBatchExecutionPlan, RuntimeError> {
        if self.n_kv == 0 {
            return Err(RuntimeError::KvBatchContextNotApplied);
        }

        let mut layers = Vec::with_capacity(layout.layers.len());
        for layer in &layout.layers {
            let n_embd_v_gqa = layer.kv_heads.saturating_mul(layer.head_dim);
            layers.push(self.layer_execution_plan(
                arena,
                layer.layer_idx,
                n_embd_v_gqa,
                v_trans,
            )?);
        }

        Ok(KvBatchExecutionPlan {
            n_kv: self.n_kv,
            layers,
        })
    }

    pub fn attention_mask(
        &self,
        kv: &KvCacheState,
        attention: AttentionMode,
        causal: bool,
    ) -> Result<KvAttentionMask, RuntimeError> {
        let sinfo = self.current_slot().ok_or(RuntimeError::EmptyKvBatchContext)?;
        let ubatch = self
            .current_ubatch()
            .ok_or(RuntimeError::EmptyKvBatchContext)?;
        validate_slot_info_shape(ubatch, sinfo)?;

        if self.n_kv == 0 {
            return Err(RuntimeError::KvBatchContextNotApplied);
        }

        let per_stream = sinfo.size();
        let mut stream_to_tokens: std::collections::BTreeMap<u32, Vec<&KvToken>> =
            std::collections::BTreeMap::new();
        for token in &ubatch.tokens {
            let seq_id = *token.seq_ids.first().ok_or(RuntimeError::KvMaskMissingSeqId)?;
            let stream = *kv
                .seq_to_stream
                .get(seq_id as usize)
                .ok_or(RuntimeError::KvMaskUnknownSeqId { seq_id })?;
            stream_to_tokens.entry(stream).or_default().push(token);
        }

        let total_tokens = u32::try_from(ubatch.n_tokens()).map_err(|_| RuntimeError::KvIndexCountOverflow {
            count: ubatch.n_tokens(),
        })?;
        let mut values = vec![f32::NEG_INFINITY; (self.n_kv as usize) * (total_tokens as usize)];
        let swa_type = match attention {
            AttentionMode::FullCausal => KvSwaType::None,
            AttentionMode::SlidingWindow { .. } if kv.swa_type == KvSwaType::None => {
                KvSwaType::Standard
            }
            AttentionMode::SlidingWindow { .. } => kv.swa_type,
        };
        let n_swa = match attention {
            AttentionMode::FullCausal => 0,
            AttentionMode::SlidingWindow { window } => window,
        };

        for (stream_group_idx, stream) in sinfo.strm.iter().copied().enumerate() {
            let tokens = stream_to_tokens
                .get(&stream)
                .ok_or(RuntimeError::KvMaskMissingStreamTokens { stream })?;
            if tokens.len() != per_stream {
                return Err(RuntimeError::KvMaskTokenShapeMismatch {
                    expected: per_stream,
                    actual: tokens.len(),
                    stream,
                });
            }

            let cells = kv
                .streams
                .get(stream as usize)
                .ok_or(RuntimeError::KvMaskUnknownStream { stream })?;

            for (ii, token) in tokens.iter().enumerate() {
                let seq_id = token.seq_ids[0];
                let p1 = token.pos;
                let dst_row = stream_group_idx * per_stream + ii;
                let row_base = dst_row * (self.n_kv as usize);

                for j in 0..(self.n_kv as usize) {
                    if j >= cells.cells.len() {
                        continue;
                    }
                    let cell = &cells.cells[j];
                    if cell.is_empty() || !cell.seq_has(seq_id) {
                        continue;
                    }

                    let p0 = cell.pos.expect("non-empty KV cell must have a position");
                    if causal && p0 > p1 {
                        continue;
                    }
                    if swa_type.is_masked(n_swa, p0, p1) {
                        continue;
                    }

                    values[row_base + j] = 0.0;
                }
            }
        }

        Ok(KvAttentionMask {
            n_tokens: total_tokens,
            n_kv: self.n_kv,
            values,
        })
    }

    pub fn batch_attention_plan(
        &self,
        kv: &KvCacheState,
        layout: &KvCacheLayout,
        causal: bool,
    ) -> Result<KvBatchAttentionPlan, RuntimeError> {
        if self.n_kv == 0 {
            return Err(RuntimeError::KvBatchContextNotApplied);
        }

        let mut layers = Vec::with_capacity(layout.layers.len());
        for layer in &layout.layers {
            layers.push(KvLayerAttentionPlan {
                layer_idx: layer.layer_idx,
                attention: layer.attention,
                mask: self.attention_mask(kv, layer.attention, causal)?,
            });
        }

        Ok(KvBatchAttentionPlan {
            n_kv: self.n_kv,
            layers,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UBatchPlan {
    pub token_offset: u32,
    pub token_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlan {
    pub token_offset: u32,
    pub token_count: u32,
    pub ubatches: Vec<UBatchPlan>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillPlan {
    pub prompt_tokens: u32,
    pub batches: Vec<BatchPlan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionSpan {
    pub start_pos: u32,
    pub token_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayerWritePlan {
    pub layer_idx: u32,
    pub write_pos: u32,
    pub token_count: u32,
    pub attention_span: AttentionSpan,
    pub sharing: KvSharing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionKvWritePlan {
    pub slot: SessionSlot,
    pub start_pos: u32,
    pub token_count: u32,
    pub layers: Vec<KvLayerWritePlan>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundKvLayerWrite {
    pub layer_idx: u32,
    pub storage_idx: u32,
    pub key_byte_offset: u64,
    pub value_byte_offset: u64,
    pub byte_len: u64,
    pub attention_span: AttentionSpan,
    pub sharing: KvSharing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundSessionKvWritePlan {
    pub slot: SessionSlot,
    pub start_pos: u32,
    pub token_count: u32,
    pub layers: Vec<BoundKvLayerWrite>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionView {
    pub slot: SessionSlot,
    pub reserved_ctx: u32,
    pub used_ctx: u32,
    pub decode_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeBatch {
    pub slots: Vec<SessionSlot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SessionState {
    reserved_ctx: u32,
    used_ctx: u32,
    decode_ready: bool,
}

#[derive(Debug)]
pub struct Runtime {
    limits: RuntimeLimits,
    cuda: CudaBackend,
    sessions: Vec<Option<SessionState>>,
    next_decode_slot: usize,
}

impl RuntimeLimits {
    pub fn validate(self) -> Result<Self, RuntimeError> {
        if self.n_ctx == 0 || self.n_batch == 0 || self.n_ubatch == 0 || self.n_seq_max == 0 {
            return Err(RuntimeError::InvalidLimits);
        }
        if self.n_batch > self.n_ctx
            || self.n_ubatch > self.n_batch
            || self.n_seq_max > self.n_batch
        {
            return Err(RuntimeError::InvalidLimits);
        }
        Ok(self)
    }
}

impl Runtime {
    pub fn new(cuda: CudaBackend, limits: RuntimeLimits) -> Result<Self, RuntimeError> {
        let limits = limits.validate()?;
        Ok(Self {
            limits,
            cuda,
            sessions: vec![None; limits.n_seq_max as usize],
            next_decode_slot: 0,
        })
    }

    pub fn limits(&self) -> RuntimeLimits {
        self.limits
    }

    pub fn active_session_count(&self) -> usize {
        self.sessions.iter().filter(|slot| slot.is_some()).count()
    }

    pub fn reserved_ctx_total(&self) -> u32 {
        self.sessions
            .iter()
            .flatten()
            .map(|session| session.reserved_ctx)
            .sum()
    }

    pub fn open_session(&mut self, reserved_ctx: u32) -> Result<SessionSlot, RuntimeError> {
        if reserved_ctx == 0 || reserved_ctx > self.limits.n_ctx {
            return Err(RuntimeError::InvalidSessionReservation {
                reserved_ctx,
                n_ctx: self.limits.n_ctx,
            });
        }

        let free_slot_idx = self
            .sessions
            .iter()
            .position(|slot| slot.is_none())
            .ok_or(RuntimeError::NoFreeSessionSlots)?;

        let reserved_total = self.reserved_ctx_total();
        let available = self.limits.n_ctx.saturating_sub(reserved_total);
        if reserved_ctx > available {
            return Err(RuntimeError::ContextReservationExhausted {
                requested_ctx: reserved_ctx,
                available_ctx: available,
            });
        }

        self.sessions[free_slot_idx] = Some(SessionState {
            reserved_ctx,
            used_ctx: 0,
            decode_ready: false,
        });
        Ok(SessionSlot {
            id: free_slot_idx as u32,
        })
    }

    pub fn close_session(&mut self, slot: SessionSlot) -> Result<(), RuntimeError> {
        let state = self.session_state_mut(slot)?;
        *state = None;
        if self.next_decode_slot >= self.sessions.len() {
            self.next_decode_slot = 0;
        }
        Ok(())
    }

    pub fn session(&self, slot: SessionSlot) -> Option<SessionView> {
        self.session_state(slot).map(|state| SessionView {
            slot,
            reserved_ctx: state.reserved_ctx,
            used_ctx: state.used_ctx,
            decode_ready: state.decode_ready,
        })
    }

    pub fn ingest_tokens(
        &mut self,
        slot: SessionSlot,
        token_count: u32,
    ) -> Result<SessionView, RuntimeError> {
        let state = self
            .session_state_mut(slot)?
            .as_mut()
            .ok_or(RuntimeError::UnknownSessionSlot { slot })?;

        let next_used = state.used_ctx.saturating_add(token_count);
        if next_used > state.reserved_ctx {
            return Err(RuntimeError::SessionContextExceeded {
                slot,
                requested_ctx: next_used,
                reserved_ctx: state.reserved_ctx,
            });
        }

        state.used_ctx = next_used;
        Ok(SessionView {
            slot,
            reserved_ctx: state.reserved_ctx,
            used_ctx: state.used_ctx,
            decode_ready: state.decode_ready,
        })
    }

    pub fn set_decode_ready(
        &mut self,
        slot: SessionSlot,
        decode_ready: bool,
    ) -> Result<(), RuntimeError> {
        let state = self
            .session_state_mut(slot)?
            .as_mut()
            .ok_or(RuntimeError::UnknownSessionSlot { slot })?;
        state.decode_ready = decode_ready;
        Ok(())
    }

    pub fn schedule_decode(&mut self, max_sequences: u32) -> Result<DecodeBatch, RuntimeError> {
        if max_sequences == 0 {
            return Err(RuntimeError::InvalidDecodeBatchSize);
        }

        let limit = max_sequences.min(self.limits.n_batch).min(self.limits.n_seq_max) as usize;
        let mut selected = Vec::new();
        let session_count = self.sessions.len();
        if session_count == 0 {
            return Ok(DecodeBatch { slots: selected });
        }

        let mut scanned = 0usize;
        let mut cursor = self.next_decode_slot;
        while scanned < session_count && selected.len() < limit {
            if let Some(state) = self.sessions[cursor].as_mut() {
                if state.decode_ready {
                    state.decode_ready = false;
                    selected.push(SessionSlot { id: cursor as u32 });
                }
            }
            cursor = (cursor + 1) % session_count;
            scanned += 1;
        }
        self.next_decode_slot = cursor;

        Ok(DecodeBatch { slots: selected })
    }

    pub fn plan_session_kv_write(
        &self,
        slot: SessionSlot,
        token_count: u32,
        layout: &KvCacheLayout,
    ) -> Result<SessionKvWritePlan, RuntimeError> {
        let state = self
            .session_state(slot)
            .ok_or(RuntimeError::UnknownSessionSlot { slot })?;
        let next_used = state.used_ctx.saturating_add(token_count);
        if next_used > state.reserved_ctx {
            return Err(RuntimeError::SessionContextExceeded {
                slot,
                requested_ctx: next_used,
                reserved_ctx: state.reserved_ctx,
            });
        }

        let layers = layout
            .layers
            .iter()
            .map(|layer| {
                let attention_tokens = match layer.attention {
                    AttentionMode::FullCausal => next_used,
                    AttentionMode::SlidingWindow { window } => next_used.min(window),
                };
                let attention_start = next_used.saturating_sub(attention_tokens);
                KvLayerWritePlan {
                    layer_idx: layer.layer_idx,
                    write_pos: state.used_ctx,
                    token_count,
                    attention_span: AttentionSpan {
                        start_pos: attention_start,
                        token_count: attention_tokens,
                    },
                    sharing: layer.sharing,
                }
            })
            .collect();

        Ok(SessionKvWritePlan {
            slot,
            start_pos: state.used_ctx,
            token_count,
            layers,
        })
    }

    pub fn commit_session_kv_write(
        &mut self,
        plan: &SessionKvWritePlan,
    ) -> Result<SessionView, RuntimeError> {
        let state = self
            .session_state_mut(plan.slot)?
            .as_mut()
            .ok_or(RuntimeError::UnknownSessionSlot { slot: plan.slot })?;

        if state.used_ctx != plan.start_pos {
            return Err(RuntimeError::StaleKvWritePlan {
                slot: plan.slot,
                expected_start_pos: state.used_ctx,
                plan_start_pos: plan.start_pos,
            });
        }

        let next_used = state.used_ctx.saturating_add(plan.token_count);
        if next_used > state.reserved_ctx {
            return Err(RuntimeError::SessionContextExceeded {
                slot: plan.slot,
                requested_ctx: next_used,
                reserved_ctx: state.reserved_ctx,
            });
        }

        state.used_ctx = next_used;
        state.decode_ready = true;

        Ok(SessionView {
            slot: plan.slot,
            reserved_ctx: state.reserved_ctx,
            used_ctx: state.used_ctx,
            decode_ready: state.decode_ready,
        })
    }

    pub fn plan_prefill(&self, prompt_tokens: u32) -> Result<PrefillPlan, RuntimeError> {
        if prompt_tokens == 0 {
            return Ok(PrefillPlan {
                prompt_tokens,
                batches: Vec::new(),
            });
        }
        if prompt_tokens > self.limits.n_ctx {
            return Err(RuntimeError::PromptExceedsContext {
                prompt_tokens,
                n_ctx: self.limits.n_ctx,
            });
        }

        let mut batches = Vec::new();
        let mut batch_offset = 0;
        while batch_offset < prompt_tokens {
            let remaining = prompt_tokens - batch_offset;
            let batch_tokens = remaining.min(self.limits.n_batch);

            let mut ubatches = Vec::new();
            let mut ubatch_offset = 0;
            while ubatch_offset < batch_tokens {
                let ubatch_remaining = batch_tokens - ubatch_offset;
                let ubatch_tokens = ubatch_remaining.min(self.limits.n_ubatch);
                ubatches.push(UBatchPlan {
                    token_offset: batch_offset + ubatch_offset,
                    token_count: ubatch_tokens,
                });
                ubatch_offset += ubatch_tokens;
            }

            batches.push(BatchPlan {
                token_offset: batch_offset,
                token_count: batch_tokens,
                ubatches,
            });
            batch_offset += batch_tokens;
        }

        Ok(PrefillPlan {
            prompt_tokens,
            batches,
        })
    }

    pub fn prepare_kv_write_bundle(
        &self,
        cache: &mut ModuleCache,
        batch: &KvBatchExecutionPlan,
        layout: &KvCacheLayout,
        options: NvrtcCompileOptions,
    ) -> Result<PreparedKvWriteBundle, RuntimeError> {
        let launches = batch.cuda_kv_write_launch_plans(layout)?;
        Ok(self.cuda.prepare_kv_write_launches(cache, &launches, options)?)
    }

    pub fn execute_prepared_kv_write_bundle_u16(
        &self,
        bundle: &PreparedKvWriteBundle,
        jobs: &[KvWriteJobU16],
        arena: &mut [u16],
    ) -> Result<(), RuntimeError> {
        execute_prepared_kv_write_bundle_u16(bundle, jobs, arena)?;
        Ok(())
    }

    pub fn execute_prepared_kv_write_bundle_f32(
        &self,
        bundle: &PreparedKvWriteBundle,
        jobs: &[KvWriteJobF32],
        arena: &mut [f32],
    ) -> Result<(), RuntimeError> {
        execute_prepared_kv_write_bundle_f32(bundle, jobs, arena)?;
        Ok(())
    }

    fn session_state(&self, slot: SessionSlot) -> Option<&SessionState> {
        self.sessions.get(slot.id as usize)?.as_ref()
    }

    fn session_state_mut(
        &mut self,
        slot: SessionSlot,
    ) -> Result<&mut Option<SessionState>, RuntimeError> {
        self.sessions
            .get_mut(slot.id as usize)
            .ok_or(RuntimeError::UnknownSessionSlot { slot })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidLimits,
    Cuda(CudaError),
    InvalidKvArenaStreams { n_stream: u32 },
    EmptyKvBatchContext,
    KvBatchContextNotApplied,
    KvIndexCountOverflow { count: usize },
    KvInputElementCountOverflow { elements: u64 },
    KvInputWidthMismatch { expected: u32, actual: u32 },
    KvInputBufferLengthMismatch {
        layer_idx: u32,
        tensor: &'static str,
        expected: usize,
        actual: usize,
    },
    KvMaskMissingSeqId,
    KvMaskUnknownSeqId { seq_id: u32 },
    KvMaskUnknownStream { stream: u32 },
    KvMaskMissingStreamTokens { stream: u32 },
    KvMaskTokenShapeMismatch {
        expected: usize,
        actual: usize,
        stream: u32,
    },
    KvViewExceedsContext { n_kv: u32, max_ctx: u32 },
    KvViewStreamRangeExceedsArena {
        stream_start: u32,
        stream_end: u32,
        n_stream: u32,
    },
    KvSlotUnavailable,
    NonRectangularKvSlotInfo,
    KvIndexTokenMismatch { ubatch_tokens: usize, slot_tokens: usize },
    MissingKvLayoutLayer { layer_idx: u32 },
    MissingKvLayerInput { layer_idx: u32 },
    IncompatibleSharedKvSpec {
        group: u32,
        first_layer: u32,
        conflicting_layer: u32,
    },
    MissingKvStorageForLayer { layer_idx: u32 },
    InvalidSessionReservation { reserved_ctx: u32, n_ctx: u32 },
    ContextReservationExhausted { requested_ctx: u32, available_ctx: u32 },
    NoFreeSessionSlots,
    UnknownSessionSlot { slot: SessionSlot },
    SessionContextExceeded {
        slot: SessionSlot,
        requested_ctx: u32,
        reserved_ctx: u32,
    },
    StaleKvWritePlan {
        slot: SessionSlot,
        expected_start_pos: u32,
        plan_start_pos: u32,
    },
    InvalidDecodeBatchSize,
    PromptExceedsContext { prompt_tokens: u32, n_ctx: u32 },
}

impl Display for RuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLimits => f.write_str("invalid runtime limits"),
            Self::Cuda(source) => write!(f, "CUDA backend error: {source}"),
            Self::InvalidKvArenaStreams { n_stream } => write!(
                f,
                "KV arena requires at least one stream, got {}",
                n_stream
            ),
            Self::EmptyKvBatchContext => f.write_str("KV batch context has no current ubatch"),
            Self::KvBatchContextNotApplied => {
                f.write_str("KV batch context must be applied before building layer execution plans")
            }
            Self::KvIndexCountOverflow { count } => {
                write!(f, "KV index count {} does not fit into u32", count)
            }
            Self::KvInputElementCountOverflow { elements } => write!(
                f,
                "KV input element count {} does not fit into host usize",
                elements
            ),
            Self::KvInputWidthMismatch { expected, actual } => write!(
                f,
                "KV input width {} does not match expected width {}",
                actual, expected
            ),
            Self::KvInputBufferLengthMismatch {
                layer_idx,
                tensor,
                expected,
                actual,
            } => write!(
                f,
                "KV {} input for layer {} has {} elements, expected {}",
                tensor, layer_idx, actual, expected
            ),
            Self::KvMaskMissingSeqId => f.write_str("KV mask token is missing a sequence id"),
            Self::KvMaskUnknownSeqId { seq_id } => {
                write!(f, "KV mask sequence {} is unknown to the cache", seq_id)
            }
            Self::KvMaskUnknownStream { stream } => {
                write!(f, "KV mask stream {} is unknown to the cache", stream)
            }
            Self::KvMaskMissingStreamTokens { stream } => {
                write!(f, "KV mask is missing tokens for stream {}", stream)
            }
            Self::KvMaskTokenShapeMismatch {
                expected,
                actual,
                stream,
            } => write!(
                f,
                "KV mask stream {} expected {} tokens but found {}",
                stream, expected, actual
            ),
            Self::KvViewExceedsContext { n_kv, max_ctx } => write!(
                f,
                "KV view length {} exceeds storage context {}",
                n_kv, max_ctx
            ),
            Self::KvViewStreamRangeExceedsArena {
                stream_start,
                stream_end,
                n_stream,
            } => write!(
                f,
                "KV view stream range {}..={} exceeds arena stream count {}",
                stream_start, stream_end, n_stream
            ),
            Self::KvSlotUnavailable => {
                f.write_str("KV cache could not allocate a slot for the requested ubatch")
            }
            Self::NonRectangularKvSlotInfo => {
                f.write_str("KV slot info is not rectangular across streams")
            }
            Self::KvIndexTokenMismatch {
                ubatch_tokens,
                slot_tokens,
            } => write!(
                f,
                "KV slot info covers {} tokens but ubatch has {}",
                slot_tokens, ubatch_tokens
            ),
            Self::MissingKvLayoutLayer { layer_idx } => {
                write!(f, "missing KV layout entry for layer {}", layer_idx)
            }
            Self::MissingKvLayerInput { layer_idx } => {
                write!(f, "missing KV input buffers for layer {}", layer_idx)
            }
            Self::IncompatibleSharedKvSpec {
                group,
                first_layer,
                conflicting_layer,
            } => write!(
                f,
                "shared KV group {} is inconsistent between layers {} and {}",
                group, first_layer, conflicting_layer
            ),
            Self::MissingKvStorageForLayer { layer_idx } => {
                write!(f, "missing KV storage for layer {}", layer_idx)
            }
            Self::InvalidSessionReservation {
                reserved_ctx,
                n_ctx,
            } => write!(
                f,
                "session reservation {} is invalid for runtime context {}",
                reserved_ctx, n_ctx
            ),
            Self::ContextReservationExhausted {
                requested_ctx,
                available_ctx,
            } => write!(
                f,
                "session reservation {} exceeds remaining context {}",
                requested_ctx, available_ctx
            ),
            Self::NoFreeSessionSlots => f.write_str("no free session slots remain"),
            Self::UnknownSessionSlot { slot } => write!(f, "unknown session slot {}", slot.id),
            Self::SessionContextExceeded {
                slot,
                requested_ctx,
                reserved_ctx,
            } => write!(
                f,
                "session {} requested context {} beyond reserved {}",
                slot.id, requested_ctx, reserved_ctx
            ),
            Self::StaleKvWritePlan {
                slot,
                expected_start_pos,
                plan_start_pos,
            } => write!(
                f,
                "stale KV write plan for session {}: expected start {}, got {}",
                slot.id, expected_start_pos, plan_start_pos
            ),
            Self::InvalidDecodeBatchSize => {
                f.write_str("decode scheduling requires a positive batch size")
            }
            Self::PromptExceedsContext {
                prompt_tokens,
                n_ctx,
            } => write!(
                f,
                "prompt length {} exceeds runtime context {}",
                prompt_tokens, n_ctx
            ),
        }
    }
}

impl Error for RuntimeError {}

impl From<CudaError> for RuntimeError {
    fn from(value: CudaError) -> Self {
        Self::Cuda(value)
    }
}

#[cfg(test)]
mod tests {
    use chew_cuda::{CudaBackend, DeviceInfo, ModuleCache, NvrtcCompileOptions};

    use super::*;

    fn runtime(n_ctx: u32, n_batch: u32, n_ubatch: u32) -> Runtime {
        Runtime::new(
            CudaBackend::new(DeviceInfo { ordinal: 0 }).unwrap(),
            RuntimeLimits {
                n_ctx,
                n_batch,
                n_ubatch,
                n_seq_max: 4,
            },
        )
        .unwrap()
    }

    #[test]
    fn rejects_invalid_limit_ordering() {
        let err = Runtime::new(
            CudaBackend::new(DeviceInfo { ordinal: 0 }).unwrap(),
            RuntimeLimits {
                n_ctx: 1024,
                n_batch: 2048,
                n_ubatch: 512,
                n_seq_max: 4,
            },
        )
        .unwrap_err();
        assert_eq!(err, RuntimeError::InvalidLimits);
    }

    #[test]
    fn plans_prefill_across_batches_and_ubatches() {
        let runtime = runtime(8192, 2048, 512);
        let plan = runtime.plan_prefill(5000).unwrap();

        assert_eq!(plan.prompt_tokens, 5000);
        assert_eq!(plan.batches.len(), 3);

        assert_eq!(plan.batches[0].token_offset, 0);
        assert_eq!(plan.batches[0].token_count, 2048);
        assert_eq!(plan.batches[0].ubatches.len(), 4);
        assert_eq!(
            plan.batches[0].ubatches[3],
            UBatchPlan {
                token_offset: 1536,
                token_count: 512,
            }
        );

        assert_eq!(plan.batches[1].token_offset, 2048);
        assert_eq!(plan.batches[1].token_count, 2048);
        assert_eq!(plan.batches[1].ubatches.len(), 4);

        assert_eq!(plan.batches[2].token_offset, 4096);
        assert_eq!(plan.batches[2].token_count, 904);
        assert_eq!(plan.batches[2].ubatches.len(), 2);
        assert_eq!(
            plan.batches[2].ubatches[1],
            UBatchPlan {
                token_offset: 4608,
                token_count: 392,
            }
        );
    }

    #[test]
    fn rejects_prompt_longer_than_context() {
        let runtime = runtime(1024, 512, 256);
        let err = runtime.plan_prefill(1025).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::PromptExceedsContext {
                prompt_tokens: 1025,
                n_ctx: 1024,
            }
        );
    }

    #[test]
    fn opens_and_closes_session_slots() {
        let mut runtime = runtime(4096, 1024, 256);
        let a = runtime.open_session(1024).unwrap();
        let b = runtime.open_session(512).unwrap();

        assert_eq!(a.id, 0);
        assert_eq!(b.id, 1);
        assert_eq!(runtime.active_session_count(), 2);
        assert_eq!(runtime.reserved_ctx_total(), 1536);

        runtime.close_session(a).unwrap();
        assert_eq!(runtime.active_session_count(), 1);
        assert_eq!(runtime.reserved_ctx_total(), 512);

        let c = runtime.open_session(256).unwrap();
        assert_eq!(c.id, 0);
    }

    #[test]
    fn rejects_session_reservations_when_slots_or_context_are_exhausted() {
        let mut saturated = runtime(2048, 1024, 256);
        saturated.open_session(512).unwrap();
        saturated.open_session(512).unwrap();
        saturated.open_session(512).unwrap();
        saturated.open_session(512).unwrap();

        let slot_err = saturated.open_session(128).unwrap_err();
        assert_eq!(slot_err, RuntimeError::NoFreeSessionSlots);

        let mut ctx_limited = runtime(1024, 1024, 256);
        ctx_limited.open_session(800).unwrap();
        let ctx_err = ctx_limited.open_session(300).unwrap_err();
        assert_eq!(
            ctx_err,
            RuntimeError::ContextReservationExhausted {
                requested_ctx: 300,
                available_ctx: 224,
            }
        );
    }

    #[test]
    fn session_token_accounting_stays_within_reserved_context() {
        let mut runtime = runtime(4096, 1024, 256);
        let slot = runtime.open_session(600).unwrap();

        let view = runtime.ingest_tokens(slot, 512).unwrap();
        assert_eq!(view.used_ctx, 512);

        let err = runtime.ingest_tokens(slot, 128).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::SessionContextExceeded {
                slot,
                requested_ctx: 640,
                reserved_ctx: 600,
            }
        );
    }

    #[test]
    fn schedules_decode_round_robin_over_ready_sessions() {
        let mut runtime = runtime(4096, 1024, 256);
        let a = runtime.open_session(512).unwrap();
        let b = runtime.open_session(512).unwrap();
        let c = runtime.open_session(512).unwrap();

        runtime.set_decode_ready(a, true).unwrap();
        runtime.set_decode_ready(b, true).unwrap();
        runtime.set_decode_ready(c, true).unwrap();

        let batch1 = runtime.schedule_decode(2).unwrap();
        assert_eq!(batch1.slots, vec![a, b]);

        runtime.set_decode_ready(a, true).unwrap();
        runtime.set_decode_ready(b, true).unwrap();

        let batch2 = runtime.schedule_decode(2).unwrap();
        assert_eq!(batch2.slots, vec![c, a]);

        let batch3 = runtime.schedule_decode(2).unwrap();
        assert_eq!(batch3.slots, vec![b]);
    }

    #[test]
    fn kv_cache_layout_summarizes_attention_and_sharing() {
        let layout = KvCacheLayout {
            layers: vec![
                LayerKvSpec {
                    layer_idx: 0,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::FullCausal,
                    sharing: KvSharing::Dedicated,
                },
                LayerKvSpec {
                    layer_idx: 1,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 1024 },
                    sharing: KvSharing::Shared { group: 0 },
                },
                LayerKvSpec {
                    layer_idx: 2,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 1024 },
                    sharing: KvSharing::Shared { group: 0 },
                },
            ],
        };

        assert_eq!(layout.layer_count(), 3);
        assert_eq!(layout.full_attention_layer_count(), 1);
        assert_eq!(layout.sliding_window_layer_count(), 2);
        assert_eq!(layout.shared_group_count(), 1);
        assert_eq!(layout.dedicated_layer_count(), 1);
    }

    #[test]
    fn kv_arena_coalesces_shared_storage_and_counts_bytes() {
        let layout = KvCacheLayout {
            layers: vec![
                LayerKvSpec {
                    layer_idx: 0,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::FullCausal,
                    sharing: KvSharing::Dedicated,
                },
                LayerKvSpec {
                    layer_idx: 1,
                    kv_heads: 4,
                    head_dim: 256,
                    attention: AttentionMode::SlidingWindow { window: 1024 },
                    sharing: KvSharing::Shared { group: 7 },
                },
                LayerKvSpec {
                    layer_idx: 2,
                    kv_heads: 4,
                    head_dim: 256,
                    attention: AttentionMode::SlidingWindow { window: 1024 },
                    sharing: KvSharing::Shared { group: 7 },
                },
            ],
        };

        let arena = KvArenaLayout::for_layout(&layout, 2048, 3, KvScalarType::F16).unwrap();
        assert_eq!(arena.storages.len(), 2);
        assert_eq!(arena.layer_to_storage, vec![0, 1, 1]);
        assert_eq!(arena.storages[0].n_stream, 3);
        assert_eq!(arena.storages[0].bytes_per_stream_cache, 2048 * 8 * 128 * 2);
        assert_eq!(arena.storages[0].bytes_per_tensor, 3 * 2048 * 8 * 128 * 2);
        assert_eq!(arena.storages[1].bytes_per_stream_cache, 2048 * 4 * 256 * 2);
        assert_eq!(arena.storages[1].bytes_per_tensor, 3 * 2048 * 4 * 256 * 2);
        assert_eq!(
            arena.total_bytes,
            arena.storages[0].total_bytes + arena.storages[1].total_bytes
        );
    }

    #[test]
    fn kv_arena_rejects_incompatible_shared_group_shapes() {
        let layout = KvCacheLayout {
            layers: vec![
                LayerKvSpec {
                    layer_idx: 0,
                    kv_heads: 4,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 1024 },
                    sharing: KvSharing::Shared { group: 3 },
                },
                LayerKvSpec {
                    layer_idx: 1,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 1024 },
                    sharing: KvSharing::Shared { group: 3 },
                },
            ],
        };

        let err = KvArenaLayout::for_layout(&layout, 2048, 1, KvScalarType::F16).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::IncompatibleSharedKvSpec {
                group: 3,
                first_layer: 0,
                conflicting_layer: 1,
            }
        );
    }

    #[test]
    fn plans_session_kv_writes_for_full_and_sliding_attention() {
        let mut runtime = runtime(4096, 1024, 256);
        let slot = runtime.open_session(512).unwrap();
        runtime.ingest_tokens(slot, 100).unwrap();

        let layout = KvCacheLayout {
            layers: vec![
                LayerKvSpec {
                    layer_idx: 0,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::FullCausal,
                    sharing: KvSharing::Dedicated,
                },
                LayerKvSpec {
                    layer_idx: 1,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 64 },
                    sharing: KvSharing::Shared { group: 0 },
                },
            ],
        };

        let plan = runtime.plan_session_kv_write(slot, 16, &layout).unwrap();
        assert_eq!(plan.slot, slot);
        assert_eq!(plan.start_pos, 100);
        assert_eq!(plan.token_count, 16);
        assert_eq!(plan.layers.len(), 2);

        assert_eq!(
            plan.layers[0],
            KvLayerWritePlan {
                layer_idx: 0,
                write_pos: 100,
                token_count: 16,
                attention_span: AttentionSpan {
                    start_pos: 0,
                    token_count: 116,
                },
                sharing: KvSharing::Dedicated,
            }
        );
        assert_eq!(
            plan.layers[1],
            KvLayerWritePlan {
                layer_idx: 1,
                write_pos: 100,
                token_count: 16,
                attention_span: AttentionSpan {
                    start_pos: 52,
                    token_count: 64,
                },
                sharing: KvSharing::Shared { group: 0 },
            }
        );
    }

    #[test]
    fn kv_arena_binds_session_write_plan_to_storage_offsets() {
        let mut runtime = runtime(4096, 1024, 256);
        let slot = runtime.open_session(512).unwrap();
        runtime.ingest_tokens(slot, 100).unwrap();

        let layout = KvCacheLayout {
            layers: vec![
                LayerKvSpec {
                    layer_idx: 0,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::FullCausal,
                    sharing: KvSharing::Dedicated,
                },
                LayerKvSpec {
                    layer_idx: 1,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 64 },
                    sharing: KvSharing::Shared { group: 0 },
                },
                LayerKvSpec {
                    layer_idx: 2,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 64 },
                    sharing: KvSharing::Shared { group: 0 },
                },
            ],
        };
        let arena = KvArenaLayout::for_layout(&layout, 4096, 1, KvScalarType::F16).unwrap();
        let plan = runtime.plan_session_kv_write(slot, 16, &layout).unwrap();
        let bound = arena.bind_session_kv_write(&plan).unwrap();

        assert_eq!(bound.layers.len(), 3);
        assert_eq!(bound.layers[1].storage_idx, bound.layers[2].storage_idx);
        assert_eq!(bound.layers[0].key_byte_offset, 100 * 8 * 128 * 2);
        assert_eq!(
            bound.layers[0].byte_len,
            16_u64 * 8_u64 * 128_u64 * 2_u64
        );
    }

    #[test]
    fn kv_arena_builds_llama_style_layer_views_with_stream_slices() {
        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 8,
                head_dim: 128,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };
        let arena = KvArenaLayout::for_layout(&layout, 4096, 4, KvScalarType::F16).unwrap();
        let sinfo = KvSlotInfo {
            s0: 1,
            s1: 2,
            strm: vec![1, 2],
            idxs: vec![vec![10, 11], vec![12, 13]],
        };

        let non_trans = arena.view_layer_kv_cache(0, &sinfo, 512, false).unwrap();
        assert_eq!(non_trans.stream_start, 1);
        assert_eq!(non_trans.stream_count, 2);
        assert_eq!(non_trans.key.dims, [128, 8, 512, 2]);
        assert_eq!(non_trans.key.strides, [2, 256, 2048, 4096 * 8 * 128 * 2]);
        assert_eq!(
            non_trans.key.byte_offset,
            arena.storages[0].key_bytes_offset + arena.storages[0].bytes_per_stream_cache
        );
        assert_eq!(non_trans.value.dims, [128, 8, 512, 2]);
        assert_eq!(non_trans.value.strides, [2, 256, 2048, 4096 * 8 * 128 * 2]);

        let trans = arena.view_layer_kv_cache(0, &sinfo, 512, true).unwrap();
        assert_eq!(trans.value.dims, [512, 8, 128, 2]);
        assert_eq!(trans.value.strides, [2, 4096 * 128 * 2, 4096 * 2, 4096 * 8 * 128 * 2]);
    }

    #[test]
    fn kv_arena_rejects_invalid_stream_count_and_view_ranges() {
        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 8,
                head_dim: 128,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };

        let err = KvArenaLayout::for_layout(&layout, 4096, 0, KvScalarType::F16).unwrap_err();
        assert_eq!(err, RuntimeError::InvalidKvArenaStreams { n_stream: 0 });

        let arena = KvArenaLayout::for_layout(&layout, 4096, 2, KvScalarType::F16).unwrap();
        let sinfo = KvSlotInfo {
            s0: 1,
            s1: 2,
            strm: vec![1],
            idxs: vec![vec![10]],
        };

        let err = arena.view_layer_kv_cache(0, &sinfo, 128, false).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::KvViewStreamRangeExceedsArena {
                stream_start: 1,
                stream_end: 2,
                n_stream: 2,
            }
        );
    }

    #[test]
    fn kv_arena_builds_llama_style_copy_targets() {
        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 8,
                head_dim: 128,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };
        let arena = KvArenaLayout::for_layout(&layout, 4096, 4, KvScalarType::F16).unwrap();

        let non_trans = arena.copy_plan_for_layer(0, false).unwrap();
        assert_eq!(non_trans.key.row_width_el, 8 * 128);
        assert_eq!(non_trans.key.row_count, 4096 * 4);
        assert_eq!(non_trans.key.row_stride_bytes, 8 * 128 * 2);
        assert_eq!(non_trans.value.row_width_el, 8 * 128);
        assert_eq!(non_trans.value.row_count, 4096 * 4);
        assert_eq!(non_trans.value.row_stride_bytes, 8 * 128 * 2);

        let trans = arena.copy_plan_for_layer(0, true).unwrap();
        assert_eq!(trans.value.row_width_el, 1);
        assert_eq!(trans.value.row_count, 4096 * 4 * 8 * 128);
        assert_eq!(trans.value.row_stride_bytes, 2);
    }

    #[test]
    fn kv_cache_state_prepare_does_not_mutate_live_cells() {
        let state = KvCacheState::new(16, 4, true);
        let ubatch = KvUbatch::single_seq(0, &[0, 1, 2, 3]);

        let slots = state.prepare(&[ubatch], true).unwrap();
        assert_eq!(slots.len(), 1);
        assert_eq!(state.stream_head(0), Some(0));
    }

    #[test]
    fn kv_cache_state_find_slot_and_apply_follow_llama_style_slot_info() {
        let mut state = KvCacheState::new(16, 4, true);
        let ubatch = KvUbatch::single_seq(0, &[10, 11, 12, 13]);

        let sinfo = state.find_slot(&ubatch, true).unwrap();
        assert_eq!(sinfo.s0, 0);
        assert_eq!(sinfo.s1, 0);
        assert_eq!(sinfo.n_stream(), 1);
        assert!(sinfo.is_contiguous());
        assert_eq!(sinfo.head(), Some(0));

        state.apply_ubatch(&sinfo, &ubatch);
        assert_eq!(state.stream_head(0), Some(4));
    }

    #[test]
    fn kv_cache_state_non_unified_maps_sequences_to_distinct_streams() {
        let state = KvCacheState::new(8, 4, false);
        let ubatch = KvUbatch::single_seq(2, &[5, 6]);
        let sinfo = state.find_slot(&ubatch, true).unwrap();

        assert_eq!(state.n_stream(), 4);
        assert_eq!(sinfo.strm, vec![2]);
        assert_eq!(sinfo.idxs[0], vec![0, 1]);
    }

    #[test]
    fn kv_cache_state_find_slot_resets_search_head_like_llama_cpp() {
        let mut state = KvCacheState::new(16, 4, true);
        state.streams[0].head = 15;
        state.streams[0].cells[0].pos = Some(1);
        state.streams[0].cells[0].seq_ids = vec![0];

        let ubatch = KvUbatch::single_seq(0, &[2, 3]);
        let sinfo = state.find_slot(&ubatch, true).unwrap();
        assert_eq!(sinfo.head(), Some(1));
    }

    #[test]
    fn kv_swa_mask_matches_local_llama_cpp_reference_cases() {
        assert!(!KvSwaType::None.is_masked(4, 0, 10));

        assert!(KvSwaType::Standard.is_masked(4, 0, 10));
        assert!(!KvSwaType::Standard.is_masked(16, 10, 20));

        assert!(KvSwaType::Chunked.is_masked(4, 3, 9));
        assert!(!KvSwaType::Chunked.is_masked(4, 8, 9));

        assert!(KvSwaType::Symmetric.is_masked(8, 0, 10));
        assert!(!KvSwaType::Symmetric.is_masked(8, 7, 10));
    }

    #[test]
    fn kv_cache_state_reuses_single_seq_cell_when_swa_masks_it() {
        let mut state = KvCacheState::new_with_swa(16, 4, true, 4, KvSwaType::Standard);
        state.streams[0].cells[0].pos = Some(0);
        state.streams[0].cells[0].seq_ids = vec![0];
        state.streams[0].cells[1].pos = Some(9);
        state.streams[0].cells[1].seq_ids = vec![0];

        let ubatch = KvUbatch::single_seq(0, &[10]);
        let sinfo = state.find_slot(&ubatch, false).unwrap();
        assert_eq!(sinfo.idxs[0], vec![0]);
    }

    #[test]
    fn kv_cache_state_get_n_kv_pads_like_local_llama_cpp() {
        let mut state = KvCacheState::new_with_policy(1024, 4, true, 32, 0, KvSwaType::None);
        state.streams[0].cells[0].pos = Some(0);
        state.streams[0].cells[0].seq_ids = vec![0];
        state.streams[0].cells[260].pos = Some(260);
        state.streams[0].cells[260].seq_ids = vec![0];

        let sinfo = KvSlotInfo {
            s0: 0,
            s1: 0,
            strm: vec![0],
            idxs: vec![vec![261, 262]],
        };

        assert_eq!(state.get_n_kv(&sinfo), 512);
    }

    #[test]
    fn kv_cache_state_builds_global_k_and_v_indices_like_local_llama_cpp() {
        let state = KvCacheState::new(16, 4, false);
        let ubatch = KvUbatch {
            tokens: vec![
                KvToken {
                    pos: 10,
                    seq_ids: vec![0],
                },
                KvToken {
                    pos: 11,
                    seq_ids: vec![0],
                },
                KvToken {
                    pos: 20,
                    seq_ids: vec![2],
                },
                KvToken {
                    pos: 21,
                    seq_ids: vec![2],
                },
            ],
        };
        let sinfo = KvSlotInfo {
            s0: 0,
            s1: 2,
            strm: vec![0, 2],
            idxs: vec![vec![3, 4], vec![1, 2]],
        };

        let k = state.build_input_k_indices(&ubatch, &sinfo).unwrap();
        let v = state.build_input_v_indices(&ubatch, &sinfo, 3, true).unwrap();

        assert_eq!(k, vec![3, 4, 33, 34]);
        assert_eq!(v, vec![3, 19, 35, 4, 20, 36, 97, 113, 129, 98, 114, 130]);
    }

    #[test]
    fn kv_cache_state_rejects_non_rectangular_slot_info_for_index_builders() {
        let state = KvCacheState::new(16, 4, false);
        let ubatch = KvUbatch {
            tokens: vec![
                KvToken {
                    pos: 10,
                    seq_ids: vec![0],
                },
                KvToken {
                    pos: 11,
                    seq_ids: vec![0],
                },
                KvToken {
                    pos: 20,
                    seq_ids: vec![2],
                },
            ],
        };
        let sinfo = KvSlotInfo {
            s0: 0,
            s1: 2,
            strm: vec![0, 2],
            idxs: vec![vec![3, 4], vec![1]],
        };

        let err = state.build_input_k_indices(&ubatch, &sinfo).unwrap_err();
        assert_eq!(err, RuntimeError::NonRectangularKvSlotInfo);
    }

    #[test]
    fn kv_cache_batch_context_applies_prepared_slots_like_local_llama_cpp() {
        let mut state = KvCacheState::new(16, 4, true);
        let ubatches = vec![
            KvUbatch::single_seq(0, &[10, 11]),
            KvUbatch::single_seq(0, &[12]),
        ];

        let mut ctx = state.prepare_context(ubatches, true).unwrap();
        assert_eq!(ctx.len(), 2);
        assert_eq!(ctx.current_slot().unwrap().idxs[0], vec![0, 1]);
        assert_eq!(ctx.build_input_k_indices().unwrap(), vec![0, 1]);

        ctx.apply(&mut state).unwrap();
        assert_eq!(ctx.n_kv(), 16);
        assert_eq!(state.stream_head(0), Some(2));

        assert!(ctx.next());
        assert_eq!(ctx.current_slot().unwrap().idxs[0], vec![2]);
        assert_eq!(ctx.build_input_k_indices().unwrap(), vec![2]);

        ctx.apply(&mut state).unwrap();
        assert_eq!(ctx.n_kv(), 16);
        assert_eq!(state.stream_head(0), Some(3));
        assert!(!ctx.next());
    }

    #[test]
    fn kv_cache_batch_context_builds_layer_execution_plan_after_apply() {
        let mut state = KvCacheState::new_with_policy(4096, 4, false, 32, 0, KvSwaType::None);
        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 8,
                head_dim: 128,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };
        let arena = KvArenaLayout::for_layout(&layout, 4096, state.n_stream(), KvScalarType::F16).unwrap();
        let ubatches = vec![KvUbatch {
            tokens: vec![
                KvToken {
                    pos: 100,
                    seq_ids: vec![0],
                },
                KvToken {
                    pos: 101,
                    seq_ids: vec![0],
                },
                KvToken {
                    pos: 200,
                    seq_ids: vec![2],
                },
                KvToken {
                    pos: 201,
                    seq_ids: vec![2],
                },
            ],
        }];

        let mut ctx = state.prepare_context(ubatches, true).unwrap();
        let err = ctx.layer_execution_plan(&arena, 0, 8 * 128, true).unwrap_err();
        assert_eq!(err, RuntimeError::KvBatchContextNotApplied);

        ctx.apply(&mut state).unwrap();
        let plan = ctx.layer_execution_plan(&arena, 0, 8 * 128, true).unwrap();
        assert_eq!(plan.n_kv, 256);
        assert_eq!(plan.cache_view.stream_start, 0);
        assert_eq!(plan.cache_view.stream_count, 3);
        assert_eq!(plan.copy_plan.value.row_count, 4096 * 4 * 8 * 128);
        assert_eq!(plan.k_indices, vec![0, 1, 8192, 8193]);
        assert_eq!(plan.v_indices.len(), 4 * 8 * 128);

        let input_plan = plan.input_copy_plan(8 * 128, 8 * 128).unwrap();
        let cuda_plan = plan.cuda_kv_write_launch_plan(8 * 128, 8 * 128).unwrap();
        assert_eq!(input_plan.key.row_width_el, 8 * 128);
        assert_eq!(input_plan.key.row_count, 4);
        assert_eq!(input_plan.value.row_width_el, 1);
        assert_eq!(input_plan.value.row_count, (4 * 8 * 128) as u32);
        assert_eq!(cuda_plan.key.module_name, "kv_cache");
        assert_eq!(cuda_plan.key.kind, KvWriteKernelKind::KeyRows);
        assert_eq!(cuda_plan.key.src_row_width_el, 8 * 128);
        assert_eq!(cuda_plan.key.index_count, 4);
        assert_eq!(cuda_plan.value.kind, KvWriteKernelKind::ValueRowsTransposed);
        assert_eq!(cuda_plan.value.src_row_width_el, 1);
        assert_eq!(cuda_plan.value.index_count, (4 * 8 * 128) as u32);
    }

    #[test]
    fn kv_cache_batch_context_builds_batch_execution_plan_for_full_layout() {
        let mut state = KvCacheState::new_with_policy(4096, 4, true, 32, 0, KvSwaType::None);
        let layout = KvCacheLayout {
            layers: vec![
                LayerKvSpec {
                    layer_idx: 0,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::FullCausal,
                    sharing: KvSharing::Dedicated,
                },
                LayerKvSpec {
                    layer_idx: 1,
                    kv_heads: 8,
                    head_dim: 128,
                    attention: AttentionMode::SlidingWindow { window: 1024 },
                    sharing: KvSharing::Shared { group: 0 },
                },
            ],
        };
        let arena = KvArenaLayout::for_layout(&layout, 4096, state.n_stream(), KvScalarType::F16).unwrap();
        let mut ctx = state
            .prepare_context(vec![KvUbatch::single_seq(0, &[100, 101, 102, 103])], true)
            .unwrap();

        ctx.apply(&mut state).unwrap();
        let batch = ctx.batch_execution_plan(&arena, &layout, true).unwrap();
        assert_eq!(batch.n_kv, 256);
        assert_eq!(batch.layers.len(), 2);
        assert_eq!(batch.layers[0].layer_idx, 0);
        assert_eq!(batch.layers[1].layer_idx, 1);
        assert_eq!(batch.layers[0].cache_view.storage_idx, 0);
        assert_eq!(batch.layers[1].cache_view.storage_idx, 1);

        let launches = batch.cuda_kv_write_launch_plans(&layout).unwrap();
        assert_eq!(launches.len(), 2);
        assert_eq!(launches[0].key.module_name, "kv_cache");
        assert_eq!(launches[1].value.kind, KvWriteKernelKind::ValueRowsTransposed);
    }

    #[test]
    fn runtime_batch_plan_writes_expected_kv_positions_via_cuda_emulation() {
        let mut state = KvCacheState::new_with_policy(128, 4, true, 32, 0, KvSwaType::None);
        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 2,
                head_dim: 2,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };
        let arena = KvArenaLayout::for_layout(&layout, 128, state.n_stream(), KvScalarType::F16).unwrap();
        let mut ctx = state
            .prepare_context(vec![KvUbatch::single_seq(0, &[10, 11])], true)
            .unwrap();
        ctx.apply(&mut state).unwrap();

        let batch = ctx.batch_execution_plan(&arena, &layout, true).unwrap();
        let mut cache = ModuleCache::default();
        let runtime = runtime(128, 64, 64);
        let bundle = runtime
            .prepare_kv_write_bundle(&mut cache, &batch, &layout, NvrtcCompileOptions::default())
            .unwrap();

        let k_src = vec![10_u16, 11, 12, 13, 20, 21, 22, 23];
        let mut arena_u16 = vec![0_u16; (arena.total_bytes / 2) as usize];
        let v_src = vec![100_u16, 101, 102, 103, 200, 201, 202, 203];
        let jobs = batch
            .jobs_u16(
                &layout,
                &[KvLayerInputU16 {
                    layer_idx: 0,
                    key_src: k_src,
                    value_src: v_src,
                }],
            )
            .unwrap();
        runtime
            .execute_prepared_kv_write_bundle_u16(&bundle, &jobs, &mut arena_u16)
            .unwrap();
        assert_eq!(&arena_u16[0..4], &[10, 11, 12, 13]);
        assert_eq!(&arena_u16[4..8], &[20, 21, 22, 23]);

        let v_base = (batch.layers[0].copy_plan.value.byte_offset / 2) as usize;
        assert_eq!(arena_u16[v_base], 100);
        assert_eq!(arena_u16[v_base + 128], 101);
        assert_eq!(arena_u16[v_base + 256], 102);
        assert_eq!(arena_u16[v_base + 384], 103);
        assert_eq!(arena_u16[v_base + 1], 200);
        assert_eq!(arena_u16[v_base + 129], 201);
        assert_eq!(arena_u16[v_base + 257], 202);
        assert_eq!(arena_u16[v_base + 385], 203);
    }

    #[test]
    fn runtime_non_unified_batch_plan_writes_global_stream_offsets_via_cuda_emulation() {
        let mut state = KvCacheState::new_with_policy(128, 4, false, 32, 0, KvSwaType::None);
        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 2,
                head_dim: 2,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };
        let arena = KvArenaLayout::for_layout(&layout, 128, state.n_stream(), KvScalarType::F16).unwrap();
        let mut ctx = state
            .prepare_context(
                vec![KvUbatch {
                    tokens: vec![
                        KvToken {
                            pos: 10,
                            seq_ids: vec![0],
                        },
                        KvToken {
                            pos: 11,
                            seq_ids: vec![0],
                        },
                        KvToken {
                            pos: 20,
                            seq_ids: vec![2],
                        },
                        KvToken {
                            pos: 21,
                            seq_ids: vec![2],
                        },
                    ],
                }],
                true,
            )
            .unwrap();
        ctx.apply(&mut state).unwrap();

        let batch = ctx.batch_execution_plan(&arena, &layout, false).unwrap();
        let mut cache = ModuleCache::default();
        let runtime = runtime(128, 64, 64);
        let bundle = runtime
            .prepare_kv_write_bundle(&mut cache, &batch, &layout, NvrtcCompileOptions::default())
            .unwrap();
        assert_eq!(batch.layers[0].k_indices, vec![0, 1, 256, 257]);
        assert_eq!(batch.layers[0].v_indices, vec![0, 1, 256, 257]);

        let mut arena_u16 = vec![0_u16; (arena.total_bytes / 2) as usize];
        let k_src = vec![10_u16, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 40, 41, 42, 43];
        let v_src = vec![100_u16, 101, 102, 103, 200, 201, 202, 203, 300, 301, 302, 303, 400, 401, 402, 403];
        let jobs = batch
            .jobs_u16(
                &layout,
                &[KvLayerInputU16 {
                    layer_idx: 0,
                    key_src: k_src,
                    value_src: v_src,
                }],
            )
            .unwrap();
        runtime
            .execute_prepared_kv_write_bundle_u16(&bundle, &jobs, &mut arena_u16)
            .unwrap();
        assert_eq!(&arena_u16[0..4], &[10, 11, 12, 13]);
        assert_eq!(&arena_u16[4..8], &[20, 21, 22, 23]);
        assert_eq!(&arena_u16[1024..1028], &[30, 31, 32, 33]);
        assert_eq!(&arena_u16[1028..1032], &[40, 41, 42, 43]);

        let v_base = (batch.layers[0].copy_plan.value.byte_offset / 2) as usize;
        assert_eq!(&arena_u16[v_base..v_base + 4], &[100, 101, 102, 103]);
        assert_eq!(&arena_u16[v_base + 4..v_base + 8], &[200, 201, 202, 203]);
        assert_eq!(&arena_u16[v_base + 1024..v_base + 1028], &[300, 301, 302, 303]);
        assert_eq!(&arena_u16[v_base + 1028..v_base + 1032], &[400, 401, 402, 403]);
    }

    #[test]
    fn kv_attention_mask_applies_causal_rules_and_keeps_padding_masked() {
        let mut state = KvCacheState::new_with_policy(512, 4, true, 32, 0, KvSwaType::None);
        let mut ctx = state
            .prepare_context(vec![KvUbatch::single_seq(0, &[10, 11])], true)
            .unwrap();
        ctx.apply(&mut state).unwrap();

        let mask = ctx
            .attention_mask(&state, AttentionMode::FullCausal, true)
            .unwrap();

        assert_eq!(mask.n_tokens, 2);
        assert_eq!(mask.n_kv, 256);
        assert_eq!(mask.values[0], 0.0);
        assert!(mask.values[1].is_infinite() && mask.values[1].is_sign_negative());
        assert_eq!(mask.values[256], 0.0);
        assert_eq!(mask.values[257], 0.0);
        assert!(mask.values[255].is_infinite() && mask.values[255].is_sign_negative());
        assert!(mask.values[511].is_infinite() && mask.values[511].is_sign_negative());
    }

    #[test]
    fn kv_attention_mask_applies_sliding_window_swa() {
        let mut state = KvCacheState::new_with_swa(512, 4, true, 2, KvSwaType::Standard);
        let mut ctx = state
            .prepare_context(vec![KvUbatch::single_seq(0, &[8, 11, 12])], true)
            .unwrap();
        ctx.apply(&mut state).unwrap();

        let mask = ctx
            .attention_mask(&state, AttentionMode::SlidingWindow { window: 2 }, true)
            .unwrap();

        assert_eq!(mask.n_tokens, 3);
        assert_eq!(mask.values[0], 0.0);
        assert!(mask.values[1].is_infinite() && mask.values[1].is_sign_negative());
        assert!(mask.values[2].is_infinite() && mask.values[2].is_sign_negative());

        let row1 = 256usize;
        assert!(mask.values[row1].is_infinite() && mask.values[row1].is_sign_negative());
        assert_eq!(mask.values[row1 + 1], 0.0);
        assert!(mask.values[row1 + 2].is_infinite() && mask.values[row1 + 2].is_sign_negative());

        let row2 = 512usize;
        assert!(mask.values[row2].is_infinite() && mask.values[row2].is_sign_negative());
        assert_eq!(mask.values[row2 + 1], 0.0);
        assert_eq!(mask.values[row2 + 2], 0.0);
    }

    #[test]
    fn kv_batch_attention_plan_tracks_full_and_sliding_layers() {
        let mut state = KvCacheState::new_with_swa(512, 4, true, 2, KvSwaType::Standard);
        let layout = KvCacheLayout {
            layers: vec![
                LayerKvSpec {
                    layer_idx: 0,
                    kv_heads: 2,
                    head_dim: 2,
                    attention: AttentionMode::FullCausal,
                    sharing: KvSharing::Dedicated,
                },
                LayerKvSpec {
                    layer_idx: 1,
                    kv_heads: 2,
                    head_dim: 2,
                    attention: AttentionMode::SlidingWindow { window: 2 },
                    sharing: KvSharing::Dedicated,
                },
            ],
        };
        let mut ctx = state
            .prepare_context(vec![KvUbatch::single_seq(0, &[8, 11, 12])], true)
            .unwrap();
        ctx.apply(&mut state).unwrap();

        let batch = ctx.batch_attention_plan(&state, &layout, true).unwrap();
        assert_eq!(batch.n_kv, 256);
        assert_eq!(batch.layers.len(), 2);
        assert_eq!(batch.layers[0].layer_idx, 0);
        assert_eq!(batch.layers[1].layer_idx, 1);

        let full_finite = batch.layers[0].mask.values.iter().filter(|v| v.is_finite()).count();
        let swa_finite = batch.layers[1].mask.values.iter().filter(|v| v.is_finite()).count();
        assert!(full_finite > swa_finite);
    }

    #[test]
    fn kv_layer_input_copy_plan_rejects_wrong_width() {
        let plan = KvLayerExecutionPlan {
            layer_idx: 0,
            n_kv: 256,
            cache_view: KvLayerCacheView {
                layer_idx: 0,
                storage_idx: 0,
                n_kv: 256,
                stream_start: 0,
                stream_count: 1,
                v_trans: false,
                key: KvTensorViewDescriptor {
                    byte_offset: 0,
                    scalar_type: KvScalarType::F16,
                    dims: [128, 8, 256, 1],
                    strides: [2, 256, 2048, 4096],
                },
                value: KvTensorViewDescriptor {
                    byte_offset: 4096,
                    scalar_type: KvScalarType::F16,
                    dims: [128, 8, 256, 1],
                    strides: [2, 256, 2048, 4096],
                },
            },
            copy_plan: KvLayerCopyPlan {
                layer_idx: 0,
                storage_idx: 0,
                v_trans: false,
                key: KvCopyTargetDescriptor {
                    byte_offset: 0,
                    scalar_type: KvScalarType::F16,
                    row_width_el: 1024,
                    row_count: 8192,
                    row_stride_bytes: 2048,
                },
                value: KvCopyTargetDescriptor {
                    byte_offset: 4096,
                    scalar_type: KvScalarType::F16,
                    row_width_el: 1024,
                    row_count: 8192,
                    row_stride_bytes: 2048,
                },
            },
            k_indices: vec![0, 1],
            v_indices: vec![0, 1],
        };

        let err = plan.input_copy_plan(512, 1024).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::KvInputWidthMismatch {
                expected: 1024,
                actual: 512,
            }
        );
    }

    #[test]
    fn kv_batch_jobs_reject_missing_or_wrong_layer_inputs() {
        let mut state = KvCacheState::new_with_policy(128, 4, true, 32, 0, KvSwaType::None);
        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 2,
                head_dim: 2,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };
        let arena = KvArenaLayout::for_layout(&layout, 128, state.n_stream(), KvScalarType::F16).unwrap();
        let mut ctx = state
            .prepare_context(vec![KvUbatch::single_seq(0, &[10, 11])], true)
            .unwrap();
        ctx.apply(&mut state).unwrap();

        let batch = ctx.batch_execution_plan(&arena, &layout, true).unwrap();
        let missing = batch.jobs_u16(&layout, &[]).unwrap_err();
        assert_eq!(missing, RuntimeError::MissingKvLayerInput { layer_idx: 0 });

        let wrong = batch
            .jobs_u16(
                &layout,
                &[KvLayerInputU16 {
                    layer_idx: 0,
                    key_src: vec![1_u16; 7],
                    value_src: vec![2_u16; 8],
                }],
            )
            .unwrap_err();
        assert_eq!(
            wrong,
            RuntimeError::KvInputBufferLengthMismatch {
                layer_idx: 0,
                tensor: "key",
                expected: 8,
                actual: 7,
            }
        );
    }

    #[test]
    fn commit_session_kv_write_advances_session_and_marks_decode_ready() {
        let mut runtime = runtime(4096, 1024, 256);
        let slot = runtime.open_session(512).unwrap();
        runtime.ingest_tokens(slot, 32).unwrap();

        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 8,
                head_dim: 128,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };

        let plan = runtime.plan_session_kv_write(slot, 16, &layout).unwrap();
        let view = runtime.commit_session_kv_write(&plan).unwrap();
        assert_eq!(view.used_ctx, 48);
        assert!(view.decode_ready);

        let scheduled = runtime.schedule_decode(1).unwrap();
        assert_eq!(scheduled.slots, vec![slot]);
    }

    #[test]
    fn stale_kv_write_plan_is_rejected() {
        let mut runtime = runtime(4096, 1024, 256);
        let slot = runtime.open_session(512).unwrap();
        runtime.ingest_tokens(slot, 32).unwrap();

        let layout = KvCacheLayout {
            layers: vec![LayerKvSpec {
                layer_idx: 0,
                kv_heads: 8,
                head_dim: 128,
                attention: AttentionMode::FullCausal,
                sharing: KvSharing::Dedicated,
            }],
        };

        let plan = runtime.plan_session_kv_write(slot, 16, &layout).unwrap();
        runtime.ingest_tokens(slot, 8).unwrap();

        let err = runtime.commit_session_kv_write(&plan).unwrap_err();
        assert_eq!(
            err,
            RuntimeError::StaleKvWritePlan {
                slot,
                expected_start_pos: 40,
                plan_start_pos: 32,
            }
        );
    }
}
