use std::collections::HashMap;
use std::fmt;

/// GGUF magic: "GGUF" as bytes [0x47, 0x47, 0x55, 0x46] read as LE u32
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Supported GGUF versions
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            MetadataValue::Uint16(v) => Some(*v as u32),
            MetadataValue::Uint8(v) => Some(*v as u32),
            MetadataValue::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::Uint64(v) => Some(*v),
            MetadataValue::Uint32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            MetadataValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Tensor quantization type — determines how weights are stored and dequantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    // 4, 5 deprecated
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            _ => None,
        }
    }

    /// Block size (number of weights per quantization block).
    /// All K-quants and IQ types use QK_K = 256.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::IQ4_NL => 32,
            // All K-quants and IQ types: QK_K = 256
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::Q8_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ4_XS
            | Self::IQ1_S
            | Self::IQ1_M => 256,
        }
    }

    /// Bytes per block of `block_size()` weights.
    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 2 + 16,                // f16 scale + 32 * 4bit / 8
            Self::Q4_1 => 2 + 2 + 16,            // f16 scale + f16 min + 16 bytes
            Self::Q5_0 => 2 + 4 + 16,            // f16 scale + 32bit hmask + 16 bytes
            Self::Q5_1 => 2 + 2 + 4 + 16,        // f16 scale + f16 min + hmask + 16 bytes
            Self::Q8_0 => 2 + 32,                // f16 scale + 32 bytes
            Self::Q8_1 => 4 + 4 + 32,            // f32 scale + f32 sum + 32 bytes
            Self::Q2_K => 2 + 2 + 16 + 64,       // f16 d + f16 dmin + 16 scales + 64 qs
            Self::Q3_K => 2 + 32 + 64 + 12,      // f16 d + 32 hmask + 64 qs + 12 scales
            Self::Q4_K => 2 + 2 + 12 + 128,      // f16 d + f16 dmin + 12 scales + 128 qs
            Self::Q5_K => 2 + 2 + 12 + 32 + 128, // + 32 qh
            Self::Q6_K => 2 + 128 + 64 + 16,     // f16 d + 128 ql + 64 qh + 16 scales
            Self::Q8_K => 4 + 256,               // f32 d + 256 qs
            Self::IQ1_S => 2 + 32 + 16,          // f16 d + 32 qs + 16 qh (8 * u16)
            Self::IQ1_M => 2 + 32 + 16 + 8,      // f16 d + 32 qs + 16 qh + 8 scales
            Self::IQ2_XXS => 2 + 64,             // f16 d + 32 * u16 qs
            Self::IQ2_XS => 2 + 64 + 8,          // f16 d + 64 qs + 8 scales
            Self::IQ2_S => 2 + 64 + 8 + 32,      // f16 d + 64 qs + 8 qh + 32 scales  -- approx
            Self::IQ3_XXS => 2 + 96 + 16,        // f16 d + 3*32 qs + 16 signs
            Self::IQ3_S => 2 + 96 + 32 + 8,      // f16 d + 96 qs + 32 qh + 8 scales
            Self::IQ4_NL => 2 + 16,              // f16 d + 16 qs (32 4-bit)
            Self::IQ4_XS => 2 + 2 + 128 + 16, // f16 d + u16 scales_h + 128 qs + 16 scales_l -- approx
        }
    }

    /// Bits per weight (for reporting/display).
    pub fn bits_per_weight(&self) -> f32 {
        (self.block_bytes() as f32 * 8.0) / self.block_size() as f32
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Information about one tensor in the GGUF file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g. "blk.0.attn_q.weight")
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Shape — stored as [cols, rows, ...] in GGUF (row-major for the outer dims)
    pub shape: Vec<u64>,
    /// Quantization type
    pub ggml_type: GgmlType,
    /// Byte offset from start of tensor data section
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }

    /// Size in bytes on disk (quantized).
    pub fn data_size(&self) -> u64 {
        let n = self.n_elements() as usize;
        let bs = self.ggml_type.block_size();
        let bb = self.ggml_type.block_bytes();
        let n_blocks = (n + bs - 1) / bs;
        (n_blocks * bb) as u64
    }
}

/// Parsed GGUF header + metadata.
#[derive(Debug)]
pub struct GgufHeader {
    pub version: u32,
    pub n_tensors: u64,
    pub metadata: HashMap<String, MetadataValue>,
}

impl GgufHeader {
    pub fn model_arch(&self) -> Option<&str> {
        self.architecture()
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.metadata
            .get("tokenizer.chat_template")
            .and_then(|v| v.as_str())
    }

    pub fn token_id(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.token_id("tokenizer.ggml.bos_token_id")
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.token_id("tokenizer.ggml.eos_token_id")
    }

    pub fn add_bos_token(&self) -> Option<bool> {
        self.metadata
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
    }

    pub fn add_eos_token(&self) -> Option<bool> {
        self.metadata
            .get("tokenizer.ggml.add_eos_token")
            .and_then(|v| v.as_bool())
    }

    pub fn preferred_eos_token_id(&self) -> Option<u32> {
        self.token_id("tokenizer.ggml.eot_token_id")
            .or_else(|| self.token_id("tokenizer.ggml.eos_token_id"))
    }

    pub fn context_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.as_u32())
    }

    pub fn block_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.block_count"))
            .and_then(|v| v.as_u32())
    }

    pub fn embedding_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.embedding_length"))
            .and_then(|v| v.as_u32())
    }

    pub fn head_count(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.attention.head_count"))
            .and_then(|v| v.as_u32())
    }

    pub fn head_count_kv(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.attention.head_count_kv"))
            .and_then(|v| v.as_u32())
    }

    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.as_f32())
    }

    pub fn rms_norm_eps(&self) -> Option<f32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
    }

    pub fn vocab_size(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.vocab_size"))
            .and_then(|v| v.as_u32())
    }

    pub fn feed_forward_length(&self) -> Option<u32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{arch}.feed_forward_length"))
            .and_then(|v| v.as_u32())
    }

    /// Generic get for any key as u32.
    pub fn get_u32(&self, key: &str) -> Result<u32, GgufError> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GgufError::TensorNotFound(key.into()))
    }

    /// Generic get for any key as f32.
    pub fn get_f32(&self, key: &str) -> Result<f32, GgufError> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| GgufError::TensorNotFound(key.into()))
    }

    /// Generic get for any key as u32 array.
    pub fn get_u32_array(&self, key: &str) -> Result<Vec<u32>, GgufError> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => Ok(arr.iter().filter_map(|v| v.as_u32()).collect()),
            _ => Err(GgufError::TensorNotFound(key.into())),
        }
    }

    /// Generic get for any key as bool array.
    pub fn get_bool_array(&self, key: &str) -> Result<Vec<bool>, GgufError> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => Ok(arr
                .iter()
                .map(|v| matches!(v, MetadataValue::Bool(true)))
                .collect()),
            _ => Err(GgufError::TensorNotFound(key.into())),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("invalid GGUF magic: expected 0x{GGUF_MAGIC:08X}, got 0x{0:08X}")]
    BadMagic(u32),
    #[error("unsupported GGUF version: {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),
    #[error("unknown ggml type: {0}")]
    UnknownType(u32),
    #[error("unknown metadata value type: {0}")]
    UnknownMetadataType(u32),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    UnsupportedArchitecture(String),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("file too small for tensor data (need {needed}, file has {file_size})")]
    FileTooSmall { needed: u64, file_size: u64 },
}
