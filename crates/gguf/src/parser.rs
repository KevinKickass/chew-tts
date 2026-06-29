use crate::types::*;
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;
use tracing::info;

/// A parsed GGUF file backed by mmap.
///
/// The file stays memory-mapped for zero-copy access to tensor data.
/// Call `tensor_data()` to get a slice pointing directly into the file.
pub struct GgufFile {
    pub header: GgufHeader,
    pub tensors: Vec<TensorInfo>,
    /// Byte offset where tensor data begins in the file
    data_offset: u64,
    /// Memory-mapped file
    mmap: Mmap,
}

impl GgufFile {
    /// Open and parse a GGUF file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap[..]);

        // Header
        let magic = cursor.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }

        let version = cursor.read_u32::<LittleEndian>()?;
        if version != GGUF_VERSION_2 && version != GGUF_VERSION_3 {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let n_tensors = cursor.read_u64::<LittleEndian>()?;
        let n_kv = cursor.read_u64::<LittleEndian>()?;

        info!(
            version,
            n_tensors,
            n_kv,
            file = %path.display(),
            "parsing GGUF"
        );

        // Metadata KV pairs
        let mut metadata = HashMap::with_capacity(n_kv as usize);
        for _ in 0..n_kv {
            let key = read_string(&mut cursor)?;
            let value = read_metadata_value(&mut cursor)?;
            metadata.insert(key, value);
        }

        let header = GgufHeader {
            version,
            n_tensors,
            metadata,
        };

        // Tensor infos
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(&mut cursor)?;
            let n_dims = cursor.read_u32::<LittleEndian>()?;
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                shape.push(cursor.read_u64::<LittleEndian>()?);
            }
            let type_id = cursor.read_u32::<LittleEndian>()?;
            let ggml_type = GgmlType::from_u32(type_id).ok_or(GgufError::UnknownType(type_id))?;
            let offset = cursor.read_u64::<LittleEndian>()?;

            tensors.push(TensorInfo {
                name,
                n_dims,
                shape,
                ggml_type,
                offset,
            });
        }

        // Tensor data starts at the next alignment boundary (32 bytes) after metadata+tensor info
        let header_end = cursor.position();
        let alignment = 32u64;
        let data_offset = (header_end + alignment - 1) / alignment * alignment;

        // Verify file is large enough
        let file_size = mmap.len() as u64;
        if let Some(last) = tensors.last() {
            let needed = data_offset + last.offset + last.data_size();
            if needed > file_size {
                return Err(GgufError::FileTooSmall { needed, file_size });
            }
        }

        let total_data_bytes: u64 = tensors.iter().map(|t| t.data_size()).sum();

        info!(
            arch = header.architecture().unwrap_or("unknown"),
            model = header.model_name().unwrap_or("unknown"),
            n_tensors,
            data_mb = total_data_bytes / (1024 * 1024),
            "GGUF parsed"
        );

        Ok(Self {
            header,
            tensors,
            data_offset,
            mmap,
        })
    }

    /// Get raw quantized bytes for a tensor — zero-copy slice into the mmap.
    pub fn tensor_data(&self, tensor: &TensorInfo) -> Result<&[u8], GgufError> {
        let start = (self.data_offset + tensor.offset) as usize;
        let size = tensor.data_size() as usize;
        let end = start + size;
        if end > self.mmap.len() {
            return Err(GgufError::FileTooSmall {
                needed: end as u64,
                file_size: self.mmap.len() as u64,
            });
        }
        Ok(&self.mmap[start..end])
    }

    /// Find a tensor by name.
    pub fn find_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Get tensor data by name.
    pub fn tensor_data_by_name(&self, name: &str) -> Result<(&TensorInfo, &[u8]), GgufError> {
        let tensor = self
            .find_tensor(name)
            .ok_or_else(|| GgufError::TensorNotFound(name.to_string()))?;
        let data = self.tensor_data(tensor)?;
        Ok((tensor, data))
    }

    /// Total size of all tensor data in bytes.
    pub fn total_data_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.data_size()).sum()
    }

    /// Print a summary of all tensors (for debugging).
    pub fn dump_tensors(&self) {
        for t in &self.tensors {
            let shape_str: Vec<String> = t.shape.iter().map(|s| s.to_string()).collect();
            println!(
                "  {:60} {:8} [{:>20}] {:>8.2} MB",
                t.name,
                format!("{}", t.ggml_type),
                shape_str.join(", "),
                t.data_size() as f64 / (1024.0 * 1024.0),
            );
        }
    }
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String, GgufError> {
    let len = cursor.read_u64::<LittleEndian>()? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_metadata_value(cursor: &mut Cursor<&[u8]>) -> Result<MetadataValue, GgufError> {
    let value_type = cursor.read_u32::<LittleEndian>()?;
    match value_type {
        0 => Ok(MetadataValue::Uint8(cursor.read_u8()?)),
        1 => Ok(MetadataValue::Int8(cursor.read_i8()?)),
        2 => Ok(MetadataValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => {
            let s = read_string(cursor)?;
            Ok(MetadataValue::String(s))
        }
        9 => {
            // Array: element type + count + elements
            let elem_type = cursor.read_u32::<LittleEndian>()?;
            let count = cursor.read_u64::<LittleEndian>()? as usize;
            let mut arr = Vec::with_capacity(count.min(1024)); // cap initial alloc
            for _ in 0..count {
                let val = read_typed_value(cursor, elem_type)?;
                arr.push(val);
            }
            Ok(MetadataValue::Array(arr))
        }
        10 => Ok(MetadataValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::Float64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(GgufError::UnknownMetadataType(value_type)),
    }
}

fn read_typed_value(cursor: &mut Cursor<&[u8]>, type_id: u32) -> Result<MetadataValue, GgufError> {
    match type_id {
        0 => Ok(MetadataValue::Uint8(cursor.read_u8()?)),
        1 => Ok(MetadataValue::Int8(cursor.read_i8()?)),
        2 => Ok(MetadataValue::Uint16(cursor.read_u16::<LittleEndian>()?)),
        3 => Ok(MetadataValue::Int16(cursor.read_i16::<LittleEndian>()?)),
        4 => Ok(MetadataValue::Uint32(cursor.read_u32::<LittleEndian>()?)),
        5 => Ok(MetadataValue::Int32(cursor.read_i32::<LittleEndian>()?)),
        6 => Ok(MetadataValue::Float32(cursor.read_f32::<LittleEndian>()?)),
        7 => Ok(MetadataValue::Bool(cursor.read_u8()? != 0)),
        8 => {
            let s = read_string(cursor)?;
            Ok(MetadataValue::String(s))
        }
        10 => Ok(MetadataValue::Uint64(cursor.read_u64::<LittleEndian>()?)),
        11 => Ok(MetadataValue::Int64(cursor.read_i64::<LittleEndian>()?)),
        12 => Ok(MetadataValue::Float64(cursor.read_f64::<LittleEndian>()?)),
        _ => Err(GgufError::UnknownMetadataType(type_id)),
    }
}
