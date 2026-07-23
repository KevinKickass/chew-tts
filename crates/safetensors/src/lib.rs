use memmap2::{Mmap, MmapOptions};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: usize,
}

pub struct MappedSafetensors {
    path: PathBuf,
    mmap: Mmap,
}

impl MappedSafetensors {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only and owns the file-backed pages for
        // the lifetime of this object. Model files must not be modified while
        // loaded, which is also required for GPU weight uploads to be sound.
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        SafeTensors::deserialize(&mmap)?;
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensor_infos(&self) -> Result<Vec<TensorInfo>, Error> {
        let tensors = SafeTensors::deserialize(&self.mmap)?;
        let mut infos = tensors
            .names()
            .into_iter()
            .map(|name| {
                let tensor = tensors.tensor(name)?;
                Ok(TensorInfo {
                    name: name.to_string(),
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    bytes: tensor.data().len(),
                })
            })
            .collect::<Result<Vec<_>, safetensors::SafeTensorError>>()?;
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(infos)
    }

    pub fn with_tensor<R>(
        &self,
        name: &str,
        use_tensor: impl FnOnce(TensorView<'_>) -> R,
    ) -> Result<R, Error> {
        let tensors = SafeTensors::deserialize(&self.mmap)?;
        Ok(use_tensor(tensors.tensor(name)?))
    }

    pub fn tensor_f16(&self, name: &str) -> Result<(Vec<usize>, Vec<half::f16>), Error> {
        self.with_tensor(name, |tensor| {
            let shape = tensor.shape().to_vec();
            let values = match tensor.dtype() {
                Dtype::F16 => tensor
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])))
                    .collect(),
                Dtype::BF16 => tensor
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| {
                        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                        half::f16::from_f32(f32::from_bits(u32::from(bits) << 16))
                    })
                    .collect(),
                Dtype::F32 => tensor
                    .data()
                    .chunks_exact(4)
                    .map(|bytes| {
                        half::f16::from_f32(f32::from_le_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3],
                        ]))
                    })
                    .collect(),
                dtype => return Err(Error::UnsupportedDtype(dtype)),
            };
            Ok((shape, values))
        })?
    }

    pub fn tensor_f32(&self, name: &str) -> Result<(Vec<usize>, Vec<f32>), Error> {
        self.with_tensor(name, |tensor| {
            let shape = tensor.shape().to_vec();
            let values = match tensor.dtype() {
                Dtype::F16 => tensor
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| {
                        half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
                    })
                    .collect(),
                Dtype::BF16 => tensor
                    .data()
                    .chunks_exact(2)
                    .map(|bytes| {
                        f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
                    })
                    .collect(),
                Dtype::F32 => tensor
                    .data()
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                    .collect(),
                dtype => return Err(Error::UnsupportedDtype(dtype)),
            };
            Ok((shape, values))
        })?
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensors: {0}")]
    Format(#[from] safetensors::SafeTensorError),
    #[error("tensor dtype {0:?} cannot be converted to f16")]
    UnsupportedDtype(Dtype),
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{TensorView, serialize_to_file};
    use std::collections::HashMap;

    #[test]
    fn maps_and_reports_tensor_metadata() {
        let path = std::env::temp_dir().join(format!(
            "chew-safetensors-{}-{}.safetensors",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let data = [0u8; 8];
        let view = TensorView::new(Dtype::F32, vec![2], &data).unwrap();
        let tensors = HashMap::from([("weight", view)]);
        serialize_to_file(tensors, None, &path).unwrap();

        let mapped = MappedSafetensors::open(&path).unwrap();
        let infos = mapped.tensor_infos().unwrap();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].name, "weight");
        assert_eq!(infos[0].shape, vec![2]);
        assert_eq!(infos[0].bytes, 8);
        let (shape, values) = mapped.tensor_f16("weight").unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(values, vec![half::f16::ZERO; 2]);
        let (shape, values) = mapped.tensor_f32("weight").unwrap();
        assert_eq!(shape, vec![2]);
        assert_eq!(values, vec![0.0; 2]);

        std::fs::remove_file(path).unwrap();
    }
}
