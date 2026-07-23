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
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("Safetensors: {0}")]
    Format(#[from] safetensors::SafeTensorError),
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

        std::fs::remove_file(path).unwrap();
    }
}
