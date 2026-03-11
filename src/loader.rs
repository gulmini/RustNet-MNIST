use ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{self, BufReader, Read};

pub struct MnistDataset {
    pub images: Array2<f64>,
    pub labels: Array1<usize>,
    pub rows: u32,
    pub cols: u32,
}

impl MnistDataset {
    pub fn load(image_path: &str, label_path: &str) -> io::Result<Self> {
        let images_data = Self::read_idx_file(image_path)?;
        let labels_data = Self::read_idx_file(label_path)?;

        if images_data.magic != 2051 || labels_data.magic != 2049 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid magic number",
            ));
        }

        let count = images_data.sizes[0] as usize;
        let rows = images_data.sizes[1];
        let cols = images_data.sizes[2];
        let image_size = (rows * cols) as usize;

        if count != labels_data.sizes[0] as usize {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Count mismatch"));
        }

        // Direct stream into flat contiguous memory. Eliminates 60k vector allocations.
        let f64_data: Vec<f64> = images_data.data.iter().map(|&b| b as f64 / 255.0).collect();
        let images = Array2::from_shape_vec((count, image_size), f64_data)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Image shape error"))?;

        let usize_labels: Vec<usize> = labels_data.data.iter().map(|&b| b as usize).collect();
        let labels = Array1::from_shape_vec(count, usize_labels)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Label shape error"))?;

        Ok(MnistDataset {
            images,
            labels,
            rows,
            cols,
        })
    }

    fn read_idx_file(path: &str) -> io::Result<IdxData> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);
        let magic = read_u32(&mut reader)?;
        let dim_count = (magic & 0xFF) as usize;

        let mut sizes = Vec::with_capacity(dim_count);
        for _ in 0..dim_count {
            sizes.push(read_u32(&mut reader)?);
        }

        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        Ok(IdxData { magic, sizes, data })
    }
}

struct IdxData {
    magic: u32,
    sizes: Vec<u32>,
    data: Vec<u8>,
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}
