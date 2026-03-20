use std::fs::File;
use std::io::{self, BufReader, Read};

/*
 * IDX File Format Reference:
 * [offset] [type]          [value]          [description]
 * 0000     32 bit integer  0x00000801(2049) magic number (1D - labels)
 * 0000     32 bit integer  0x00000803(2051) magic number (3D - images)
 * 0004     32 bit integer  60000            number of items
 * 0008     32 bit integer  28               number of rows (images only)
 * 0012     32 bit integer  28               number of columns (images only)
 * nnnn     unsigned byte   ??               pixel/label data
 */

#[derive(Debug)]
pub struct MnistImage {
    pub data: Vec<u8>,
    pub label: u8,
}

pub struct MnistDataset {
    pub images: Vec<MnistImage>,
}

impl MnistDataset {
    pub fn load(image_path: &str, label_path: &str) -> io::Result<Self> {
        let images_idx = IdxData::read(image_path)?;
        let labels_idx = IdxData::read(label_path)?;

        // Validation
        if images_idx.magic != 2051 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid image magic number",
            ));
        }
        if labels_idx.magic != 2049 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid label magic number",
            ));
        }

        let count = images_idx.sizes[0];
        let rows = images_idx.sizes[1];
        let cols = images_idx.sizes[2];

        if count != labels_idx.sizes[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Mismatched image/label counts",
            ));
        }

        let image_size = (rows * cols) as usize;

        // Map data into MnistImage structs
        let images = images_idx
            .data
            .chunks_exact(image_size)
            .take(count as usize)
            .zip(labels_idx.data.iter())
            .map(|(chunk, &label)| MnistImage {
                data: chunk.to_vec(),
                label,
            })
            .collect();

        Ok(MnistDataset { images })
    }
}

struct IdxData {
    magic: u32,
    sizes: Vec<u32>,
    data: Vec<u8>,
}

impl IdxData {
    fn read(path: &str) -> io::Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        let magic = read_u32(&mut reader)?;
        // Last byte of magic number defines the number of dimensions
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

#[inline]
fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}
