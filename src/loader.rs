use std::fs::File;
use std::io::{self, BufReader, Read};

#[derive(Debug)]
pub struct MnistImage {
    pub data: Vec<u8>,
    pub label: u8,
}

pub struct MnistDataset {
    pub images: Vec<MnistImage>,
    pub rows: u32,
    pub cols: u32,
}

impl MnistDataset {
    pub fn load(image_path: &str, label_path: &str) -> io::Result<Self> {
        let images_data = Self::read_idx_file(image_path)?;
        let labels_data = Self::read_idx_file(label_path)?;

        // Validate magic numbers (Image magic: 2051, Label magic: 2049)
        if images_data.magic != 2051 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid image file magic number",
            ));
        }
        if labels_data.magic != 2049 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid label file magic number",
            ));
        }

        // Parse dimensions
        let count = images_data.sizes[0];
        let rows = images_data.sizes[1];
        let cols = images_data.sizes[2];

        let label_count = labels_data.sizes[0];

        if count != label_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Image and label counts do not match",
            ));
        }

        let image_size = (rows * cols) as usize;
        let mut result = Vec::with_capacity(count as usize);

        // Chunk the raw bytes into individual images and zip with labels
        for (i, chunk) in images_data.data.chunks(image_size).enumerate() {
            if i >= count as usize {
                break;
            }

            result.push(MnistImage {
                data: chunk.to_vec(),
                label: labels_data.data[i],
            });
        }

        Ok(MnistDataset {
            images: result,
            rows,
            cols,
        })
    }

    // Generic helper to read IDX files
    fn read_idx_file(path: &str) -> io::Result<IdxData> {
        let f = File::open(path)?;
        let mut reader = BufReader::new(f);

        // 1. Read Magic Number
        let magic = read_u32(&mut reader)?;

        // 2. Parse dimensions based on magic number
        // The 3rd byte of magic number tells us how many dimensions (sizes) follow
        // e.g., 0x08 0x03 means data type 08 (unsigned byte) and 3 dimensions (N, Rows, Cols)
        let dim_count = (magic & 0xFF) as usize;

        let mut sizes = Vec::with_capacity(dim_count);
        for _ in 0..dim_count {
            sizes.push(read_u32(&mut reader)?);
        }

        // 3. Read the rest of the data
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;

        Ok(IdxData { magic, sizes, data })
    }
}

// Intermediate struct to hold raw file data
struct IdxData {
    magic: u32,
    sizes: Vec<u32>,
    data: Vec<u8>,
}

// Helper to read Big Endian u32
fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buffer = [0u8; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}
