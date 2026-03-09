use colored::Colorize;

pub fn visualize_image(image: &[u8], rows: usize, cols: usize) {
    for row in 0..rows {
        for col in 0..cols {
            let pixel = image[row * cols + col];
            print!("{}", "██".truecolor(pixel, pixel, pixel));
        }
        println!();
    }
}
