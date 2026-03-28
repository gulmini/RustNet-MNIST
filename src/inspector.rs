use crate::layer::activation_layer::ActivationLayer;
use crate::layer::dense_layer::DenseLayer;
use plotters::prelude::*;

pub trait LayerInspector {
    fn inspect_dense(
        &self,
        _index: usize,
        _layer: &DenseLayer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
    fn inspect_activation(
        &self,
        _index: usize,
        _layer: &ActivationLayer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

pub struct WeightDistributionInspector;

impl LayerInspector for WeightDistributionInspector {
    fn inspect_dense(
        &self,
        index: usize,
        layer: &DenseLayer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // let min = *layer.weights.min()?;
        // let max = *layer.weights.max()?;

        let data: Vec<f32> = layer.weights.iter().copied().collect();

        let filename = format!("visualizations/layer_{}_weight_distribution.svg", index);
        let drawing_area = SVGBackend::new(&filename, (1080, 768)).into_drawing_area();
        drawing_area.fill(&WHITE).unwrap();

        let mut chart_builder = ChartBuilder::on(&drawing_area);
        chart_builder
            .margin(5)
            .set_left_and_bottom_label_area_size(50);

        let mut chart_context = chart_builder
            .build_cartesian_2d((-0.5..0.5f32).step(0.01).use_round(), 0f32..0.1f32)
            .unwrap();
        chart_context.configure_mesh().draw().unwrap();

        let unit: f32 = 1f32 / data.len() as f32;
        chart_context
            .draw_series(
                Histogram::vertical(&chart_context)
                    .style(BLUE.filled())
                    .margin(1)
                    .data(data.into_iter().map(|x| (x, unit))),
            )
            .unwrap();

        println!("Exported weight distribution to {}", filename);
        Ok(())
    }
}

pub struct HeatMapInspector;

impl LayerInspector for HeatMapInspector {
    fn inspect_dense(
        &self,
        index: usize,
        layer: &DenseLayer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if index != 0 || layer.weights.nrows() != 784 {
            return Ok(());
        }

        let mut norms = Vec::with_capacity(784);
        for row in layer.weights.rows() {
            let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            norms.push(norm);
        }

        let max_norm = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_norm = norms.iter().cloned().fold(f32::INFINITY, f32::min);
        let range = (max_norm - min_norm).max(1e-8);

        let filename = format!("visualizations/layer_{}_heatmap.svg", index);
        let drawing_area = SVGBackend::new(&filename, (600, 600)).into_drawing_area();
        drawing_area.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&drawing_area)
            .margin(20)
            .build_cartesian_2d(0usize..28usize, 28usize..0usize)
            .unwrap();

        chart
            .draw_series(norms.iter().enumerate().map(|(i, &norm)| {
                let x = i % 28;
                let y = i / 28;

                let normalized = (norm - min_norm) / range;

                let hue = (1.0 - normalized) * (2.0 / 3.0);
                let color = HSLColor(hue as f64, 1.0, 0.5);

                Rectangle::new([(x, y), (x + 1, y + 1)], color.filled())
            }))
            .unwrap();

        println!("Exported heatmap to {}", filename);
        Ok(())
    }
}
