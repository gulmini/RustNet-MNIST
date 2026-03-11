use ndarray::{Array2, Axis};

/// Computes Softmax Cross-Entropy Loss and the gradient w.r.t the pre-softmax logits.
pub fn cross_entropy_with_logits(
    logits: &Array2<f64>,
    targets: &Array2<f64>,
) -> (f64, Array2<f64>) {
    let batch_len = logits.nrows() as f64;

    let max_per_row = logits.map_axis(Axis(1), |row| {
        row.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    });

    let shifted = logits - &max_per_row.insert_axis(Axis(1));
    let exp_scores = shifted.mapv(|x| x.exp());
    let sum_exp = exp_scores.sum_axis(Axis(1));
    let probabilities = exp_scores / &sum_exp.insert_axis(Axis(1));

    let epsilon = 1e-8;
    let loss = -(targets * probabilities.mapv(|p| (p + epsilon).ln())).sum() / batch_len;

    // Gradient of loss w.r.t pre-activation logits
    let grad = (&probabilities - targets) / batch_len;

    (loss, grad)
}
