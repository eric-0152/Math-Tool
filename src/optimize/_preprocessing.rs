use crate::vector::Vector;

pub fn normalize(data: &Vector) -> Vector{
    if data.size == 0 {
        return data.clone();
    }

    let mut min: f64 = data.entries[0];
    let mut max: f64 = data.entries[0];
    for e in 0..data.size {
        if data.entries[e] < min {
            min = data.entries[e];
        } else if data.entries[e] > max {
            max = data.entries[e];
        }
    }
    data.display();
    println!("{max} {min}");
    let mut norm_data: Vector = data.clone();
    let denominator: f64 = max - min;
    for e in 0..norm_data.size {
        norm_data.entries[e] = (norm_data.entries[e] - min) / denominator;
    }

    norm_data
}