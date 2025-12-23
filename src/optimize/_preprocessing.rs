use crate::{matrix::Matrix};

pub fn normalize(data: &Matrix) -> Matrix {
    if data.shape.0 == 0 {
        return data.clone();
    }

    let mut result_matrix: Matrix = Matrix::zeros(data.shape.0, data.shape.1);
    let mut min: f64 = data.real[0][0]; 
    let mut max: f64 = data.real[0][0]; 
    for r in 0..data.shape.0 {
        for c in 0..data.shape.1 {
            if data.real[r][c] > max {max = data.real[r][c]}
            else if data.real[r][c] < min {min = data.real[r][c]}
        }
    }
    
    for r in 0..data.shape.0 {
        for c in 0..data.shape.1 {
            result_matrix.real[r][c] = (data.real[r][c] - max) / (max - min);
        }
    }

    result_matrix
}

/// ## NEED TO FIX
pub fn principle_component_analysis(matrix: &Matrix, dimension: usize) -> Result<Matrix, String> {
    if dimension > matrix.shape.0 {
        return Err("Input Error: Parameter dimension cannot be greater than matrix's row".to_string());
    }
    
    let mut row_mean: Matrix = Matrix::zeros(matrix.shape.0, 1);
    for r in 0..matrix.shape.0 {
        let mut sum: f64 = 0.0;
        for c in 0..matrix.shape.1 {
            sum += matrix.real[r][c];
        }
        row_mean.real[r][0] = sum / matrix.shape.1 as f64;
    }
    let mean_matrix: Matrix = &row_mean * &Matrix::ones(1, matrix.shape.1);
    let residual_matrix: Matrix = matrix - &mean_matrix;
    let (u, _, _) = residual_matrix.singular_value_decomposition()?;
    let mut principle_component: Matrix = Matrix::zeros(0, 0);
    for d in 0..dimension {
        if d == u.shape.1 {break;}
        principle_component = principle_component.append(&u.get_column_vector(d)?.transpose(), 0)?;
    }

    let result_matrix: Matrix = &principle_component * &mean_matrix;
    Ok(result_matrix)
}