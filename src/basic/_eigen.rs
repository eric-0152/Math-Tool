use crate::matrix::Matrix;
use crate::solve;

pub fn rayleigh_quotient(eigenvector: &Matrix, matrix_a: &Matrix) -> f64 {
    let numerator: f64 = (&eigenvector.transpose() * &(matrix_a * eigenvector)).real[0][0];
    let denominator: f64 = (&eigenvector.transpose() * eigenvector).real[0][0];

    numerator / denominator
}

/// Return a similar matrix.
///
/// ### Algorithm :
/// ***A = QR*** => let ***S = RQ*** => ***IS = Q^(-1)QRQ*** => ***S = Q^(-1)AQ***
pub fn similar_matrix(matrix: &Matrix) -> Result<Matrix, String> {
    if matrix.shape.0 != matrix.shape.1 {
        return Err("Input Error: The matrix is not square.".to_string());
    }
    
    match matrix.qr_decomposition() {
        Ok(tuple) => {
            Ok(&tuple.1 * &tuple.0)
        }
        Err(error_msg) => Err(error_msg),
    }
}

fn no_check_similar_matrix(matrix: &Matrix) -> Matrix {
    let tuple = matrix.qr_decomposition().unwrap();

    &tuple.1 * &tuple.0
}

pub fn shift_qr_algorithm(
    matrix: &Matrix,
    max_iter: u32,
    error_thershold: f64,
) -> Result<(Matrix, f64), String> {
    match similar_matrix(matrix) {
        Ok(mut matrix_similar) => {
            let last_row_idx: usize = matrix_similar.shape.0 - 1;
            let last_col_idx: usize = matrix_similar.shape.1 - 1;
            let matrix_size: usize = matrix_similar.shape.0;
            let mut last_eigenvalue: f64 = matrix_similar.real[last_row_idx][last_col_idx];
            let mut difference: f64 = 1.0;
            let mut step: u32 = 0;
            while difference > error_thershold && step < max_iter {
                let shift: Matrix = &Matrix::identity(matrix_size) * &last_eigenvalue;
                matrix_similar = &matrix_similar - &shift;
                matrix_similar = &no_check_similar_matrix(&matrix_similar) + &shift;
                difference = (matrix_similar.real[last_row_idx][last_col_idx] - last_eigenvalue).abs();
                last_eigenvalue = matrix_similar.real[last_row_idx][last_col_idx];
                step += 1;
            }

            Ok((matrix_similar, difference))
        }

        Err(error_msg) => Err(error_msg),
    }
}

/// ## NEED TO FIX
/// ## Can not return complex eigenvalue
/// Return a vector which contains eigenvalue of matrix and the difference.
///
/// Check the difference, if it's too large, the eigenvalue may contains complex number.
pub fn eigenvalue(
    matrix: &Matrix,
    max_iter: u32,
    error_thershold: f64,
) -> Result<(Matrix, f64), String> {
    match similar_matrix(matrix) {
        Err(error_msg) => Err(error_msg),
        Ok(_) => match shift_qr_algorithm(matrix, max_iter, error_thershold) {
            Err(error_msg) => Err(error_msg),
            Ok(tuple) => {
                let matrix_similar: Matrix = tuple.0;
                let difference: f64 = tuple.1;
                let mut eigenvalue: Vec<f64> = Vec::new();
                for d in 0..matrix_similar.shape.0 {
                    eigenvalue.push(matrix_similar.real[d][d]);
                }

                Ok((Matrix::from_vec(&eigenvalue), difference))
            }
        }
    }
}

/// ## Can not return complex eigenvector
pub fn eigenvector(
    matrix: &Matrix,
    eigen_value: f64,
) -> Result<Matrix, String> {
    if matrix.shape.0 != matrix.shape.1 {
        return Err("Input Error: The input matrix is not square.".to_string());
    }

    let eigen_kernel: Matrix = &(&Matrix::identity(matrix.shape.0) * &eigen_value) - &matrix;
    Ok(solve::null_space(&eigen_kernel))
}
