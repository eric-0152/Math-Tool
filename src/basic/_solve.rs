use std::ptr::null;

use rand_distr::num_traits::ToPrimitive;

use crate::matrix::Matrix;
use crate::vector::Vector;

/// Given a upper triangular matrix ***A*** and vector ***b***, return a vector ***x***
/// such that ***Ax*** = ***b***.
pub fn upper_triangular(matrix: &Matrix, b: &Vector) -> Result<Vector, String> {
    if matrix.row != b.size {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    } else if !matrix.is_upper_triangular() {
        return Err("Input Error: The input matrix is not upper triangular.".to_string());
    }

    let mut vector_x: Vector = Vector::zeros(matrix.col);
    let min_range: usize = matrix.row.min(matrix.col);
    for diag in (0..min_range).rev() {
        vector_x.entries[diag] = b.entries[diag] / matrix.entries[diag][diag];
        for prev in ((diag + 1)..min_range).rev() {
            vector_x.entries[diag] -=
                matrix.entries[diag][prev] * vector_x.entries[prev] / matrix.entries[diag][diag];
        }
    }

    // Check consistency
    for e in 0..vector_x.size {
        if vector_x.entries[e].is_nan() {
            return Err("Value Error: The system is not consistent".to_string());
        }
    }

    Ok(vector_x)
}

/// Given a lower triangular matrix ***A*** and vector ***b***, return a vector ***x***
/// such that ***Ax*** = ***b***.
pub fn lower_triangular(matrix: &Matrix, b: &Vector) -> Result<Vector, String> {
    if matrix.row != b.size {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    } else if !matrix.is_lower_triangular() {
        return Err("Input Error: The input matrix is not lower triangular.".to_string());
    }

    let mut vector_x: Vector = Vector::zeros(matrix.col);
    let min_range = matrix.row.min(matrix.col);
    for diag in 0..min_range {
        vector_x.entries[diag] = b.entries[diag] / matrix.entries[diag][diag];
        for prev in 0..diag {
            vector_x.entries[diag] -=
                matrix.entries[diag][prev] * vector_x.entries[prev] / matrix.entries[diag][diag];
        }
    }

    // Check consistency
    for e in 0..vector_x.size {
        if vector_x.entries[e].is_nan() {
            return Err("Value Error: The system is not consistent".to_string());
        }
    }

    Ok(vector_x)
}

/// Return the tuple contains matrix, b and permutation after Gaussian Jordan elimination.
///
/// The algorithm will swap rows if needed (diagnal has 0), if the order of rows is
/// important, use swap_with_permutation() to yield the correct order.
pub fn gauss_jordan_elimination(
    matrix: &Matrix,
    b: &Vector,
) -> Result<(Matrix, Vector, Matrix), String> {
    if matrix.row != b.size {
        return Err("Input Error: The size of input matrix and vector b do not match.".to_string());
    }

    // Reduce to upper triangular form.
    let mut result_matrix: Matrix = matrix.clone();
    let mut result_vector: Vector = b.clone();
    let mut permutation: Matrix = Matrix::identity(matrix.row);
    let mut pivot_row: usize = 0;
    let mut pivot_col: usize = 0;
    let mut last_operate: i32 = 0;
    while pivot_row < result_matrix.row && pivot_col < result_matrix.col {
        // If the pivot is 0.0, swap to non zero.
        if result_matrix.entries[pivot_row][pivot_col] == 0.0 {
            let mut is_swap = false;
            for r in (pivot_row + 1)..result_matrix.row {
                if result_matrix.entries[r][pivot_col] != 0.0 {
                    result_matrix = result_matrix.swap_row(pivot_row, r).unwrap();
                    result_vector = result_vector.swap_element(pivot_row, r).unwrap();
                    permutation = permutation.swap_row(pivot_row, r).unwrap();
                    is_swap = true;
                    break;
                }
            }
            if !is_swap {
                last_operate = 0;
                pivot_col += 1;
                continue;
            }
        }

        for r in (pivot_row + 1)..result_matrix.row {
            let scale: f64 = result_matrix.entries[r][pivot_col] / result_matrix.entries[pivot_row][pivot_col];
            result_vector.entries[r] -= scale * result_vector.entries[pivot_row];
            for e in 0..matrix.col {
                result_matrix.entries[r][e] -= scale * result_matrix.entries[pivot_row][e];
            }
        }

        pivot_row += 1;
        pivot_col += 1;
        last_operate = 1;
    }

    // Reduce to diagonal form
    if last_operate == 0 {
        pivot_col -= 1;
    } else if last_operate == 1 {
        pivot_row -= 1;
        pivot_col -= 1;
    }
    while pivot_row > 0 {
        for r in 0..pivot_row {
            let scale: f64 = result_matrix.entries[r][pivot_col] / result_matrix.entries[pivot_row][pivot_col];
            result_vector.entries[r] -= scale * result_vector.entries[pivot_row];
            for c in pivot_col..result_matrix.col {
                result_matrix.entries[r][c] -= scale * result_matrix.entries[pivot_row][c];
            }
        }
        pivot_row -= 1;
        pivot_col -= 1;
    }

    // Pivots -> 1
    for r in 0..result_matrix.row {
        for c in r..result_matrix.col {
            if result_matrix.entries[r][c] != 0.0 {
                let scale: f64 = result_matrix.entries[r][c];
                for e in c..result_matrix.col {
                    result_matrix.entries[r][e] /= scale;
                }
                result_vector.entries[r] /= scale;

                break;
            }
        }
    }

    Ok((result_matrix, result_vector, permutation))
}

pub fn null_space(matrix: &Matrix) -> Matrix {
    let rref: Matrix = gauss_jordan_elimination(matrix, &Vector::zeros(matrix.row)).unwrap().0;
    
    // Construct the matrix that contains relationship between each pivot and behind element.
    // Each column only contains two element.
    const THERESHOLD: f64 = 1e-8;
    let mut null_relate: Matrix = Matrix::zeros(0, 0);
    for r in (0..rref.row.min(rref.col)).rev() {
        let mut pivot = r;
        while rref.entries[r][pivot].abs() < THERESHOLD {
            pivot += 1;
            if pivot == rref.col {break}
        }
        
        for right in (pivot + 1)..rref.col {
            if rref.entries[r][right].abs() < THERESHOLD {continue}

            let mut relate_vector: Vector = Vector::zeros(rref.col);
            relate_vector.entries[pivot] = 1.0;
            relate_vector.entries[right] = -1.0 / rref.entries[r][right];
            null_relate = null_relate.append_Vector(&relate_vector, 1).unwrap();
        }
    }

    // Each column divide by the bottom value.
    for c in 0..null_relate.col {
        let mut find_base: bool = false;
        let mut base_value: f64 = 0.0;
        for r in (0..null_relate.row).rev() {
            if null_relate.entries[r][c] == 0.0 {
                continue;
            } else if !find_base {
                base_value = null_relate.entries[r][c];
                null_relate.entries[r][c] = 1.0;
                find_base = true;
            } else {
                null_relate.entries[r][c] /= base_value;
            }
            
        }
    }

    // Combine columns if has the same bottom value.
    let mut null_basis: Matrix = Matrix::zeros(0, 0);
    for r in (0.. null_relate.row).rev() {
        let mut null_vector = Vector::zeros(rref.col);
        null_vector.entries[r] = 1.0;
        for c in 0..null_relate.col {
            if null_relate.entries[r][c] == 1.0 {
                for e in 0..r {
                    if null_relate.entries[e][c] != 0.0 {
                        null_vector.entries[e] = null_relate.entries[e][c];
                        break;
                    }
                }
            }
        }

        let mut element_num: i32 = 0;
        for e in 0..null_vector.size {
            if null_vector.entries[e] != 0.0 {
                element_num += 1;
            }
            if element_num == 2 {
                null_basis = null_basis.append_Vector(&null_vector, 1).unwrap();
                break;
            }
        }
    }

    // Complete the eigenvector
    for c in 0.. rref.col {
        let mut zero_num: usize = 0;
        for r in 0.. rref.row {
            if rref.entries[r][c] == 0.0 {zero_num += 1}
            else {break;}
        }
        if zero_num == rref.col {
            let mut zero_vector = Vector::zeros(rref.col);
            zero_vector.entries[c] = 1.0;
            null_basis = null_basis.append_Vector(&zero_vector, 1).unwrap();
        }
    }
    if null_basis.row == 0 {
        null_basis = null_basis.append_Vector(&Vector::zeros(rref.col), 1).unwrap();
    }

    null_basis
}