use std::vec;

use crate::vector::Vector;
use rand::Rng;

#[derive(Clone)]
pub struct Matrix {
    pub entries: Vec<Vec<f64>>,
    pub row: usize,
    pub col: usize,
}

impl Matrix {
    pub fn from_double_vec(double_vector: &Vec<Vec<f64>>) -> Matrix {
        Matrix {
            entries: double_vector.clone(),
            row: double_vector.len(),
            col: double_vector[0].len(),
        }
    }

    /// Transform a Vector into a Matrix.
    ///
    /// If axis == 0 : vector as a row.
    ///
    /// If axis == 1 : vector as a column.
    pub fn from_Vector(vector: &Vector, axis: usize) -> Result<Matrix, String> {
        match axis {
            x if x == 0 => {
                return Ok(Matrix {
                    entries: vector.to_Matrix(0).unwrap().entries,
                    row: 1,
                    col: vector.size,
                });
            }

            x if x == 1 => {
                return Ok(Matrix {
                    entries: vector.to_Matrix(1).unwrap().entries,
                    row: vector.size,
                    col: 1,
                });
            }

            _ => {
                return Err("Input Error: Input axis is not valid.".to_string());
            }
        }
    }

    pub fn get_column_vector(self: &Self, col: usize) -> Result<Vector, String> {
        if col > self.col {
            return Err("Input Error: Input col is out of bound.".to_string());
        }

        let mut col_vector: Vec<f64> = Vec::new();
        for r in 0..self.row {
            col_vector.push(self.entries[r][col]);
        }

        Ok(Vector::from_vec(&col_vector))
    }

    pub fn get_row_vector(self: &Self, row: usize) -> Result<Vector, String> {
        if row > self.row {
            return Err("Input Error: Input col is out of bound.".to_string());
        }

        Ok(Vector::from_vec(&self.entries[row].clone()))
    }

    /// Return the matrix that round to the digit after decimal point.
    pub fn round(self: &Self, digit: u32) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        let scale: f64 = 10_i32.pow(digit as u32) as f64;
        for r in 0..self.row {
            for c in 0..self.col {
                // result_matrix.entries[r][c] = (scale * result_matrix.entries[r][c]).round() / scale;
                result_matrix.entries[r][c] = (scale * result_matrix.entries[r][c]).round();

                if result_matrix.entries[r][c] >= 1.0 || result_matrix.entries[r][c] <= -1.0 {
                    result_matrix.entries[r][c] /= scale;
                } else if result_matrix.entries[r][c].is_nan() {
                    continue;
                } else {
                    result_matrix.entries[r][c] = 0.0;
                }
            }
        }

        result_matrix
    }

    pub fn replace_nan(self: &Self) -> Matrix {
        let mut result_matrix = self.clone();
        for r in 0..result_matrix.row {
            for c in 0..result_matrix.col {
                if result_matrix.entries[r][c].is_nan() {
                    result_matrix.entries[r][c] = 0.0;
                }
            }
        }

        result_matrix
    }

    pub fn display(self: &Self) {
        if self.row == 0 {
            println!(
                "Matrix: [[]], shape: {} x {}",
                self.row, self.col
            );
            return;
        } else if self.row == 1 {
            println!(
                "Matrix: [{:?}], shape: {} x {}",
                self.entries[0], self.row, self.col
            );
            return;
        }

        println!("Matrix: [{:8?}", self.entries[0]);
        for r in 1..(self.row - 1) {
            println!("         {:8?}", self.entries[r]);
        }
        println!(
            "{}",
            format!(
                "         {:8?}], shape: {} x {}",
                self.entries[self.row - 1],
                self.row,
                self.col
            )
        );
    }

    /// Return a matrix contains all one entries with m rows and n cols.
    pub fn ones(m: usize, n: usize) -> Matrix {
        Matrix {
            entries: vec![vec![1.0; n]; m],
            row: m,
            col: n,
        }
    }

    /// Return a matrix contains all zero entries with m rows and n cols.
    pub fn zeros(m: usize, n: usize) -> Matrix {
        Matrix {
            entries: vec![vec![0.0; n]; m],
            row: m,
            col: n,
        }
    }

    pub fn identity(m: usize) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, m);
        for d in 0..m {
            result_matrix.entries[d][d] = 1.0;
        }

        result_matrix
    }

    pub fn random_matrix(m: usize, n: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, n);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            for c in 0..n {
                result_matrix.entries[r][c] = generator.random_range(min..max);
            }
        }

        result_matrix
    }

    /// Return the upper triangular matrix or self.
    pub fn random_upper_triangular(m: usize, n: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, n);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            for c in r..n {
                result_matrix.entries[r][c] = generator.random_range(min..max);
            }
        }

        result_matrix
    }

    /// Return the lower triangular matrix or self.
    pub fn random_lower_triangular(m: usize, n: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, n);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            for c in 0..(r + 1).min(n) {
                result_matrix.entries[r][c] = generator.random_range(min..max);
            }
        }

        result_matrix
    }

    pub fn random_diagonal_matrix(m: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, m);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            result_matrix.entries[r][r] = generator.random_range(min..max);
        }

        result_matrix
    }

    pub fn random_symmetric_matrix(m: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::random_diagonal_matrix(m, min, max);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            for c in (r + 1)..m {
                result_matrix.entries[r][c] = generator.random_range(min..max);
                result_matrix.entries[c][r] = result_matrix.entries[r][c];
            }
        }

        result_matrix
    }

    /// Sum up all the entries in matrix.
    pub fn entries_sum(self: &Self) -> f64 {
        let mut summation: f64 = 0.0;
        for r in 0..self.row {
            for c in 0..self.col {
                summation += self.entries[r][c];
            }
        }

        summation
    }

    pub fn trace(self: &Self) -> f64 {
        let mut summation: f64 = 0.0;
        for r in 0..self.row {
            summation += self.entries[r][r];
        }

        summation
    }

    /// Add two matrix element-wise.
    pub fn add_Matrix(self: &Self, matrix: &Matrix) -> Result<Matrix, String> {
        if self.row != matrix.row || self.col != matrix.col {
            return Err("Input Error: The size of input matrix does not match.".to_string());
        }

        let mut result_matrix = self.clone();
        for r in 0..self.row {
            for c in 0..self.col {
                result_matrix.entries[r][c] += matrix.entries[r][c];
            }
        }

        Ok(result_matrix)
    }

    /// Substract two matrix element-wise.
    pub fn substract_Matrix(self: &Self, matrix: &Matrix) -> Result<Matrix, String> {
        if self.row != matrix.row || self.col != matrix.col {
            return Err("Input Error: The size of input matrix does not match.".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.row {
            for c in 0..self.col {
                result_matrix.entries[r][c] -= matrix.entries[r][c];
            }
        }

        Ok(result_matrix)
    }

    pub fn multiply_scalar(self: &Self, scalar: &f64) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(self.row, self.col);
        for r in 0..self.row {
            for c in 0..self.col {
                result_matrix.entries[r][c] = scalar * self.entries[r][c];
            }
        }

        result_matrix
    }

    pub fn multiply_Vector(self: &Self, vector: &Vector) -> Result<Vector, String> {
        if self.col != vector.size {
            return Err("Input Error: The size of vector does not match.".to_string());
        }

        let mut dot: f64 = 0.0;
        let mut result_vector: Vector = Vector::zeros(self.row);
        for r in 0..self.row {
            dot = 0.0;
            for c in 0..self.col {
                dot += self.entries[r][c] * vector.entries[c];
            }
            result_vector.entries[r] = dot;
        }

        Ok(result_vector)
    }

    pub fn multiply_Matrix(self: &Self, matrix: &Matrix) -> Result<Matrix, String> {
        if self.col != matrix.row {
            return Err("Input Error: The size of two matrixes do not match.".to_string());
        }

        let mut dot: f64 = 0.0;
        let mut result_matrix: Matrix = Self::zeros(self.row, matrix.col);
        for r in 0..self.row {
            for c in 0..matrix.col {
                dot = 0.0;
                for e in 0..self.col {
                    dot += self.entries[r][e] * matrix.entries[e][c];
                }
                result_matrix.entries[r][c] = dot;
            }
        }

        Ok(result_matrix)
    }

    /// Append a vector to a matrix along the axis.
    ///
    /// If axis == 0 : append vector as a row.
    ///
    /// If axis == 1 : append vector as a column.
    pub fn append_Vector(self: &Self, vector: &Vector, axis: usize) -> Result<Matrix, String> {
        if self.row == 0 {
            match axis {
            x if x == 0 => return Ok(vector.to_Matrix(0).unwrap()),
            x if x == 1 => return Ok(vector.to_Matrix(1).unwrap()),
            _ => return Err("Input Error: Input axis is not valid.".to_string())
            }
        }
        
        match axis {
            x if x == 0 => {
                if self.col != vector.size {
                    return Err(
                        "Input Error: Vector size does not match the size of matrix's row."
                            .to_string(),
                    );
                }

                let mut result_matrix: Matrix = self.clone();
                result_matrix.entries.push(vector.entries.clone());
                result_matrix.row += 1;

                return Ok(result_matrix);
            }

            x if x == 1 => {
                if self.row != vector.size {
                    return Err(
                        "Input Error: Vector size does not match the size of matrix's column."
                            .to_string(),
                    );
                }

                let mut result_matrix: Matrix = self.clone();
                for e in 0..vector.size {
                    result_matrix.entries[e].push(vector.entries[e]);
                }
                result_matrix.col += 1;

                return Ok(result_matrix);
            }

            _ => {
                return Err("Input Error: Input axis is not valid.".to_string());
            }
        }
    }

    /// Append a vector to a matrix along the axis.
    ///
    /// If axis == 0 : append matirx to the bottom.
    ///   
    /// If axis == 1 : append matirx to the right.
    pub fn append_Matrix(self: &Self, matrix: &Matrix, axis: usize) -> Result<Matrix, String> {
        if self.row == 0 {
            match axis {
            x if x == 0 => return Ok(matrix.clone()),
            x if x == 1 => return Ok(matrix.clone()),
            _ => return Err("Input Error: Input axis is not valid.".to_string())
            }
        }

        match axis {
            x if x == 0 => {
                if self.col != matrix.col {
                    return Err("Input Error: The size of row does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                for r in 0..matrix.row {
                    result_matrix.entries.push(matrix.entries[r].clone());
                }
                result_matrix.row += matrix.row;

                return Ok(result_matrix);
            }

            x if x == 1 => {
                if self.row != matrix.row {
                    return Err("Input Error: The size of column does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                for r in 0..matrix.row {
                    for c in 0..matrix.col {
                        result_matrix.entries[r].push(matrix.entries[r][c]);
                    }
                }
                result_matrix.col += matrix.col;

                return Ok(result_matrix);
            }

            _ => {
                return Err("Input Error: Input axis is not valid.".to_string());
            }
        }
    }

    /// Reshape the matrix into the shape(row, column).
    pub fn reshpae(self: &Self, shape: (usize, usize)) -> Result<Matrix, String> {
        if self.row * self.col != shape.0 * shape.1 {
            return Err(format!(
                "Input Error: The matrix can't not reshape to the shape ({}, {})",
                shape.0, shape.1
            ));
        }

        let mut element: Vec<f64> = Vec::new();
        for r in 0..self.row {
            for c in 0..self.col {
                element.push(self.entries[r][c]);
            }
        }

        element.reverse();
        let mut result_matrix: Matrix = Self::zeros(shape.0, shape.1);
        for r in 0..shape.0 {
            for c in 0..shape.1 {
                result_matrix.entries[r][c] = element.pop().unwrap();
            }
        }

        Ok(result_matrix)
    }

    pub fn transpose(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(self.col, self.row);
        for r in 0..result_matrix.row {
            for c in 0..result_matrix.col {
                result_matrix.entries[r][c] = self.entries[c][r];
            }
        }

        result_matrix
    }

    /// Return the matrix whithout the row.
    ///
    /// Parameter row is start from 0.
    pub fn remove_row(self: &Self, row: usize) -> Result<Matrix, String> {
        if row >= self.row {
            return Err("Input Error: Input row is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        result_matrix.row -= 1;
        result_matrix.entries.remove(row);

        Ok(result_matrix)
    }

    /// Return the matrix whithout the row.
    ///
    /// Parameter row is start from 0.
    pub fn remove_col(self: &Self, col: usize) -> Result<Matrix, String> {
        if col < 0 {
            return Err("Input Error: Input col is less than zero".to_string());
        } else if col >= self.col {
            return Err("Input Error: Input col is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        result_matrix.col -= 1;
        for r in 0..self.row {
            result_matrix.entries[r].remove(col);
        }

        Ok(result_matrix)
    }

    pub fn swap_row(self: &Self, row1: usize, row2: usize) -> Result<Matrix, String> {
        if row1 >= self.row || row2 >= self.row {
            return Err("Input Error: Input row1 or row2 is out of bound".to_string());
        }
        let mut result_matrix: Matrix = self.clone();
        result_matrix.entries[row1] = self.entries[row2].clone();
        result_matrix.entries[row2] = self.entries[row1].clone();

        Ok(result_matrix)
    }

    pub fn swap_column(self: &Self, col1: usize, col2: usize) -> Result<Matrix, String> {
        if col1 >= self.col || col2 >= self.col {
            return Err("Input Error: Input row1 or row2 is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        for r in 0..result_matrix.row {
            result_matrix.entries[r][col1] = self.entries[r][col2];
            result_matrix.entries[r][col2] = self.entries[r][col1];
        }

        Ok(result_matrix)
    }

    /// Swap the rows according to the order of permutaion matrix.
    pub fn swap_with_permutation(self: &Self, permutation: &Matrix) -> Result<Matrix, String> {
        if self.row != permutation.row {
            return Err(
                "Input Error: The row size of permutation matrix does not match".to_string(),
            );
        }
        Ok(permutation.multiply_Matrix(self).unwrap())
    }

    /// Return
    pub fn determinant(self: &Self) -> Result<f64, String> {
        if !self.is_square() {
            return Err("Value Error: This matrix is not a square matrix.".to_string());
        }

        let mut matrix_u: Matrix = self.clone();
        let mut matrix_l: Matrix = Matrix::zeros(self.row, self.row);
        let mut permutation: Matrix = Matrix::identity(self.row);
        for c in 0..self.col {
            // If the pivot is 0.0, swap to non zero.
            let mut is_swap = false;
            if matrix_u.entries[c][c] == 0.0 {
                for r in (c + 1)..matrix_u.row {
                    if matrix_u.entries[r][c] != 0.0 {
                        matrix_u = matrix_u.swap_row(c, r).unwrap();
                        matrix_l = matrix_l.swap_row(c, r).unwrap();
                        permutation = permutation.swap_row(c, r).unwrap();
                        is_swap = true;
                        break;
                    }
                }
                if !is_swap {
                    continue;
                }
            }

            for r in (c + 1)..self.row {
                matrix_l.entries[r][c] = matrix_u.entries[r][c] / matrix_u.entries[c][c];
                for e in 0..self.col {
                    matrix_u.entries[r][e] -= matrix_l.entries[r][c] * matrix_u.entries[c][e];
                }
            }
        }
        matrix_l = matrix_l.add_Matrix(&Matrix::identity(self.row)).unwrap();

        let mut det_l: f64 = matrix_l.entries[0][0];
        let mut det_u: f64 = matrix_u.entries[0][0];
        for r in 1..matrix_l.row {
            det_l *= matrix_l.entries[r][r];
            det_u *= matrix_u.entries[r][r];
        }

        Ok(det_l * det_u)
    }

    pub fn adjoint(self: &mut Matrix) -> Matrix {
        let mut adjoint_matrix: Matrix = Self::zeros(self.row, self.col);
        let mut sign: f64 = 1.0;
        for r in 0..adjoint_matrix.row {
            for c in 0..adjoint_matrix.col {
                let sub_matrix: Matrix = self.remove_row(r).unwrap().remove_col(c).unwrap();
                adjoint_matrix.entries[r][c] = sign * sub_matrix.determinant().unwrap();
                sign *= -1.0;
            }
        }
        adjoint_matrix.transpose()
    }

    /// Return the inverse matrix of self if have.
    /// Using
    pub fn inverse(self: &Self) -> Result<Matrix, String> {
        if self.row != self.col {
            return Err("Value Error: This matrix is not a squared matrix.".to_string());
        } else if self.row == 0 {
            return Err("Value Error: This matrix is empty.".to_string());
        }

        if self.row == 1 {
            return Ok(Matrix {
                entries: vec![vec![1.0 / self.entries[0][0]]],
                row: 1,
                col: 1,
            });
        }

        let determinant: f64 = self.determinant().unwrap();
        if determinant == 0.0 {
            return Err("Value Error: This matrix is not invertible".to_string());
        }

        // Get upper triangular form.
        let mut matrix: Matrix = self.clone();
        let mut inverse_matrix: Matrix = Self::identity(self.row);
        for d in 0..matrix.col {
            // If the pivot is 0.0, swap to non zero.
            if matrix.entries[d][d] == 0.0 {
                for r in (d + 1)..matrix.row {
                    if matrix.entries[r][d] != 0.0 {
                        matrix = matrix.swap_row(d, r).unwrap();
                        inverse_matrix = inverse_matrix.swap_row(d, r).unwrap();
                    }
                }
            }

            for r in (d + 1)..matrix.row {
                let scale: f64 = matrix.entries[r][d] / matrix.entries[d][d];
                for e in 0..matrix.col {
                    matrix.entries[r][e] -= scale * matrix.entries[d][e];
                    inverse_matrix.entries[r][e] -= scale * inverse_matrix.entries[d][e];
                }
            }
        }

        // To identity
        for d in (0..matrix.col).rev() {
            for r in (0..d).rev() {
                let scale = matrix.entries[r][d] / matrix.entries[d][d];
                matrix.entries[r][d] -= scale * matrix.entries[d][d];
                for c in 0..inverse_matrix.col {
                    inverse_matrix.entries[r][c] -= scale * inverse_matrix.entries[d][c];
                }
            }
        }

        // Pivots -> 1
        for r in 0..matrix.row {
            for c in r..matrix.col {
                if matrix.entries[r][c] != 0.0 {
                    let scale: f64 = matrix.entries[r][c];
                    for e in c..matrix.col {
                        matrix.entries[r][e] /= scale;
                    }
                    for e in 0..inverse_matrix.col {
                        inverse_matrix.entries[r][e] /= scale;
                    }

                    break;
                }
            }
        }

        Ok(inverse_matrix)
    }

    pub fn is_square(self: &Self) -> bool {
        self.row == self.col
    }

    pub fn is_upper_triangular(self: &Self) -> bool {
        for r in 1..self.row {
            for c in 0..r.min(self.col) {
                if self.entries[r][c] != 0.0 {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_lower_triangular(self: &Self) -> bool {
        for r in 0..self.row.min(self.col) {
            for c in (r + 1)..self.col {
                if self.entries[r][c] != 0.0 {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_symmetric(self: &Self) -> bool {
        if !self.is_square() {
            return false;
        }

        for r in 0..self.row {
            for c in (r + 1)..self.col {
                if self.entries[r][c] != self.entries[c][r] {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_invertible(self: &Self) -> bool {
        match self.determinant() {
            Err(_) => {
                return false;
            }
            Ok(d) => {
                if d != 0.0 {
                    return true;
                } else {
                    return false;
                }
            }
        }
    }

    /// Need to Update!
    pub fn is_positive_definite(self: &Self) -> bool {
        if !self.is_symmetric() {
            return false;
        }

        for d in 1..self.row {
            if self.entries[d][d - 1].powi(2) >= self.entries[d][d] {
                return false;
            }
        }

        true
    }

    pub fn calculate_square_error(self: &Self, matrix: &Matrix) -> Result<f64, String> {
        if self.row != matrix.row || self.col != matrix.col {
            return Err("Input Error: The size of input matrix does not match.".to_string());
        }

        let mut error: f64 = 0.0;
        for r in 0..self.row {
            for c in 0..self.col {
                error += (self.entries[r][c] - matrix.entries[r][c]).powi(2);
            }
        }

        Ok(error)
    }

    /// Return the matrix that took square root on each element.
    pub fn square_root(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.row {
            for c in 0..self.col {
                result_matrix.entries[r][c] = result_matrix.entries[r][c].sqrt();
            }
        }

        result_matrix
    }

    /// Return the matrix that took power of 2 on each element.
    pub fn to_powi(self: &Self, power: i32) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.row {
            for c in 0..self.col {
                result_matrix.entries[r][c] = result_matrix.entries[r][c].powi(power);
            }
        }

        result_matrix
    }

    /// Return the upper triangular form of self.
    ///
    /// Eliminate those elements which lay in lower triangular.
    pub fn eliminate_lower_triangular(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        let col_bound = result_matrix.col;
        for r in 1..self.row {
            for c in 0..r.min(col_bound) {
                result_matrix.entries[r][c] = 0.0;
            }
        }

        result_matrix
    }

    /// Return the lower triangular form of self.
    ///
    /// Eliminate those elements which lay in upper triangular.
    pub fn eliminate_upper_triangular(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        let col_bound = result_matrix.col;
        for r in 0..self.row {
            for c in (r + 1)..self.col {
                result_matrix.entries[r][c] = 0.0;
            }
        }

        result_matrix
    }

    /// Return a matrix only contains the diagonal entries.
    pub fn take_diagonal_entries(self: &Self) -> Matrix {
        self.eliminate_lower_triangular()
            .eliminate_upper_triangular()
    }

    /// Return a matrix only contains the diagonal entries.
    ///
    /// Parameter row is start from 0.
    pub fn take_row(self: &Self, row: usize) -> Result<Vector, String> {
        if row >= self.row {
            return Err("Input Error: Parameter row is out of bound.".to_string());
        }

        Ok(Vector::from_vec(&self.entries[row]))
    }

    /// Return a matrix only contains the diagonal entries.
    ///
    /// Parameter col is start from 0.
    pub fn take_col(self: &Self, col: usize) -> Result<Vector, String> {
        if col >= self.col {
            return Err("Input Error: Parameter col is out of bound.".to_string());
        }

        let mut result_vector: Vec<f64> = Vec::new();
        for r in 0..self.row {
            result_vector.push(self.entries[r][col]);
        }

        Ok(Vector::from_vec(&result_vector))
    }
}
