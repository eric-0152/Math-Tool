use std::ops::{Add, Sub, Mul};
use rand::Rng;

#[derive(Clone)]
pub struct Matrix {
    pub shape: (usize, usize),
    pub real: Vec<Vec<f64>>,
    pub imaginary: Vec<Vec<f64>>,
}

#[macro_export]
macro_rules! to_mat {
    ([$([$($e: expr), *]), *]) => {{
        let mut rows = Vec::new();

        $(
            let mut row: Vec<f64> = Vec::new();
            $(
                row.push($e);
            )*

            rows.push(row);
        )*

        let result_matrix = Matrix::from_double_vec(&rows);

        result_matrix
    }};
}

#[macro_export]
macro_rules! to_mat_t {
    // ([$([$($c: expr), *]), *]) => {{
    ([$([$($c: expr), *]), *]) => {{
        let mut real_rows = Vec::new();
        let mut img_rows = Vec::new();
        $(
            let mut real_row: Vec<f64> = Vec::new();
            let mut img_row: Vec<f64> = Vec::new();
            $(
                let complex = $c.to_string();
                println!("{:?}", $c);
                // if complex.contains("+") {
                //     let mut part = complex.split("+");
                //     let real = part.nth(0).unwrap().parse().unwrap();
                //     let img = part.nth(1).unwrap().parse().unwrap();
                //     real_row.push(real);
                //     img_row.push(img);
                // } else {
                //     let mut part = complex.split("-");
                //     let real = part.nth(0).unwrap().parse().unwrap();
                //     let img = part.nth(1).unwrap().parse().unwrap();
                //     real_row.push(real);
                //     img_row.push(img);
                // }
            )*

            real_rows.push(real_row);
            img_rows.push(img_row);
        )*

        println!("{:?}", real_rows);
        println!("{:?}", img_rows);
        Matrix {
            shape: (real_rows.len(), real_rows[0].len()),
            real: real_rows,
            imaginary: img_rows
        }
    }};
}

impl Add for &Matrix {
    type Output = Matrix;
    fn add(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.real[r][c] += matrix.real[r][c];
                result_matrix.imaginary[r][c] += matrix.imaginary[r][c];
            }

        }

        result_matrix
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.real[r][c] -= matrix.real[r][c];
                result_matrix.imaginary[r][c] -= matrix.imaginary[r][c];
            }

        }

        result_matrix
    }
}

impl Mul<&f64> for &Matrix {
    type Output = Matrix;
    fn mul(self: Self, scalar: &f64) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.real[r][c] *= scalar;
                result_matrix.imaginary[r][c] *= scalar;
            }

        }

        result_matrix
    }
}

impl Mul<&Matrix> for &f64 {
    type Output = Matrix;

    fn mul(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = matrix.clone();
        for r in 0..matrix.shape.0 {
            for c in 0..matrix.shape.1 {
                result_matrix.real[r][c] *= self;
                result_matrix.imaginary[r][c] *= self;
            }

        }

        result_matrix
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;

    fn mul(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = Matrix::zeros(self.shape.0, matrix.shape.1).clone();
        for r in 0..result_matrix.shape.0 {
            for c in 0..result_matrix.shape.1 {
                for e in 0..self.shape.1 {
                    result_matrix.real[r][c] += self.real[r][e] * matrix.real[e][c];
                    result_matrix.real[r][c] -= self.imaginary[r][e] * matrix.imaginary[e][c];
                    result_matrix.imaginary[r][c] += self.real[r][e] * matrix.imaginary[e][c];
                    result_matrix.imaginary[r][c] += self.imaginary[r][e] * matrix.real[e][c];
                }
            }

        }

        result_matrix
    }
}



impl Matrix {
    pub fn from_vec(vec: &Vec<f64>) -> Matrix {
        let mut entries: Vec<Vec<f64>> = Vec::new();
        for e in 0..vec.len() {
            entries.push(vec![vec[e]]);
        }

        Matrix {
            shape: (vec.len(), 1),
            real: entries,
            imaginary: vec![vec![0.0; 1]; vec.len()],
        }
    }

    pub fn from_double_vec(double_vector: &Vec<Vec<f64>>) -> Matrix {
        Matrix {
            shape: (double_vector.len(), double_vector[0].len()),
            real: double_vector.clone(),
            imaginary: vec![vec![0.0; double_vector[0].len()]; double_vector.len()]
        }
    }

    pub fn get_column_vector(self: &Self, col: usize) -> Result<Matrix, String> {
        if col > self.shape.1 {
            return Err("Input Error: Input col is out of bound.".to_string());
        }

        let mut col_vector: Vec<f64> = Vec::new();
        for r in 0..self.shape.0 {
            col_vector.push(self.real[r][col]);
        }

        Ok(Matrix::from_vec(&col_vector))
    }

    pub fn get_row_vector(self: &Self, row: usize) -> Result<Matrix, String> {
        if row > self.shape.0 {
            return Err("Input Error: Input col is out of bound.".to_string());
        }

        Ok(Matrix::from_vec(&self.real[row].clone()))
    }

    /// Return the matrix that round to the digit after decimal point.
    pub fn round(self: &Self, digit: u32) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        let scale: f64 = 10_i32.pow(digit as u32) as f64;
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                // result_matrix.real[r][c] = (scale * result_matrix.real[r][c]).round() / scale;
                result_matrix.real[r][c] = (scale * result_matrix.real[r][c]).round();

                if result_matrix.real[r][c] >= 1.0 || result_matrix.real[r][c] <= -1.0 {
                    result_matrix.real[r][c] /= scale;
                } else if result_matrix.real[r][c].is_nan() {
                    continue;
                } else {
                    result_matrix.real[r][c] = 0.0;
                }
            }
        }

        result_matrix
    }

    pub fn replace_nan(self: &Self) -> Matrix {
        let mut result_matrix = self.clone();
        for r in 0..result_matrix.shape.0 {
            for c in 0..result_matrix.shape.1 {
                if result_matrix.real[r][c].is_nan() {
                    result_matrix.real[r][c] = 0.0;
                }
            }
        }

        result_matrix
    }

    pub fn display(self: &Self) {
        if self.shape.0 == 0 {
            println!("[[]], shape: {} x {}", self.shape.0, self.shape.1);
            return;
        } else if self.shape.0 == 1 {
            println!(
                "[{:?}], shape: {} x {}",
                self.real[0], self.shape.0, self.shape.1
            );
            return;
        }

        println!("[{:8?}", self.real[0]);
        for r in 1..(self.shape.0 - 1) {
            println!(" {:8?}", self.real[r]);
        }
        println!(
            "{}",
            format!(
                " {:8?}], shape: {} x {}",
                self.real[self.shape.0 - 1],
                self.shape.0,
                self.shape.1
            )
        );
    }

    /// Return a matrix contains all one real with m rows and n cols.
    pub fn ones(m: usize, n: usize) -> Matrix {
        Matrix {
            shape: (m, n),
            real: vec![vec![1.0; n]; m],
            imaginary: vec![vec![0.0; n]; m],
        }
    }

    /// Return a matrix contains all zero real with m rows and n cols.
    pub fn zeros(m: usize, n: usize) -> Matrix {
        Matrix {
            shape: (m, n),
            real: vec![vec![0.0; n]; m],
            imaginary: vec![vec![0.0; n]; m],
        }
    }

    pub fn identity(m: usize) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, m);
        for d in 0..m {
            result_matrix.real[d][d] = 1.0;
        }

        result_matrix
    }

    pub fn random_matrix(m: usize, n: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, n);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            for c in 0..n {
                result_matrix.real[r][c] = generator.random_range(min..max);
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
                result_matrix.real[r][c] = generator.random_range(min..max);
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
                result_matrix.real[r][c] = generator.random_range(min..max);
            }
        }

        result_matrix
    }

    pub fn random_diagonal_matrix(m: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, m);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            result_matrix.real[r][r] = generator.random_range(min..max);
        }

        result_matrix
    }

    pub fn random_symmetric_matrix(m: usize, min: f64, max: f64) -> Matrix {
        let mut result_matrix: Matrix = Self::random_diagonal_matrix(m, min, max);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        for r in 0..m {
            for c in (r + 1)..m {
                result_matrix.real[r][c] = generator.random_range(min..max);
                result_matrix.real[c][r] = result_matrix.real[r][c];
            }
        }

        result_matrix
    }

    /// Sum up all the real in matrix.
    pub fn entries_sum(self: &Self) -> f64 {
        let mut summation: f64 = 0.0;
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                summation += self.real[r][c];
            }
        }

        summation
    }

    pub fn trace(self: &Self) -> f64 {
        let mut summation: f64 = 0.0;
        for r in 0..self.shape.0 {
            summation += self.real[r][r];
        }

        summation
    }

    /// Append a matrix along the axis.
    ///
    /// If axis == 0 : append matirx to the bottom.
    ///   
    /// If axis == 1 : append matirx to the right.
    pub fn append(self: &Self, matrix: &Matrix, axis: usize) -> Result<Matrix, String> {
        if self.shape.0 == 0 {
            match axis {
                x if x == 0 => return Ok(matrix.clone()),
                x if x == 1 => return Ok(matrix.clone()),
                _ => return Err("Input Error: Input axis is not valid.".to_string()),
            }
        }

        match axis {
            x if x == 0 => {
                if self.shape.1 != matrix.shape.1 {
                    return Err("Input Error: The size of row does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                for r in 0..matrix.shape.0 {
                    result_matrix.real.push(matrix.real[r].clone());
                }
                result_matrix.shape.0 += matrix.shape.0;

                return Ok(result_matrix);
            }

            x if x == 1 => {
                if self.shape.0 != matrix.shape.0 {
                    return Err("Input Error: The size of column does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                for r in 0..matrix.shape.0 {
                    for c in 0..matrix.shape.1 {
                        result_matrix.real[r].push(matrix.real[r][c]);
                    }
                }
                result_matrix.shape.1 += matrix.shape.1;

                return Ok(result_matrix);
            }

            _ => {
                return Err("Input Error: Input axis is not valid.".to_string());
            }
        }
    }

    /// Reshape the matrix into the shape(row, column).
    pub fn reshpae(self: &Self, shape: (usize, usize)) -> Result<Matrix, String> {
        if self.shape.0 * self.shape.1 != shape.0 * shape.1 {
            return Err(format!(
                "Input Error: The matrix can't not reshape to the shape ({}, {})",
                shape.0, shape.1
            ));
        }

        let mut element: Vec<f64> = Vec::new();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                element.push(self.real[r][c]);
            }
        }

        element.reverse();
        let mut result_matrix: Matrix = Self::zeros(shape.0, shape.1);
        for r in 0..shape.0 {
            for c in 0..shape.1 {
                result_matrix.real[r][c] = element.pop().unwrap();
            }
        }

        Ok(result_matrix)
    }

    pub fn transpose(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(self.shape.1, self.shape.0);
        for r in 0..result_matrix.shape.0 {
            for c in 0..result_matrix.shape.1 {
                result_matrix.real[r][c] = self.real[c][r];
            }
        }

        result_matrix
    }

    /// Return the matrix whithout the row.
    ///
    /// Parameter row is start from 0.
    pub fn remove_row(self: &Self, row: usize) -> Result<Matrix, String> {
        if row >= self.shape.0 {
            return Err("Input Error: Input row is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        result_matrix.shape.0 -= 1;
        result_matrix.real.remove(row);

        Ok(result_matrix)
    }

    /// Return the matrix whithout the row.
    ///
    /// Parameter row is start from 0.
    pub fn remove_col(self: &Self, col: usize) -> Result<Matrix, String> {
        if col < 0 {
            return Err("Input Error: Input col is less than zero".to_string());
        } else if col >= self.shape.1 {
            return Err("Input Error: Input col is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        result_matrix.shape.1 -= 1;
        for r in 0..self.shape.0 {
            result_matrix.real[r].remove(col);
        }

        Ok(result_matrix)
    }

    pub fn swap_row(self: &Self, row1: usize, row2: usize) -> Result<Matrix, String> {
        if row1 >= self.shape.0 || row2 >= self.shape.0 {
            return Err("Input Error: Input row1 or row2 is out of bound".to_string());
        }
        let mut result_matrix: Matrix = self.clone();
        result_matrix.real[row1] = self.real[row2].clone();
        result_matrix.real[row2] = self.real[row1].clone();

        Ok(result_matrix)
    }

    pub fn swap_column(self: &Self, col1: usize, col2: usize) -> Result<Matrix, String> {
        if col1 >= self.shape.1 || col2 >= self.shape.1 {
            return Err("Input Error: Input row1 or row2 is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        for r in 0..result_matrix.shape.0 {
            result_matrix.real[r][col1] = self.real[r][col2];
            result_matrix.real[r][col2] = self.real[r][col1];
        }

        Ok(result_matrix)
    }

    /// Swap the rows according to the order of permutaion matrix.
    pub fn swap_with_permutation(self: &Self, permutation: &Matrix) -> Result<Matrix, String> {
        if self.shape.0 != permutation.shape.0 {
            return Err(
                "Input Error: The row size of permutation matrix does not match".to_string(),
            );
        }
        Ok(permutation * self)
    }

    /// Return
    pub fn determinant(self: &Self) -> Result<f64, String> {
        if !self.is_square() {
            return Err("Value Error: This matrix is not a square matrix.".to_string());
        }

        let mut matrix_u: Matrix = self.clone();
        let mut matrix_l: Matrix = Matrix::zeros(self.shape.0, self.shape.0);
        let mut permutation: Matrix = Matrix::identity(self.shape.0);
        for c in 0..self.shape.1 {
            // If the pivot is 0.0, swap to non zero.
            let mut is_swap = false;
            if matrix_u.real[c][c] == 0.0 {
                for r in (c + 1)..matrix_u.shape.0 {
                    if matrix_u.real[r][c] != 0.0 {
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

            for r in (c + 1)..self.shape.0 {
                matrix_l.real[r][c] = matrix_u.real[r][c] / matrix_u.real[c][c];
                for e in 0..self.shape.1 {
                    matrix_u.real[r][e] -= matrix_l.real[r][c] * matrix_u.real[c][e];
                }
            }
        }
        matrix_l = &matrix_l + &Matrix::identity(self.shape.0);

        let mut det_l: f64 = matrix_l.real[0][0];
        let mut det_u: f64 = matrix_u.real[0][0];
        for r in 1..matrix_l.shape.0 {
            det_l *= matrix_l.real[r][r];
            det_u *= matrix_u.real[r][r];
        }

        Ok(det_l * det_u)
    }

    pub fn euclidean_distance(self: &Self) -> Result<f64, String> {
        if self.shape.0 != 1 || self.shape.1 != 1 {
            return Err("Value Error: The Matrix should be a vector.".to_string());
        }

        let mut distance: f64 = 0.0;
        for e in 0..self.shape.0 {
            distance += self.real[e][0].powi(2);
        } 
        
        Ok(distance.sqrt())
    }
    pub fn adjoint(self: &mut Matrix) -> Matrix {
        let mut adjoint_matrix: Matrix = Self::zeros(self.shape.0, self.shape.1);
        let mut sign: f64 = 1.0;
        for r in 0..adjoint_matrix.shape.0 {
            for c in 0..adjoint_matrix.shape.1 {
                let sub_matrix: Matrix = self.remove_row(r).unwrap().remove_col(c).unwrap();
                adjoint_matrix.real[r][c] = sign * sub_matrix.determinant().unwrap();
                sign *= -1.0;
            }
        }
        adjoint_matrix.transpose()
    }

    /// Return the inverse matrix of self if have.
    /// Using
    pub fn inverse(self: &Self) -> Result<Matrix, String> {
        if self.shape.0 != self.shape.1 {
            return Err("Value Error: This matrix is not a squared matrix.".to_string());
        } else if self.shape.0 == 0 {
            return Err("Value Error: This matrix is empty.".to_string());
        }

        if self.shape.0 == 1 {
            return Ok(Matrix {
                shape: (1, 1),
                real: vec![vec![1.0 / self.real[0][0]]],
                imaginary: vec![vec![-1.0 / self.real[0][0]]],
            });
        }

        let determinant: f64 = self.determinant()?;
        if determinant == 0.0 {
            return Err("Value Error: This matrix is not invertible".to_string());
        }

        // Get upper triangular form.
        let mut matrix: Matrix = self.clone();
        let mut inverse_matrix: Matrix = Self::identity(self.shape.0);
        for d in 0..matrix.shape.1 {
            // If the pivot is 0.0, swap to non zero.
            if matrix.real[d][d] == 0.0 {
                for r in (d + 1)..matrix.shape.0 {
                    if matrix.real[r][d] != 0.0 {
                        matrix = matrix.swap_row(d, r)?;
                        inverse_matrix = inverse_matrix.swap_row(d, r)?;
                    }
                }
            }

            for r in (d + 1)..matrix.shape.0 {
                let scale: f64 = matrix.real[r][d] / matrix.real[d][d];
                for e in 0..matrix.shape.1 {
                    matrix.real[r][e] -= scale * matrix.real[d][e];
                    inverse_matrix.real[r][e] -= scale * inverse_matrix.real[d][e];
                }
            }
        }

        // To identity
        for d in (0..matrix.shape.1).rev() {
            for r in (0..d).rev() {
                let scale = matrix.real[r][d] / matrix.real[d][d];
                matrix.real[r][d] -= scale * matrix.real[d][d];
                for c in 0..inverse_matrix.shape.1 {
                    inverse_matrix.real[r][c] -= scale * inverse_matrix.real[d][c];
                }
            }
        }

        // Pivots -> 1
        for r in 0..matrix.shape.0 {
            for c in r..matrix.shape.1 {
                if matrix.real[r][c] != 0.0 {
                    let scale: f64 = matrix.real[r][c];
                    for e in c..matrix.shape.1 {
                        matrix.real[r][e] /= scale;
                    }
                    for e in 0..inverse_matrix.shape.1 {
                        inverse_matrix.real[r][e] /= scale;
                    }

                    break;
                }
            }
        }

        Ok(inverse_matrix)
    }

    pub fn normalize(self: &Self) -> Matrix {
        if self.shape.0 == 0 {
            return self.clone();
        }

        let mut result_matrix: Matrix = Matrix::zeros(self.shape.0, self.shape.1);
        let mut min: f64 = self.real[0][0]; 
        let mut max: f64 = self.real[0][0]; 
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                if self.real[r][c] > max {max = self.real[r][c]}
                else if self.real[r][c] < min {min = self.real[r][c]}
            }
        }
        
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.real[r][c] = (self.real[r][c] - max) / (max - min);
            }
        }

        result_matrix
    }

    pub fn is_square(self: &Self) -> bool {
        self.shape.0 == self.shape.1
    }

    pub fn is_upper_triangular(self: &Self) -> bool {
        for r in 1..self.shape.0 {
            for c in 0..r.min(self.shape.1) {
                if self.real[r][c] != 0.0 {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_lower_triangular(self: &Self) -> bool {
        for r in 0..self.shape.0.min(self.shape.1) {
            for c in (r + 1)..self.shape.1 {
                if self.real[r][c] != 0.0 {
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

        for r in 0..self.shape.0 {
            for c in (r + 1)..self.shape.1 {
                if self.real[r][c] != self.real[c][r] {
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

        for d in 1..self.shape.0 {
            if self.real[d][d - 1].powi(2) >= self.real[d][d] {
                return false;
            }
        }

        true
    }

    pub fn calculate_square_error(self: &Self, matrix: &Matrix) -> Result<f64, String> {
        if self.shape.0 != matrix.shape.0 || self.shape.1 != matrix.shape.1 {
            return Err("Input Error: The size of input matrix does not match.".to_string());
        }

        let mut error: f64 = 0.0;
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                error += (self.real[r][c] - matrix.real[r][c]).powi(2);
            }
        }

        Ok(error)
    }

    /// Return the matrix that took square root on each element.
    pub fn square_root(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.real[r][c] = result_matrix.real[r][c].sqrt();
            }
        }

        result_matrix
    }

    /// Return the matrix that took power of 2 on each element.
    pub fn to_powi(self: &Self, power: i32) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.real[r][c] = result_matrix.real[r][c].powi(power);
            }
        }

        result_matrix
    }

    /// Return the upper triangular form of self.
    ///
    /// Eliminate those elements which lay in lower triangular.
    pub fn eliminate_lower_triangular(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        let col_bound = result_matrix.shape.1;
        for r in 1..self.shape.0 {
            for c in 0..r.min(col_bound) {
                result_matrix.real[r][c] = 0.0;
            }
        }

        result_matrix
    }

    /// Return the lower triangular form of self.
    ///
    /// Eliminate those elements which lay in upper triangular.
    pub fn eliminate_upper_triangular(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        let col_bound = result_matrix.shape.1;
        for r in 0..self.shape.0 {
            for c in (r + 1)..self.shape.1 {
                result_matrix.real[r][c] = 0.0;
            }
        }

        result_matrix
    }

    /// Return a matrix only contains the diagonal real.
    pub fn take_diagonal_real(self: &Self) -> Matrix {
        self.eliminate_lower_triangular()
            .eliminate_upper_triangular()
    }

    /// Return a matrix only contains the diagonal real.
    ///
    /// Parameter row is start from 0.
    pub fn take_row(self: &Self, row: usize) -> Result<Matrix, String> {
        if row >= self.shape.0 {
            return Err("Input Error: Parameter row is out of bound.".to_string());
        }

        Ok(Matrix::from_vec(&self.real[row]))
    }

    /// Return a matrix only contains the diagonal real.
    ///
    /// Parameter col is start from 0.
    pub fn take_col(self: &Self, col: usize) -> Result<Matrix, String> {
        if col >= self.shape.1 {
            return Err("Input Error: Parameter col is out of bound.".to_string());
        }

        let mut result_vector: Vec<f64> = Vec::new();
        for r in 0..self.shape.0 {
            result_vector.push(self.real[r][col]);
        }

        Ok(Matrix::from_vec(&result_vector))
    }
}
