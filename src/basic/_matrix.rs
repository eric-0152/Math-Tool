use std::ops::{Add, Mul, Sub, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};
use rand::Rng;
use num_complex::Complex64;
use crate::vector::Vector;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub shape: (usize, usize),
    pub entries: Vec<Vec<Complex64>>,
}




impl Matrix {
    pub fn new(double_vec: &Vec<Vec<Complex64>>) -> Result<Matrix, String> {
        if double_vec.is_empty() {
            return Err("Input Error: This matrix is empty.".to_string());
        }
        
        for r in 1..double_vec.len() {
            if double_vec[r].len() != double_vec[r - 1].len() {
                return Err("Input Error: The vector should be same size in each row.".to_string());
            } 
        }
        
        Ok(Matrix {
            shape: (double_vec.len(), double_vec[0].len()),
            entries: double_vec.clone(),
        })
    }

    /// Return the selected column as a vector.
    pub fn get_column_vector(self: &Self, col: usize) -> Result<Vector, String> {
        if col > self.shape.1 {
            return Err("Input Error: Input col is out of bound.".to_string());
        }

        let mut col_vector: Vec<Complex64> = Vec::new();
        for r in 0..self.shape.0 {
            col_vector.push(self.entries[r][col]);
        }

        Ok(Vector::new(&col_vector))
    }

    /// Return the selected row as a vector.
    pub fn get_row_vector(self: &Self, row: usize) -> Result<Vector, String> {
        if row > self.shape.0 {
            return Err("Input Error: Input col is out of bound.".to_string());
        }

        Ok(Vector::new(&self.entries[row].clone()))
    }

    /// Return the matrix that round to the digit after decimal point.
    pub fn round(self: &Self, digit: usize) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        let scale: f64 = 10_i32.pow(digit as u32) as f64;
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c].re = (scale * result_matrix.entries[r][c].re).round();
                result_matrix.entries[r][c].im = (scale * result_matrix.entries[r][c].im).round();

                if result_matrix.entries[r][c].re >= 1.0 || result_matrix.entries[r][c].re <= -1.0 {
                    result_matrix.entries[r][c].re /= scale;
                } else if result_matrix.entries[r][c].is_nan() {
                    continue;
                } else {
                    result_matrix.entries[r][c].re = 0.0;
                }

                if result_matrix.entries[r][c].im >= 1.0 || result_matrix.entries[r][c].im <= -1.0 {
                    result_matrix.entries[r][c].im /= scale;
                } else if result_matrix.entries[r][c].is_nan() {
                    continue;
                } else {
                    result_matrix.entries[r][c].im = 0.0;
                } 
            }
        }

        result_matrix
    }

    pub fn replace_nan(self: &Self) -> Matrix {
        let mut result_matrix = self.clone();
        for r in 0..result_matrix.shape.0 {
            for c in 0..result_matrix.shape.1 {
                if result_matrix.entries[r][c].re.is_nan() {
                    result_matrix.entries[r][c].re = 0.0;
                } else if result_matrix.entries[r][c].im.is_nan() {
                    result_matrix.entries[r][c].im = 0.0;
                }
            }
        }

        result_matrix
    }

    fn fmt_line(row: &Vec<Complex64>, show_im: bool) -> String {
        let mut string: String = "[".to_string();
        if show_im {
            for e in 0..row.len() {
                if row[e].im >= 0.0 {
                    string.push_str(format!("{:>11?} {:>11} j", row[e].re, format!("+ {:<?}", row[e].im.abs())).as_str());
                } else {
                    string.push_str(format!("{:>11?} {:>11} j", row[e].re, format!("- {:<?}", row[e].im.abs())).as_str());
                }
                
                if e != row.len() - 1 {
                    string.push_str(",");
                }
            }
            string.push_str("]");
        } else {
            for e in 0..row.len() {
                string.push_str(format!("{:11?}", row[e].re).as_str());                
                if e != row.len() - 1 {string.push_str(",")}
            }
            string.push_str("]");
        }

        string
    }

    pub fn fmt(self: &Self, show_im: bool) -> String {
        if self.shape.0 == 0 {
            return format!("[[]], shape: {} x {}", self.shape.0, self.shape.1);
        } else if self.shape.0 == 1 {
            return format!(
                "[{}], shape: {} x {}",
                Self::fmt_line(&self.entries[0], show_im), self.shape.0, self.shape.1
            );
        }

        let mut string: String = format!("[{}", Self::fmt_line(&self.entries[0], show_im));
        string.push('\n');
        for r in 1..(self.shape.0 - 1) {
            string.push_str(format!(" {}", Self::fmt_line(&self.entries[r], show_im)).as_str());
            string.push('\n');
        }
        string.push_str(format!(" {}], shape: {} x {}",
                Self::fmt_line(&self.entries[self.shape.0 - 1], show_im),
                self.shape.0,
                self.shape.1
            ).as_str());
        string.push('\n');
        
        string
    }

    pub fn display(self: &Self) {
        let mut show_im = false;
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                if self.entries[r][c].im != 0.0 {
                    show_im = true;
                    break;
                }
            }
        }
        
        if self.shape.0 == 0 {
            println!("[[]], shape: {} x {}", self.shape.0, self.shape.1);
            return;
        } else if self.shape.0 == 1 {
            println!(
                "[{}], shape: {} x {}",
                Self::fmt_line(&self.entries[0], show_im), self.shape.0, self.shape.1
            );
            return;
        }
        

        println!("[{}", Self::fmt_line(&self.entries[0], show_im));
        for r in 1..(self.shape.0 - 1) {
            println!(" {}", Self::fmt_line(&self.entries[r], show_im));
        }
        println!(
            "{}",
            format!(
                " {}], shape: {} x {}",
                Self::fmt_line(&self.entries[self.shape.0 - 1], show_im),
                self.shape.0,
                self.shape.1
            )
        );
    }

    /// Return a matrix contains all one entries with m rows and n cols.
    pub fn ones(m: usize, n: usize) -> Matrix {
        Matrix {
            shape: (m, n),
            entries: vec![vec![Complex64::ONE; n]; m],
        }
    }

    /// Return a matrix contains all zero entries with m rows and n cols.
    pub fn zeros(m: usize, n: usize) -> Matrix {
        Matrix {
            shape: (m, n),
            entries: vec![vec![Complex64::ZERO; n]; m],
        }
    }

    pub fn identity(m: usize) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, m);
        for d in 0..m {
            result_matrix.entries[d][d] = Complex64::ONE;
        }

        result_matrix
    }

    pub fn random_matrix(m: usize, n: usize, min: f64, max: f64, is_complex: bool) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, n);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        if is_complex {
            for r in 0..m {
                for c in 0..n {
                    result_matrix.entries[r][c].re = generator.random_range(min..max);
                    result_matrix.entries[r][c].im = generator.random_range(min..max);
                }
            }
        } else {
            for r in 0..m {
                for c in 0..n {
                    result_matrix.entries[r][c].re = generator.random_range(min..max);
                }
            }
        }

        result_matrix
    }

    /// Return the upper triangular matrix or self.
    pub fn random_upper_triangular(m: usize, n: usize, min: f64, max: f64, is_complex: bool) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, n);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        if is_complex {
            for r in 0..m {
                for c in r..n {
                    result_matrix.entries[r][c].re = generator.random_range(min..max);
                    result_matrix.entries[r][c].im = generator.random_range(min..max);
                }
            }
        } else {
            for r in 0..m {
                for c in r..n {
                    result_matrix.entries[r][c].re = generator.random_range(min..max);
                }
            }
        }

        result_matrix
    }

    /// Return the lower triangular matrix or self.
    pub fn random_lower_triangular(m: usize, n: usize, min: f64, max: f64, is_complex: bool) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, n);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        if is_complex {
            for r in 0..m {
                for c in 0..(r + 1).min(n) {
                    result_matrix.entries[r][c].re = generator.random_range(min..max);
                    result_matrix.entries[r][c].im = generator.random_range(min..max);
                }
            }
        } else {
            for r in 0..m {
                for c in 0..(r + 1).min(n) {
                    result_matrix.entries[r][c].re = generator.random_range(min..max);
                }
            }
        }

        result_matrix
    }

    pub fn random_diagonal_matrix(m: usize, min: f64, max: f64, is_complex: bool) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(m, m);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        if is_complex {
            for r in 0..m {
                result_matrix.entries[r][r].re = generator.random_range(min..max);
                result_matrix.entries[r][r].im = generator.random_range(min..max);
            }
            
        } else {
            for r in 0..m {
                result_matrix.entries[r][r].re = generator.random_range(min..max);
            }
        }

        result_matrix
    }

    pub fn random_symmetric_matrix(m: usize, min: f64, max: f64, is_complex: bool) -> Matrix {
        let mut result_matrix: Matrix = Self::random_upper_triangular(m, m, min, max, is_complex);
        for r in 0..m {
            for c in (r + 1)..m {
                result_matrix.entries[c][r] = result_matrix.entries[r][c];
            }
        }

        result_matrix
    }

    pub fn arrange(start: f64, end: f64, step: f64) -> Result<Matrix, String> {
        if start < end {
            if step < 0.0 {
                return Err("Input Error: Parameter step should be positive when start < end.".to_string());
            } else {
                let mut vector: Vec<Complex64> = Vec::new();
                let mut current = start;
                while current <= end {
                    vector.push(Complex64::new(current, 0.0));
                    current += step;
                }
        
                Ok(Matrix::new(&vec![vector]).unwrap().transpose())                
            }
        } else {
            if step > 0.0 {
                return Err("Input Error: Parameter step should be negative when start > end.".to_string());
            } else {
                let mut vector: Vec<Complex64> = Vec::new();
                let mut current = start;
                while current >= end {
                    vector.push(Complex64::new(current, 0.0));
                    current += step;
                }
                
                Ok(Matrix::new(&vec![vector]).unwrap().transpose())
            }
        }
    }
    
    pub fn linspace(start: f64, end: f64, size: usize) -> Matrix {
        let mut vector: Vec<Complex64> = Vec::new();
        let mut current = start;
        let step = (end - start) / size as f64;
        for _ in 0..size {
            vector.push(Complex64::new(current, 0.0));
            current += step;
        }

        Matrix::new(&vec![vector]).unwrap().transpose()
    }
    
    /// Sum up all the entries in matrix.
    pub fn entries_sum(self: &Self) -> Complex64 {
        let mut entries_sum: Complex64 = Complex64::ZERO;
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                entries_sum += self.entries[r][c];
            }
        }

        entries_sum
    }

    pub fn trace(self: &Self) -> Complex64 {
        let mut entries_sum: Complex64 = Complex64::ZERO;
        for r in 0..self.shape.0 {
            entries_sum += self.entries[r][r];
        }

        entries_sum
    }

    /// Append a matrix along the axis.
    ///
    /// &emsp; If axis == 0 : append matirx to the bottom.
    ///   
    /// &emsp; If axis == 1 : append matirx to the right.
    pub fn append_matrix(self: &Self, matrix: &Matrix, axis: usize) -> Result<Matrix, String> {
        if self.shape.0 == 0 {
            match axis {
                x if x == 0 || x == 1 => return Ok(matrix.clone()),
                _ => return Err("Input Error: Input axis is not valid.".to_string()),
            }
        }

        match axis {
            x if x == 0 => {
                if self.shape.1 != matrix.shape.1 {
                    return Err("Input Error: The size of column does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                for r in 0..matrix.shape.0 {
                    result_matrix.entries.push(matrix.entries[r].clone());
                }
                result_matrix.shape.0 += matrix.shape.0;

                return Ok(result_matrix);
            }

            x if x == 1 => {
                if self.shape.0 != matrix.shape.0 {
                    return Err("Input Error: The size of row does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                for r in 0..matrix.shape.0 {
                    for c in 0..matrix.shape.1 {
                        result_matrix.entries[r].push(matrix.entries[r][c]);
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
    
    /// Append a vector along the axis.
    ///
    /// &emsp; If axis == 0 : append vector to the bottom.
    ///   
    /// &emsp; If axis == 1 : append vector to the right.
    pub fn append_vector(self: &Self, vector: &Vector, axis: usize) -> Result<Matrix, String> {
        if self.shape.0 == 0 {
            match axis {
                x if x == 0 || x == 1 => return Ok(vector.as_matrix()),
                _ => return Err("Input Error: Input axis is not valid.".to_string()),
            }
        }

        match axis {
            x if x == 0 => {
                if self.shape.1 != vector.size {
                    return Err("Input Error: The size of vector does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                result_matrix.entries.push(vector.entries.clone());
                
                result_matrix.shape.0 += 1;
                return Ok(result_matrix);
            }

            x if x == 1 => {
                if self.shape.0 != vector.size {
                    return Err("Input Error: The size of vector does not match .".to_string());
                }

                let mut result_matrix: Matrix = self.clone();
                for r in 0..result_matrix.shape.0 {
                    result_matrix.entries[r].push(vector.entries[r]);
                }
                
                result_matrix.shape.1 += 1;
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

        let mut entries_element: Vec<Complex64> = Vec::new();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                entries_element.push(self.entries[r][c]);
            }
        }

        entries_element.reverse();
        let mut result_matrix: Matrix = Self::zeros(shape.0, shape.1);
        for r in 0..shape.0 {
            for c in 0..shape.1 {
                result_matrix.entries[r][c] = entries_element.pop().unwrap();
            }
        }

        Ok(result_matrix)
    }

    pub fn transpose(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Self::zeros(self.shape.1, self.shape.0);
        for r in 0..result_matrix.shape.0 {
            for c in 0..result_matrix.shape.1 {
                result_matrix.entries[r][c].re = self.entries[c][r].re;
                result_matrix.entries[r][c].im = -self.entries[c][r].im;
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
        result_matrix.entries.remove(row);

        Ok(result_matrix)
    }

    /// Return the matrix whithout the row.
    ///
    /// Parameter row is start from 0.
    pub fn remove_col(self: &Self, col: usize) -> Result<Matrix, String> {
        if col >= self.shape.1 {
            return Err("Input Error: Input col is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        result_matrix.shape.1 -= 1;
        for r in 0..self.shape.0 {
            result_matrix.entries[r].remove(col);
        }

        Ok(result_matrix)
    }

    pub fn swap_row(self: &Self, row1: usize, row2: usize) -> Result<Matrix, String> {
        if row1 >= self.shape.0 || row2 >= self.shape.0 {
            return Err("Input Error: Input row1 or row2 is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        result_matrix.entries[row1] = self.entries[row2].clone();
        result_matrix.entries[row2] = self.entries[row1].clone();

        Ok(result_matrix)
    }

    pub fn swap_column(self: &Self, col1: usize, col2: usize) -> Result<Matrix, String> {
        if col1 >= self.shape.1 || col2 >= self.shape.1 {
            return Err("Input Error: Input row1 or row2 is out of bound".to_string());
        }

        let mut result_matrix: Matrix = self.clone();
        for r in 0..result_matrix.shape.0 {
            result_matrix.entries[r][col1] = self.entries[r][col2];
            result_matrix.entries[r][col2] = self.entries[r][col1];
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
    pub fn determinant(self: &Self) -> Result<Complex64, String> {
        if !self.is_square() {
            return Err("Value Error: This matrix is not a square matrix.".to_string());
        }

        let mut matrix_u: Matrix = self.clone();
        let mut matrix_l: Matrix = Matrix::zeros(self.shape.0, self.shape.0);
        let mut permutation: Matrix = Matrix::identity(self.shape.0);
        for c in 0..self.shape.1 {
            // If the pivot is 0.0, swap to non zero.
            let mut is_swap = false;
            if matrix_u.entries[c][c] == Complex64::ZERO {
                for r in (c + 1)..matrix_u.shape.0 {
                    if matrix_u.entries[r][c] != Complex64::ZERO {
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
                matrix_l.entries[r][c] = matrix_u.entries[r][c] / matrix_u.entries[c][c];
                for e in 0..self.shape.1 {
                    let row_element: Complex64 = matrix_u.entries[c][e];
                    matrix_u.entries[r][e] -= matrix_l.entries[r][c] * row_element;
                }
            }
        }
        matrix_l = &matrix_l + &Matrix::identity(self.shape.0);

        let mut det_l: Complex64 = matrix_l.entries[0][0];
        let mut det_u: Complex64 = matrix_u.entries[0][0];
        for r in 1..matrix_l.shape.0 {
            det_l *= matrix_l.entries[r][r];
            det_u *= matrix_u.entries[r][r];
        }

        Ok(det_l * det_u)
    }

    pub fn adjoint(self: &mut Matrix) -> Matrix {
        let mut adjoint_matrix: Matrix = Self::zeros(self.shape.0, self.shape.1);
        let mut sign: f64 = 1.0;
        for r in 0..adjoint_matrix.shape.0 {
            for c in 0..adjoint_matrix.shape.1 {
                let sub_matrix: Matrix = self.remove_row(r).unwrap().remove_col(c).unwrap();
                adjoint_matrix.entries[r][c] = sign * sub_matrix.determinant().unwrap();
                sign *= -1.0;
            }
        }
        adjoint_matrix.transpose()
    }

    /// Return the inverse matrix of self if have.
    pub fn inverse(self: &Self) -> Result<Matrix, String> {
        if self.shape.0 != self.shape.1 {
            return Err("Value Error: This matrix is not a squared matrix.".to_string());
        } else if self.shape.0 == 0 {
            return Err("Value Error: This matrix is empty.".to_string());
        }

        if self.shape.0 == 1 {
            return Ok(Matrix {
                shape: (1, 1),
                entries: vec![vec![1.0 / self.entries[0][0]]],
            });
        }

        let determinant: Complex64 = self.determinant()?;
        if determinant == Complex64::ZERO {
            return Err("Value Error: This matrix is not invertible".to_string());
        }

        // Get upper triangular form.
        let mut matrix: Matrix = self.clone();
        let mut inverse_matrix: Matrix = Self::identity(self.shape.0);
        for d in 0..matrix.shape.1 {
            // If the pivot is 0.0, swap to non zero.
            if matrix.entries[d][d] == Complex64::ZERO {
                for r in (d + 1)..matrix.shape.0 {
                    if matrix.entries[r][d] != Complex64::ZERO {
                        matrix = matrix.swap_row(d, r)?;
                        inverse_matrix = inverse_matrix.swap_row(d, r)?;
                    }
                }
            }

            for r in (d + 1)..matrix.shape.0 {
                let scale: Complex64 = matrix.entries[r][d] / matrix.entries[d][d];
                for e in 0..matrix.shape.1 {
                    let sub_element: Complex64 = matrix.entries[d][e];
                    matrix.entries[r][e] -= scale * sub_element;
                    let inv_sub_element: Complex64 = inverse_matrix.entries[d][e];
                    inverse_matrix.entries[r][e] -= scale * inv_sub_element;
                }
            }
        }

        // To identity
        for d in (0..matrix.shape.1).rev() {
            for r in (0..d).rev() {
                let scale = matrix.entries[r][d] / matrix.entries[d][d];
                let diag_element: Complex64 = matrix.entries[d][d];
                matrix.entries[r][d] -= scale * diag_element;
                for c in 0..inverse_matrix.shape.1 {
                    let row_element: Complex64 = inverse_matrix.entries[d][c];
                    inverse_matrix.entries[r][c] -= scale * row_element;
                }
            }
        }

        // Pivots -> 1
        for r in 0..matrix.shape.0 {
            for c in r..matrix.shape.1 {
                if matrix.entries[r][c] != Complex64::ZERO {
                    let scale: Complex64 = matrix.entries[r][c];
                    for e in c..matrix.shape.1 {
                        matrix.entries[r][e] /= scale;
                    }
                    for e in 0..inverse_matrix.shape.1 {
                        inverse_matrix.entries[r][e] /= scale;
                    }

                    break;
                }
            }
        }

        Ok(inverse_matrix)
    }

    pub fn is_square(self: &Self) -> bool {
        self.shape.0 == self.shape.1
    }

    pub fn is_upper_triangular(self: &Self) -> bool {
        for r in 1..self.shape.0 {
            for c in 0..r.min(self.shape.1) {
                if self.entries[r][c] != Complex64::ZERO {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_lower_triangular(self: &Self) -> bool {
        for r in 0..self.shape.0.min(self.shape.1) {
            for c in (r + 1)..self.shape.1 {
                if self.entries[r][c] != Complex64::ZERO {
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
                if d != Complex64::ZERO {
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

        // for d in 1..self.shape.0 {
        //     if self.entries[d][d - 1].powi(2) >= self.entries[d][d] {
        //         return false;
        //     }
        // }

        true
    }

    /// Return the matrix that took square root on each element.
    pub fn square_root(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c] = result_matrix.entries[r][c].sqrt();
            }
        }

        result_matrix
    }

    /// Return the matrix that took power on each element.
    pub fn pow(self: &Self, power: f64) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c] = result_matrix.entries[r][c].powf(power);
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
                result_matrix.entries[r][c] = Complex64::ZERO;
            }
        }

        result_matrix
    }

    /// Return the lower triangular form of self.
    ///
    /// Eliminate those elements which lay in upper triangular.
    pub fn eliminate_upper_triangular(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in (r + 1)..self.shape.1 {
                result_matrix.entries[r][c] = Complex64::ZERO;
            }
        }

        result_matrix
    }

    /// Return a matrix only remain the diagonal entries.
    pub fn take_diagonal(self: &Self) -> Matrix {
        self.eliminate_lower_triangular()
            .eliminate_upper_triangular()
    }
}




impl Add<&Matrix> for &Matrix {
    type Output = Matrix;    
    #[inline]
    fn add(self: Self, matrix: &Matrix) -> Matrix {
        if self.shape != matrix.shape {
            panic!("Matrix shapes do not match");
        }
        
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c] += matrix.entries[r][c];
            }
        }

        result_matrix
    }
}

impl Add<f64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn add(self: Self, constant: f64) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c] += constant;
            }
        }

        result_matrix
    }
}
impl Add<Complex64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn add(self: Self, constant: Complex64) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c] += constant;
            }
        }

        result_matrix
    }
}
impl Add<&Matrix> for f64 {
    type Output = Matrix;
    #[inline]
    fn add(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = matrix.clone();
        for r in 0..matrix.shape.0 {
            for c in 0..matrix.shape.1 {
                result_matrix.entries[r][c] += self;
            }
        }

        result_matrix
    }
}
impl Add<&Matrix> for Complex64 {
    type Output = Matrix;
    #[inline]
    fn add(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = matrix.clone();
        for r in 0..matrix.shape.0 {
            for c in 0..matrix.shape.1 {
                result_matrix.entries[r][c] += self;
            }
        }

        result_matrix
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    #[inline]
    fn sub(self: Self, matrix: &Matrix) -> Matrix {
        self + &(-1.0 * matrix)
    }
}
impl Sub<f64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn sub(self: Self, float: f64) -> Matrix {
        self + (-float)
    }
}
impl Sub<Complex64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn sub(self: Self, complex: Complex64) -> Matrix {
        self + -complex
    }
}
impl Sub<&Matrix> for f64 {
    type Output = Matrix;
    #[inline]
    fn sub(self: Self, matrix: &Matrix) -> Matrix {
        self + &(-1.0 * matrix)
    }
}
impl Sub<&Matrix> for Complex64 {
    type Output = Matrix;
    #[inline]
    fn sub(self: Self, matrix: &Matrix) -> Matrix {
        self + &(-1.0 * matrix)
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self: Self, scalar: f64) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c] *= scalar;
            }

        }

        result_matrix
    }
}
impl Mul<Complex64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self: Self, scalar: Complex64) -> Matrix {
        let mut result_matrix: Matrix = self.clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_matrix.entries[r][c] *= scalar;
            }

        }

        result_matrix
    }
}
impl Mul<&Matrix> for f64 {
    type Output = Matrix;
    #[inline]
    fn mul(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = matrix.clone();
        for r in 0..matrix.shape.0 {
            for c in 0..matrix.shape.1 {
                result_matrix.entries[r][c] *= self;
            }

        }

        result_matrix
    }
}
impl Mul<&Matrix> for Complex64 {
    type Output = Matrix;
    #[inline]
    fn mul(self: Self, matrix: &Matrix) -> Matrix {
        let mut result_matrix: Matrix = matrix.clone();
        for r in 0..matrix.shape.0 {
            for c in 0..matrix.shape.1 {
                result_matrix.entries[r][c] *= self;
            }

        }

        result_matrix
    }
}
impl Mul for &Matrix {
    type Output = Matrix;
    #[inline]
    fn mul(self: Self, matrix: &Matrix) -> Matrix {
        if self.shape.1 != matrix.shape.0 {
            panic!("Matrix shapes do not match");
        } 
        
        let mut result_matrix: Matrix = Matrix::zeros(self.shape.0, matrix.shape.1).clone();
        for r in 0..result_matrix.shape.0 {
            for c in 0..result_matrix.shape.1 {
                for e in 0..self.shape.1 {
                    result_matrix.entries[r][c] += self.entries[r][e] * matrix.entries[e][c];
                }
            }
        }

        result_matrix
    }
}
impl Mul<&Vector> for &Matrix {
    type Output = Vector;
    #[inline]
    fn mul(self: Self, vector: &Vector) -> Vector {
        if self.shape.1 != vector.size {
            panic!("Matrix column and vector size do not match.");
        }
        let mut result_vector: Vector = Vector::zeros(self.shape.0).clone();
        for r in 0..self.shape.0 {
            for c in 0..self.shape.1 {
                result_vector.entries[r] += self.entries[r][c] * vector.entries[c];
            }
        }

        result_vector
    }
}
impl Div<f64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn div(self: Self, scalar: f64) -> Matrix {
        if scalar != 0.0 {
            self * (1.0 / scalar)
        } else {
            panic!("Division by zero");
        }
    }
}
impl Div<Complex64> for &Matrix {
    type Output = Matrix;
    #[inline]
    fn div(self: Self, scalar: Complex64) -> Matrix {
        if scalar != Complex64::ZERO {
            self * (1.0 / scalar)
        } else {
            panic!("Division by zero");
        }
    }
}
impl Div<&Matrix> for f64 {
    type Output = Matrix;
    #[inline]
    fn div(self: Self, matrix: &Matrix) -> Matrix {
        if self != 0.0 {
            matrix * (1.0 / self)
        } else {
            panic!("Division by zero");
        }
    }
}
impl Div<&Matrix> for Complex64 {
    type Output = Matrix;
    #[inline]
    fn div(self: Self, matrix: &Matrix) -> Matrix {
        if self != Complex64::ZERO {
            matrix * (1.0 / self)
        } else {
            panic!("Division by zero");
        }
    }
}


impl AddAssign for Matrix {
    #[inline]
    fn add_assign(&mut self, other: Matrix) {
        *self = &self.clone() + &other;
    }
}
impl AddAssign<f64> for Matrix {
    #[inline]
    fn add_assign(&mut self, other: f64) {
        *self = &self.clone() + other;
    }
}
impl AddAssign<Complex64> for Matrix {
    #[inline]
    fn add_assign(&mut self, other: Complex64) {
        *self = &self.clone() + other;
    }
}
impl SubAssign for Matrix {
    #[inline]
    fn sub_assign(&mut self, other: Matrix) {
        *self = &self.clone() - &other;
    }
}
impl SubAssign<f64> for Matrix {
    #[inline]
    fn sub_assign(&mut self, other: f64) {
        *self = &self.clone() - other;
    }
}
impl SubAssign<Complex64> for Matrix {
    #[inline]
    fn sub_assign(&mut self, other: Complex64) {
        *self = &self.clone() - other;
    }
}
impl Neg for Matrix {
    type Output = Matrix;
    #[inline]
    fn neg(self: Self) -> Matrix {
        -1.0 * &self.clone()
    }
}
impl MulAssign for Matrix {
    #[inline]
    fn mul_assign(&mut self, other: Matrix) {
        *self = &self.clone() * &other;
    }
}
impl MulAssign<f64> for Matrix {
    #[inline]
    fn mul_assign(&mut self, other: f64) {
        *self = &self.clone() * other;
    }
}
impl MulAssign<Complex64> for Matrix {
    #[inline]
    fn mul_assign(&mut self, other: Complex64) {
        *self = &self.clone() * other;
    }
}
impl DivAssign<f64> for Matrix {
    #[inline]
    fn div_assign(&mut self, other: f64) {
        *self = &self.clone() / other;
    }
}
impl DivAssign<Complex64> for Matrix {
    #[inline]
    fn div_assign(&mut self, other: Complex64) {
        *self = &self.clone() / other;
    }
}
