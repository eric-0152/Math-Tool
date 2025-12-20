use crate::matrix::Matrix;
use rand::Rng;

#[derive(Clone)]
pub struct Vector {
    pub entries: Vec<f64>,
    pub size: usize,
}

impl Vector {
    pub fn from_vec(vector: &Vec<f64>) -> Vector {
        Vector {
            entries: vector.clone(),
            size: vector.len(),
        }
    }

    /// Return the vector that round to the digit after decimal point.
    pub fn round(self: &Self, digit: u32) -> Vector {
        let scale: f64 = 10_i32.pow(digit as u32) as f64;
        let mut result_vector: Vector = Self::zeros(self.size);

        for s in 0..self.size {
            result_vector.entries[s] = (scale * self.entries[s]).round();

            if result_vector.entries[s] >= 1.0 || result_vector.entries[s] <= -1.0 {
                result_vector.entries[s] /= scale;
            } else if result_vector.entries[s].is_nan() {
                continue;
            } else {
                result_vector.entries[s] = 0.0;
            }
        }

        result_vector
    }

    pub fn replace_nan(self: &Self) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..result_vector.size {
            if result_vector.entries[e].is_nan() {
                result_vector.entries[e] = 0.0;
            
            }
        }

        result_vector
    }

    pub fn display(self: &Self) {
        println!("Vector:  {:8?}, size: {}", self.entries, self.size);
    }

    /// Return a vector contains all zero entries with size m.
    pub fn zeros(m: usize) -> Vector {
        Vector {
            entries: vec![0.0; m],
            size: m,
        }
    }

    /// Return a vector contains all one entries with size m.
    pub fn ones(m: usize) -> Vector {
        Vector {
            entries: vec![1.0; m],
            size: m,
        }
    }

    /// Return a vector contains a arithmetic sequence. [[start, end]]
    pub fn arithmetic_sequence(start: f64, end: f64, interval: f64) -> Vector {
        let mut result_vector: Vec<f64> = vec![start];
        let mut current_element: f64 = start;
        let mut size: usize = 1;
        while current_element < end {
            current_element += interval;
            result_vector.push(current_element);
            size += 1;
        }

        Vector {
            entries: result_vector,
            size: size,
        }
    }

    /// Return a vector contains a arithmetic sequence. [start, end)
    pub fn fixed_size_arithmetic_sequence(start: f64, end: f64, vector_size: usize) -> Vector {
        let mut result_vector: Vec<f64> = vec![start];
        let mut current_element: f64 = start;
        let mut interval: f64 = (end - start) / vector_size as f64;
        let mut step: usize = 1;
        while step < vector_size {
            current_element += interval;
            result_vector.push(current_element);
            step += 1;
        }

        Vector {
            entries: result_vector,
            size: vector_size,
        }
    }

    pub fn random_vector(m: usize, min: f64, max: f64) -> Vector {
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        let mut entries: Vec<f64> = Vec::new();
        for _ in 0..m {
            entries.push(generator.random_range(min..max));
        }
        Vector {
            entries: entries,
            size: m,
        }
    }

    /// Sum up all the entries in vector.
    pub fn entries_sum(self: &Self) -> f64 {
        let mut summation: f64 = 0.0;
        for e in 0..self.size {
            summation += self.entries[e];
        }

        summation
    }

    pub fn clone(self: &Self) -> Vector {
        Vector {
            entries: self.entries.clone(),
            size: self.size,
        }
    }

    pub fn reverse(self: &Self) -> Vector {
        let mut new_entries: Vec<f64> = self.entries.clone();
        new_entries.reverse();

        Vector {
            entries: new_entries,
            size: self.size,
        }
    }

    pub fn transpose(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Matrix::zeros(1, self.size);
        for r in 0..result_matrix.col {
            result_matrix.entries[0][r] = self.entries[r];
        }

        result_matrix
    }

    pub fn append_Vector(self: &Self, vector: &Vector) -> Vector {
        let mut result_vector = self.clone();
        for e in 0..vector.size {
            result_vector.entries.push(vector.entries[e]);
        }
        result_vector.size += vector.size;

        result_vector
    }

    /// Transform a Vector into a Matrix.
    ///
    /// If axis == 0 : vector as a row.
    ///
    /// If axis == 1 : vector as a column.
    pub fn to_Matrix(self: &Self, axis: usize) -> Result<Matrix, String> {
        match axis {
            x if x == 0 => {
                return Ok(Matrix {
                    entries: vec![self.entries.clone()],
                    row: 1,
                    col: self.size,
                });
            }

            x if x == 1 => {
                return Ok(Matrix {
                    entries: self.transpose().transpose().entries,
                    row: self.size,
                    col: 1,
                });
            }

            _ => {
                return Err("Input Error: Input axis is not valid.".to_string());
            }
        }
    }

    pub fn swap_element(self: &Self, a: usize, b: usize) -> Result<Vector, String> {
        if a >= self.size || b >= self.size {
            return Err("Input Error: The input a or b is out of bound.".to_string());
        }

        let mut result_Vector: Vector = self.clone();
        result_Vector.entries[a] = self.entries[b];
        result_Vector.entries[b] = self.entries[a];

        Ok(result_Vector)
    }

    pub fn remove(self: &Self, index: usize) -> Result<Vector, String> {
        if index >= self.size {
            return Err("Input Error: The input index is out of bound.".to_string());
        }

        let mut entrie_vec: Vec<f64> = self.entries.clone();
        entrie_vec.remove(index);
        
        Ok(Vector::from_vec(&entrie_vec))
    }

    /// Swap the element according to the order of permutaion matrix.
    pub fn swap_with_permutation(self: &Self, permutation: &Matrix) -> Result<Vector, String> {
        if self.size != permutation.row {
            return Err(
                "Input Error: The row size of permutation matrix does not match".to_string(),
            );
        }

        let mut result_vector: Vector = self.clone();
        let mut order: Matrix = permutation.clone();
        for r in 0..order.row {
            if order.entries[r][r] != 1.0 {
                for bottom_r in (r + 1)..order.row {
                    if order.entries[bottom_r][r] == 1.0 {
                        order = order.swap_row(r, bottom_r).unwrap();
                        result_vector = result_vector.swap_element(r, bottom_r).unwrap();
                    }
                }
            }
        }

        Ok(result_vector)
    }

    /// Add two vector element-wise.
    pub fn add_Vector(self: &Self, vector: &Vector) -> Result<Vector, String> {
        if self.size != vector.size {
            return Err("Input Error: The size of input vector does not match.".to_string());
        }

        let mut result_vector: Vector = self.clone();
        for s in 0..self.size {
            result_vector.entries[s] += vector.entries[s];
        }

        Ok(result_vector)
    }

    /// Substract two vector element-wise.
    pub fn substract_Vector(self: &Self, vector: Vector) -> Result<Vector, String> {
        if self.size != vector.size {
            return Err("Input Error: The size of input vector does not match.".to_string());
        }

        let mut result_vector: Vector = self.clone();
        for s in 0..self.size {
            result_vector.entries[s] -= vector.entries[s];
        }

        Ok(result_vector)
    }

    pub fn multiply_scalar(self: &Self, scalar: f64) -> Vector {
        let mut result_vector: Vector = Self::zeros(self.size);
        for s in 0..self.size {
            result_vector.entries[s] = scalar * self.entries[s];
        }

        result_vector
    }

    pub fn inner_product(self: &Self, vector: &Vector) -> Result<f64, String> {
        if self.size != vector.size {
            return Err("Input Error: The size of two vectors do not match.".to_string());
        }

        let mut result: f64 = 0.0;
        for e in 0..self.size {
            result += self.entries[e] * vector.entries[e];
        }

        Ok(result)
    }

    /// Compute cross peoduct for two 3-dimension vectors.
    pub fn cross_product(self: &Vector, vector: &Vector) -> Result<Vector, String> {
        if self.size != vector.size || self.size != 3 {
            return Err("Input Error: The size of input vectors should be 3.".to_string());
        }

        let mut result_vector: Vector = Self::zeros(3);
        let mut matrix: Matrix =
            Matrix::from_double_vec(&vec![self.entries.clone(), vector.entries.clone()]);
        for d in 0..3 {
            result_vector.entries[d] = matrix.remove_col(d).unwrap().determinant().unwrap();
        }

        Ok(result_vector)
    }

    pub fn euclidean_distance(self: &Self) -> f64 {
        let mut disatnce: f64 = 0.0;

        for s in 0..self.size {
            disatnce += self.entries[s] * self.entries[s];
        }

        disatnce.sqrt()
    }

    pub fn to_diagonal(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Matrix::zeros(self.size, self.size);
        for d in 0..self.size {
            result_matrix.entries[d][d] = self.entries[d];
        }

        result_matrix
    }

    pub fn normalize(self: &Self) -> Vector {
        let norm: f64 = self.euclidean_distance();
        let mut normal_vector: Vector = self.clone();
        for e in 0..self.size {
            normal_vector.entries[e] = normal_vector.entries[e] / norm;
        }

        normal_vector
    }

    pub fn square_root(self: &Self) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..self.size {
            result_vector.entries[e] = result_vector.entries[e].sqrt();
        }

        result_vector
    }
}
