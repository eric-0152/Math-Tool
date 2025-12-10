use crate::matrix::Matrix;
use crate::vector::Vector;


/// For shift_qr_algorithm
fn similar_matrix(matrix: &Matrix) -> Result<Matrix, String> {
    match matrix.qr_decomposition() {
        Ok(tuple) => {
            Ok(tuple.1.multiply_Matrix(&tuple.0).unwrap())
        }

        Err(error_msg) => {
            Err(error_msg)
        }
    }
}

/// For SVD
fn shift_qr_algorithm(matrix: &Matrix, max_iter: i32, error_thershold: f64) -> Result<(Vector, f64), String> {
    match similar_matrix(matrix) {
        Ok(mut matrix_s) => {
            let last_row_idx: usize = matrix_s.row - 1;
            let last_col_idx: usize = matrix_s.col - 1;
            let matrix_size: usize = matrix_s.row;
            let mut shift: Matrix = Matrix::zeros(0, 0);
            let mut last_eigenvalue: f64 = matrix_s.entries[last_row_idx][last_col_idx];
            let mut difference: f64 = 1.0;
            let mut step: i32 = 0;
            while difference > error_thershold && step < max_iter {
                shift = Matrix::identity(matrix_size). multiply_scalar(&last_eigenvalue);
                matrix_s = matrix_s.substract_Matrix(&shift).unwrap();
                matrix_s = similar_matrix(&matrix_s).unwrap().add_Matrix(&shift).unwrap();
                difference = (matrix_s.entries[last_row_idx][last_col_idx] - last_eigenvalue).abs();
                last_eigenvalue = matrix_s.entries[last_row_idx][last_col_idx];
                step += 1;
            }

            let mut eigenvalue: Vec<f64> = Vec::new();
            for d in 0..matrix_s.row {
                eigenvalue.push(matrix_s.entries[d][d]);
            }

            Ok((Vector::from_vec(&eigenvalue), difference))
        }

        Err(error_msg) => {
            Err(error_msg)
        }
    }
}

impl Matrix {
    /// Return a tuple (***L, U, P***).
    ///  
    /// <br>
    ///
    /// ### *LU* Decomposition:
    /// &emsp; ***A = LU***, ***L*** is lower triagular, and ***U*** is upper triangular.
    ///
    /// <br>
    ///
    /// The algorithm will swap rows if needed (diagnal has 0), if the order of rows is
    /// important, use swap_with_permutation() with P to yield the correct order:
    ///
    /// &emsp; ***A = (LU).swap_with_permutation(P)***
    pub fn lu_decomposition(self: &Self) -> (Matrix, Matrix, Matrix) {
        let mut matrix_u: Matrix = self.clone();
        let mut matrix_l: Matrix = Matrix::zeros(self.row, self.row);
        let mut permutation: Matrix = Matrix::identity(self.row);
        for c in 0..self.col.min(self.row) {
            // If the pivot is 0.0, swap to non zero.
            if matrix_u.entries[c][c] == 0.0 {
                let mut is_swap = false;
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

        (matrix_l, matrix_u, permutation)
    }

    /// Return a tuple (***L, D, V, P***).
    ///
    /// <br>
    ///
    /// ### *LDV* Decomposition:
    /// &emsp; ***A = LDV***, ***L*** is lower triagular, ***D*** is diagonal, and ***V***
    /// is upper triangular.  
    ///
    /// <br>
    ///
    /// The algorithm will swap rows if needed (diagnal has 0), if the order of rows is     
    /// important, use swap_with_permutation() with P to yield the correct order:
    ///
    /// &emsp; ***A = LDV.swap_with_permutation(***P***)
    pub fn ldv_decomposition(self: &Self) -> Result<(Matrix, Matrix, Matrix, Matrix), String> {
        let tuple = self.lu_decomposition();
        let matrix_l: Matrix = tuple.0;
        let matrix_u: Matrix = tuple.1;
        let permutation: Matrix = tuple.2;
        let mut matrix_d: Matrix = Matrix::identity(self.row);
        for d in 0..self.row.min(self.col) {
            matrix_d.entries[d][d] = matrix_u.entries[d][d];
        }

        match matrix_d.inverse() {
            Ok(inverse_d) => {
                let matrix_v: Matrix = inverse_d.multiply_Matrix(&matrix_u).unwrap();
                Ok((matrix_l, matrix_d, matrix_v, permutation))
            }

            Err(err_msg) => Err(err_msg),
        }
    }

    /// Return a tuple (***L, L^T***).
    ///
    /// ### Cholesky Decomposition:
    /// &nbsp; ***A = LL^T***.
    pub fn cholesky_decomposition(self: &Self) -> Result<(Matrix, Matrix), String> {
        if !self.is_positive_definite() {
            return Err("Value Error: This matrix is not a positive definite matrix.".to_string());
        }

        let mut matrix_l: Matrix = Matrix::zeros(self.row, self.col);
        for r in 0..matrix_l.row {
            for c in 0..(r + 1) {
                let mut summation: f64 = 0.0;
                if r == c {
                    for e in 0..c {
                        summation += matrix_l.entries[c][e].powi(2);
                    }
                    matrix_l.entries[r][c] = (self.entries[c][c] - summation).sqrt();
                } else {
                    for e in 0..c {
                        summation += matrix_l.entries[r][e] * matrix_l.entries[c][e];
                    }
                    matrix_l.entries[r][c] =
                        (self.entries[r][c] - summation) / matrix_l.entries[c][c];
                }
            }
        }

        Ok((matrix_l.clone(), matrix_l.transpose()))
    }

    /// Return a tuple (***L, D, L^T***).
    ///
    /// ### ***LDLT*** Decomposition:
    /// &nbsp; ***A = CC^T*** (from Cholesky decomposition) = ***LD^(1/2) @ (LD^(1/2))^T***
    /// = ***L @ D^(1/2)^2 @ L^T = LDL^T***.
    pub fn ldlt_decomposition(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        match self.cholesky_decomposition() {
            Ok(tuple) => {
                let matrix_c: Matrix = tuple.0;
                let mut matrix_d: Matrix = Self::identity(self.row);
                let mut matrix_l: Matrix = matrix_c.clone();
                for d in 0..matrix_l.col {
                    matrix_d.entries[d][d] = matrix_c.entries[d][d].powi(2);
                    let inverse_sqrt_diagnol: f64 = matrix_d.entries[d][d].sqrt();
                    for r in d..matrix_l.row {
                        matrix_l.entries[r][d] = matrix_c.entries[r][d] / inverse_sqrt_diagnol;
                    }
                }

                Ok((matrix_l.clone(), matrix_d, matrix_l.transpose()))
            }

            Err(error_msg) => Err(error_msg),
        }
    }

    /// Return a tuple (***Q, R***).
    ///
    /// <br>
    ///
    /// ### *QR* Decomposition:
    /// &emsp; ***A = QR***, ***Q*** is a matrix contains orthonormal basis
    /// , from doing ***Gram Schmidt*** process. ***R*** is a upper triangular
    /// matrix contains inner products with ***A***.
    pub fn qr_decomposition(self: &Self) -> Result<(Matrix, Matrix), String> {
        match self.gram_schmidt() {
            Ok(matrix_q) => {
                let mut matrix_r: Matrix = Matrix::zeros(self.col, self.col);
                for r in 0..matrix_r.row {
                    let orthonormal_col: Vector = matrix_q.get_column_vector(r).unwrap();
                    for c in r..matrix_r.col {
                        matrix_r.entries[r][c] = self
                            .get_column_vector(c)
                            .unwrap()
                            .inner_product(&orthonormal_col)
                            .unwrap();
                    }
                }

                Ok((matrix_q, matrix_r))
            }

            Err(error_msg) => Err(error_msg),
        }
    }

    /// # NEED TO FIX
    /// Return a tuple (***U, Σ, V^T***).
    ///
    /// <br>
    ///
    /// ### *SVD* Decomposition:
    /// &emsp; ***A*** = ***UΣV^T***.
    /// 
    /// Using Biorthogonalization SVD
    pub fn singular_value_decomposition(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        let mut matrix_a = self.clone();
        if matrix_a.col > matrix_a.row {
            matrix_a = matrix_a.multiply_Matrix(&matrix_a.transpose()).unwrap();
        } else {
            matrix_a = matrix_a.transpose().multiply_Matrix(&matrix_a).unwrap();
        }

        let mut matrix_q: Matrix = Self::random_matrix(matrix_a.col, matrix_a.row, -1.0, 1.0);
        let mut tuple: (Matrix, Matrix) = matrix_q.qr_decomposition().unwrap();
        matrix_q = tuple.0;
        let mut matrix_r: Matrix = tuple.1;
        let mut previous_q: Matrix = matrix_q.clone();
        let mut step: i32 = 0;
        let mut difference: f64 = 1.0;
        while difference > 1e-32 && step < 10000 {
            step += 1;
            difference = 0.0;

            let matrix_z: Matrix = matrix_a.multiply_Matrix(&matrix_q).unwrap();
            tuple = matrix_z.qr_decomposition().unwrap();
            matrix_q = tuple.0;
            matrix_r = tuple.1;
            difference = matrix_q.calculate_square_error(&previous_q).unwrap();
            previous_q = matrix_q.clone();
        }

        let mut matrix_s: Matrix = matrix_r.take_diagonal_entries().square_root();
        if self.row > self.col {
            let inverse_s: Matrix = matrix_s.inverse().unwrap();
            matrix_s = matrix_s.append_Matrix(&Matrix::zeros(self.row - self.col, self.col), 0).unwrap();
            let matrix_v_t: Matrix = matrix_q.transpose().clone();
            let matrix_u: Matrix = self.multiply_Matrix(&matrix_v_t.transpose()).unwrap().multiply_Matrix(&inverse_s).unwrap();
            
            return Ok((matrix_u, matrix_s, matrix_v_t));
        } else if self.row == self.col {
            let matrix_v_t: Matrix = matrix_q.transpose().clone();
            let matrix_u: Matrix = matrix_q.transpose().clone();
            matrix_s = matrix_s.to_powi(2);

            return Ok((matrix_u, matrix_s, matrix_v_t));
        } else {
            let inverse_s: Matrix = matrix_s.inverse().unwrap();
            matrix_s = matrix_s.append_Matrix(&Matrix::zeros(self.row, self.col - self.row), 1).unwrap();
            let matrix_u: Matrix = matrix_q.transpose().clone();
            let matrix_v_t: Matrix = inverse_s.multiply_Matrix(&matrix_u.transpose()).unwrap().multiply_Matrix(&self).unwrap();
        
            return Ok((matrix_u, matrix_s, matrix_v_t));
        }
    }

    /// # NEED TO FIX
    /// Return a tuple (***U, Σ, V^T***).
    ///
    /// <br>
    ///
    /// ### *SVD* Decomposition:
    /// &emsp; ***A*** = ***UΣV^T***.
    /// 
    /// Using Biorthogonalization SVD
    pub fn singular_value_decomposition_test(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        let mut matrix_a: Matrix = self.clone();
        match shift_qr_algorithm(&matrix_a, 1000, 1e-16) {
            Ok(tuple) => {
                let mut eigenvalue: Vector = tuple.0;
                let mut matrix_eigenvalue: Matrix = eigenvalue.to_diagonal();
                let mut matrix_sigma: Matrix = eigenvalue.square_root().to_diagonal();
                let matrix_u: Matrix = matrix_a.multiply_Matrix(&matrix_a.transpose()).unwrap();
                let matrix_v_t: Matrix = matrix_a.transpose().multiply_Matrix(&matrix_a).unwrap();

                Ok((matrix_u, matrix_sigma, matrix_v_t))
            }

            Err(error_msg) => {
                Err(error_msg)
            }
        }
    }
}
