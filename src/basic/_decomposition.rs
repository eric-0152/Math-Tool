use crate::{eigen, solve};
use crate::matrix::Matrix;
use crate::vector::Vector;

/// For shift_qr_algorithm
fn similar_matrix(matrix: &Matrix) -> Result<Matrix, String> {
    match matrix.qr_decomposition() {
        Err(error_msg) => Err(error_msg),
        Ok(tuple) => Ok(tuple.1.multiply_Matrix(&tuple.0).unwrap())
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
            Err(err_msg) => Err(err_msg),
            Ok(inverse_d) => {
                let matrix_v: Matrix = inverse_d.multiply_Matrix(&matrix_u).unwrap();
                Ok((matrix_l, matrix_d, matrix_v, permutation))
            }
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
            Err(error_msg) => Err(error_msg),
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
            Err(error_msg) => Err(error_msg),
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

                Ok((matrix_q.replace_nan(), matrix_r.replace_nan()))
            }
        }
    }

    pub fn singular_value_decomposition(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        let ata: Matrix = self.transpose().multiply_Matrix(self).unwrap();
        match eigen::eigenvalue_with_qr(&ata, 10000, 1e-32) {
            Err(error_msg) => Err(error_msg),
            Ok(tuple) => {
                // Remove if eigenvalue is 0.
                let mut eigenvalue: Vec<f64> = Vec::new();  
                for e in 0..tuple.0.size {
                    if tuple.0.entries[e] == 0.0 {
                        break;
                    }
                    eigenvalue.push(tuple.0.entries[e]);
                }
                
                let eigenvalue: Vector = Vector::from_vec(&eigenvalue); 
                for e in 0..eigenvalue.size {
                    let eigenkernel = Matrix::identity(ata.row).multiply_scalar(&eigenvalue.entries[e]).substract_Matrix(&ata).unwrap();
                    solve::null_space(&eigenkernel);
                }

                Ok((ata.clone(), ata.clone(), ata))
            }
        }
    }
}