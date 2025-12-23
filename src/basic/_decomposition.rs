use crate::matrix::Matrix;
use crate::eigen;

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
        let mut matrix_l: Matrix = Matrix::zeros(self.shape.0, self.shape.0);
        let mut permutation: Matrix = Matrix::identity(self.shape.0);
        for c in 0..self.shape.1.min(self.shape.0) {
            // If the pivot is 0.0, swap to non zero.
            if matrix_u.real[c][c] == 0.0 {
                let mut is_swap = false;
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
        let mut matrix_d: Matrix = Matrix::identity(self.shape.0);
        for d in 0..self.shape.0.min(self.shape.1) {
            matrix_d.real[d][d] = matrix_u.real[d][d];
        }

        match matrix_d.inverse() {
            Err(err_msg) => Err(err_msg),
            Ok(inverse_d) => {
                let matrix_v: Matrix = &inverse_d * &matrix_u;
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

        let mut matrix_l: Matrix = Matrix::zeros(self.shape.0, self.shape.1);
        for r in 0..matrix_l.shape.0 {
            for c in 0..(r + 1) {
                let mut summation: f64 = 0.0;
                if r == c {
                    for e in 0..c {
                        summation += matrix_l.real[c][e].powi(2);
                    }
                    matrix_l.real[r][c] = (self.real[c][c] - summation).sqrt();
                } else {
                    for e in 0..c {
                        summation += matrix_l.real[r][e] * matrix_l.real[c][e];
                    }
                    matrix_l.real[r][c] =
                        (self.real[r][c] - summation) / matrix_l.real[c][c];
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
                let mut matrix_d: Matrix = Self::identity(self.shape.0);
                let mut matrix_l: Matrix = matrix_c.clone();
                for d in 0..matrix_l.shape.1 {
                    matrix_d.real[d][d] = matrix_c.real[d][d].powi(2);
                    let inverse_sqrt_diagnol: f64 = matrix_d.real[d][d].sqrt();
                    for r in d..matrix_l.shape.0 {
                        matrix_l.real[r][d] = matrix_c.real[r][d] / inverse_sqrt_diagnol;
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
            Ok(mut matrix_q) => {
                let mut matrix: Matrix = self.clone();
                let mut has_transpose: bool = false;
                if matrix.shape.0 < matrix.shape.1 {
                    has_transpose = true;
                    matrix = matrix.transpose();
                } 

                let mut matrix_r: Matrix = Matrix::zeros(matrix.shape.1, matrix.shape.1);
                for r in 0..matrix_r.shape.0 {
                    let orthonormal_col: Matrix = matrix_q.get_column_vector(r)?;
                    for c in r..matrix_r.shape.1 {
                        matrix_r.real[r][c] = (&matrix.get_column_vector(c)? * &orthonormal_col).real[0][0];
                    }
                }

                if has_transpose {
                    let tmp: Matrix = matrix_q.transpose();
                    matrix_q = matrix_r.transpose();
                    matrix_r = tmp;
                }

                Ok((matrix_q.replace_nan(), matrix_r.replace_nan()))
            }
        }
    }

    /// ## NEET TO FIX
    /// Return the tuple ***(U, Σ, V^T)***
    pub fn singular_value_decomposition(self: &Self) -> Result<(Matrix, Matrix, Matrix), String> {
        let ata: Matrix = &self.transpose() * self;
        match eigen::eigenvalue(&ata, 1000, 1e-16) {
            Err(error_msg) => Err(error_msg),
            Ok(tuple) => {
                // Get eigenvalue
                const THERESHOLD: f64 = 1e-08;
                let mut eigenvalue: Matrix = tuple.0;
                for e in (0..eigenvalue.shape.0).rev() {
                    if eigenvalue.real[e][0] < THERESHOLD {
                        eigenvalue = eigenvalue.remove_row(e)?;
                    }
                }

                // Build V^T
                let mut vt: Matrix = Matrix::zeros(0, 0);
                for e in 0..eigenvalue.shape.0 {
                    let eigenvector: Matrix =
                        eigen::eigenvector(&ata, eigenvalue.real[e][0])?;
                    for c in 0..eigenvector.shape.1 {
                        vt = vt
                            .append(
                                &eigenvector.get_column_vector(c)?.normalize(),
                                0,
                            )
                            ?;
                    }
                }

                // Build Σ
                let sigma: Matrix = eigenvalue.square_root();
                
                // Build U
                let mut u: Matrix = Matrix::zeros(0, 0);
                for r in 0..vt.shape.0 {
                    u = u
                        .append(&(&(self * &vt.get_row_vector(r)?) * (&(1.0_f64 / sigma.real[r][r]))), 1,)
                        ?;
                }

                Ok((u, sigma, vt))
            }
        }
    }
}
