use crate::matrix::{self, Matrix};

impl Matrix {
    // / Return the matrix which contains orthonormal basis.
    pub fn gram_schmidt(self: &Self) -> Result<Matrix, String> {
        if self.shape.0 == 0 {
            return Err("Value Error: This matrix has no column.".to_string());
        }

        let mut matrix: Matrix = self.clone();
        if matrix.shape.0 < matrix.shape.1 {
            matrix = matrix.transpose();
        } 

        let mut current_col: Matrix = matrix.get_column_vector(0)?;
        let mut orthonormal_matrix: Matrix = &current_col * &(1.0 / current_col.euclidean_distance()?);

        for c in 1..matrix.shape.1 {
            current_col = matrix.get_column_vector(c)?;
            let mut new_orthonormal: Matrix = current_col.clone();
            for pre_c in 0..c {
                let previous_orthonormal: Matrix = orthonormal_matrix.get_column_vector(pre_c)?;
                let dot_product: f64 = (&previous_orthonormal * &current_col).real[0][0];
                new_orthonormal = &new_orthonormal - &(&previous_orthonormal * &dot_product);
            }

            new_orthonormal = &new_orthonormal * &(1.0 / new_orthonormal.euclidean_distance()?);
            orthonormal_matrix = orthonormal_matrix.append(&new_orthonormal, 1)?;
        }

        Ok(orthonormal_matrix)
    }
}
