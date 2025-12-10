use crate::matrix::Matrix;
use crate::vector::Vector;

impl Matrix {
    /// Return the matrix which contains orthonormal basis.
    pub fn gram_schmidt(self: &Self) -> Result<Matrix, String> {
        if self.row == 0 {
            return Err("Value Error: This matrix has no column.".to_string());
        }

        let mut current_col: Vector = self.get_column_vector(0).unwrap();
        let mut orthonormal_matrix: Matrix = current_col
            .multiply_scalar(1.0 / current_col.euclidean_distance())
            .transpose()
            .transpose();

        for c in 1..self.col {
            current_col = self.get_column_vector(c).unwrap();
            for pre_c in 0..c {
                let previous_col: Vector = orthonormal_matrix.get_column_vector(pre_c).unwrap();
                let dot_product: f64 = previous_col.inner_product(&current_col).unwrap();
                let norm_square: f64 = previous_col.euclidean_distance().powi(2);
                current_col = current_col
                    .substract_Vector(previous_col.multiply_scalar(dot_product / norm_square))
                    .unwrap();
            }
            current_col = current_col.multiply_scalar(1.0 / current_col.euclidean_distance());
            orthonormal_matrix = orthonormal_matrix.append_Vector(&current_col, 1).unwrap();
        }

        Ok(orthonormal_matrix)
    }

}