use crate::matrix::Matrix;

impl Matrix {
    /// Return the matrix after rotation.
    /// 
    /// Paramter i,j start from zero.
    pub fn givens_rotation(self: &Self, i: usize, j: usize, angle: f64) -> Result<Matrix, String> {
        if i >= self.row || j >= self.row {
            return Err("Input Error: Parameter i or j is out of bound.".to_string());
        }

        let mut rotation_matrix: Matrix = Matrix::identity(self.row);
        rotation_matrix.entries[i][i] = angle.cos();
        rotation_matrix.entries[j][j] = angle.cos();
        rotation_matrix.entries[j][i] = angle.sin();
        rotation_matrix.entries[i][j] = -angle.sin();

        Ok(rotation_matrix.multiply_Matrix(self).unwrap())
    }
}