use crate::matrix::Matrix;

impl Matrix {
    /// Return the matrix after rotation.
    ///
    /// Paramter i,j start from zero.
    pub fn givens_rotation(self: &Self, i: usize, j: usize, angle: f64) -> Result<Matrix, String> {
        if i >= self.shape.0 || j >= self.shape.0 {
            return Err("Input Error: Parameter i or j is out of bound.".to_string());
        }

        let mut rotation_matrix: Matrix = Matrix::identity(self.shape.0);
        rotation_matrix.real[i][i] = angle.cos();
        rotation_matrix.real[j][j] = angle.cos();
        rotation_matrix.real[j][i] = angle.sin();
        rotation_matrix.real[i][j] = -angle.sin();

        Ok(&rotation_matrix * self)
    }
}
