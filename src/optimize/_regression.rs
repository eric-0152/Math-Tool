use crate::matrix::Matrix;
use crate::vector::Vector;

/// Given a matrix ***A*** and a vecor of answer ***y***, return a vector ***x*** which |***Ax - y***| is minimized.
///
/// ### Formula :
/// &emsp; ***x = (A^T @ A)^-1 @ A^T @ y***
pub fn least_squared_approximation(kernel_matrix: &Matrix, y: &Vector) -> Result<Vector, String> {
    if kernel_matrix.row != y.size {
        return Err("Input Error: The size of vector does not match.".to_string());
    }
    let transposed_matrix: Matrix = kernel_matrix.transpose();
    let result: Result<Matrix, String> = transposed_matrix
        .multiply_Matrix(kernel_matrix)
        .unwrap()
        .inverse();

    match result {
        Err(error_msg) =>  Err(error_msg),
        Ok(inverse) => Ok(inverse
            .multiply_Matrix(&transposed_matrix)
            .unwrap()
            .multiply_Vector(y)
            .unwrap())
    }
}

/// Return a tuple that contains the kernel matrix which can be applied to **least_squared_approximation()**,
/// and the vector ***y***.
///
/// ***y*** will not be modified, but return it for convenience.
///
/// The coefficients vector after **least_squared_approximation()** is [[***c_0, c_1, ..., c_n***]],
/// such that ***c_0 + c_1 x + ... c_n x^n = y***.
pub fn polynomial_kernel(
    x: &Vector,
    y: &Vector,
    degree: usize,
) -> Result<(Matrix, Vector), String> {
    if x.size != y.size {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let mut kernel_matrix: Matrix = Vector::ones(x.size).to_Matrix(1).unwrap();
    let mut powered_x: Vector = x.clone();
    for _ in 0..degree {
        kernel_matrix = kernel_matrix.append_Vector(&powered_x, 1).unwrap();
        for s in 0..x.size {
            powered_x.entries[s] *= x.entries[s];
        }
    }

    Ok((kernel_matrix, y.clone()))
}

/// Return a Vector which contains the coefficients, the order is from constant to higest degree.
///
/// Given a corresponding ***x*** and answer ***y***, using a polynomial function to do the
/// regression.
pub fn polynomial_regression(x: &Vector, y: &Vector, degree: usize) -> Result<Vector, String> {
    match polynomial_kernel(x, y, degree) {
        Err(error_msg) => Err(error_msg),
        Ok(tuple) => match least_squared_approximation(&tuple.0, &tuple.1) {
            Err(error_msg) => Err(error_msg),
            Ok(coefficients) => Ok(coefficients)
        }
    }
}

/// Return a tuple that contains the kernel matrix which can be applied to **least_squared_approximation()**,
/// and the vector ***f(x) = ln(y)***.
///
/// Size of kernel matrix : x.size x 2.
///
/// The coefficients vector after **least_squared_approximation()** is [[***ln(c), a***]],
/// such that ***ce^(ax) = y***.
///
/// ### Formula :
/// &emsp; ***y = ce^(ax)*** => ***ln(y) = ln(c)*** + ***ax***
pub fn exponential_kernel(x: &Vector, y: &Vector) -> Result<(Matrix, Vector), String> {
    if x.size != y.size {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let mut kernel_matrix: Matrix = Vector::ones(x.size).to_Matrix(1).unwrap();
    kernel_matrix = kernel_matrix.append_Vector(x, 1).unwrap();

    let mut fx = y.clone();
    for e in 0..fx.size {
        fx.entries[e] = fx.entries[e].ln();
    }

    Ok((kernel_matrix, fx))
}

/// Return a Vector [[***ln(c), a***]], such that ***ce^(ax) = y***.
///
/// Given a corresponding ***x*** and answer ***y***, using a exponential function to do the
/// regression.
pub fn exponential_regression(x: &Vector, y: &Vector) -> Result<Vector, String> {
    match exponential_kernel(x, y) {
        Err(error_msg) => Err(error_msg),
        Ok(tuple) => {
            match least_squared_approximation(&tuple.0, &tuple.1) {
                Err(error_msg) => Err(error_msg),
                Ok(coefficients) => Ok(coefficients)
            }
        }
    }
}

/// Return a tuple that contains the kernel matrix which can be applied to **least_squared_approximation()**, <br>
/// and the vector ***fx = ln(y^2)***.
///
/// Size of kernel matrix : x.size x 2.
///
/// The coefficients vector after **least_squared_approximation()** is [[***2 * ln(a), 1/c^2***]],
/// such that
///  
/// ### Formula :
/// &emsp; ***y = a e^(-1/2 * ((x-μ)/c)^2)*** <br>
/// &emsp; ***ln(y) = ln(a) + (-1/2 * ((x-μ)/c)^2)*** <br>
/// &emsp; ***2 * ln(y) = 2 * ln(a) + -((x-μ)/c)^2*** <br>
/// &emsp; ***2 * ln(y) = 2 * ln(a) + -((x-μ)^2 / c^2*** <br>
pub fn gaussian_1d_kernel(x: &Vector, y: &Vector) -> Result<(Matrix, Vector), String> {
    if x.size != y.size {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let average: f64 = x.entries_sum() / x.size as f64;
    let mut kernel_matrix: Matrix = Matrix::ones(x.size, 1);
    kernel_matrix = kernel_matrix.append_Vector(x, 1).unwrap();
    for e in 0..kernel_matrix.row {
        kernel_matrix.entries[e][1] = -(kernel_matrix.entries[e][1] - average).powi(2);
    }

    let mut fx = y.clone();

    for e in 0..fx.size {
        fx.entries[e] = 2.0 * fx.entries[e].ln();
    }

    Ok((kernel_matrix, fx))
}

/// Return a vector [[***2 * ln(a), 1/c^2***]], such that ***a e^(-1/2 * ((x-μ)/c)^2) = y***.
///
/// Given a corresponding ***x*** and answer ***y***, using a Gaussian 1D function to do the
/// regression.
pub fn gaussian_1d_regression(x: &Vector, y: &Vector) -> Result<Vector, String> {
    match gaussian_1d_kernel(x, y) {
        Err(error_msg) => Err(error_msg),
        Ok(tuple) => {
            match least_squared_approximation(&tuple.0, &tuple.1) {
                Err(error_msg) => Err(error_msg),
                Ok(coefficients) => Ok(coefficients)
            }
        }
    }
}
