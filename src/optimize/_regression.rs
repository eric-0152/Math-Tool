use crate::matrix::Matrix;

/// Given a matrix ***A*** and a vecor of answer ***y***, return a vector ***x*** which |***Ax - y***| is minimized.
///
/// ### Formula :
/// &emsp; ***x = (A^T @ A)^-1 @ A^T @ y***
pub fn least_squared_approximation(kernel_matrix: &Matrix, y: &Matrix) -> Result<Matrix, String> {
    if y.shape.1 != 1 {
        return Err("Input Error: The input y is not a vector.".to_string());
    } else if kernel_matrix.shape.0 != y.shape.0 {
        return Err("Input Error: The size of vector does not match.".to_string());
    }

    let transposed_matrix: Matrix = kernel_matrix.transpose();
    let result: Result<Matrix, String> = (&transposed_matrix * kernel_matrix).inverse();

    match result {
        Err(error_msg) => Err(error_msg),
        Ok(inverse) => Ok(&(&inverse * &transposed_matrix) * y),
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
    x: &Matrix,
    y: &Matrix,
    degree: usize,
) -> Result<(Matrix, Matrix), String> {
    if x.shape.1 != 1 {
        return Err("Input Error: The input x is not a vector.".to_string());
    } else if y.shape.1 != 1 {
        return Err("Input Error: The input y is not a vector.".to_string());
    } else if x.shape.0 != y.shape.0 {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let mut kernel_matrix: Matrix = Matrix::ones(x.shape.0, 1);
    let mut powered_x: Matrix = x.clone();
    for _ in 0..degree {
        kernel_matrix = kernel_matrix.append(&powered_x, 1)?;
        for s in 0..x.shape.0 {
            powered_x.real[s][0] *= x.real[s][0];
        }
    }

    Ok((kernel_matrix, y.clone()))
}

/// Return a Vector which contains the coefficients, the order is from constant to higest degree.
///
/// Given a corresponding ***x*** and answer ***y***, using a polynomial function to do the
/// regression.
pub fn polynomial_regression(x: &Matrix, y: &Matrix, degree: usize) -> Result<Matrix, String> {
    match polynomial_kernel(x, y, degree) {
        Err(error_msg) => Err(error_msg),
        Ok(tuple) => match least_squared_approximation(&tuple.0, &tuple.1) {
            Err(error_msg) => Err(error_msg),
            Ok(coefficients) => Ok(coefficients),
        },
    }
}

/// Return a tuple that contains the kernel matrix which can be applied to **least_squared_approximation()**,
/// and the vector ***f(x) = ln(y)***.
///
/// Size of kernel matrix : x.shape.0 x 2.
///
/// The coefficients vector after **least_squared_approximation()** is [[***ln(c), a***]],
/// such that ***ce^(ax) = y***.
///
/// ### Formula :
/// &emsp; ***y = ce^(ax)*** => ***ln(y) = ln(c)*** + ***ax***
pub fn exponential_kernel(x: &Matrix, y: &Matrix) -> Result<(Matrix, Matrix), String> {
    if x.shape.1 != 1 {
        return Err("Input Error: The input x is not a vector.".to_string());
    } else if y.shape.1 != 1 {
        return Err("Input Error: The input y is not a vector.".to_string());
    } else if x.shape.0 != y.shape.0 {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let mut kernel_matrix: Matrix = Matrix::ones(x.shape.0, 1);
    kernel_matrix = kernel_matrix.append(x, 1)?;

    let mut fx = y.clone();
    for e in 0..fx.shape.0 {
        fx.real[e][0] = fx.real[e][0].ln();
    }

    Ok((kernel_matrix, fx))
}

/// Return a Vector [[***ln(c), a***]], such that ***ce^(ax) = y***.
///
/// Given a corresponding ***x*** and answer ***y***, using a exponential function to do the
/// regression.
pub fn exponential_regression(x: &Matrix, y: &Matrix) -> Result<Matrix, String> {
    match exponential_kernel(x, y) {
        Err(error_msg) => Err(error_msg),
        Ok(tuple) => match least_squared_approximation(&tuple.0, &tuple.1) {
            Err(error_msg) => Err(error_msg),
            Ok(coefficients) => Ok(coefficients),
        },
    }
}

/// Return a tuple that contains the kernel matrix which can be applied to **least_squared_approximation()**, <br>
/// and the vector ***fx = ln(y^2)***.
///
/// Size of kernel matrix : x.shape.0 x 2.
///
/// The coefficients vector after **least_squared_approximation()** is [[***2 * ln(a), 1/c^2***]],
/// such that
///  
/// ### Formula :
/// &emsp; ***y = a e^(-1/2 * ((x-μ)/c)^2)*** <br>
/// &emsp; ***ln(y) = ln(a) + (-1/2 * ((x-μ)/c)^2)*** <br>
/// &emsp; ***2 * ln(y) = 2 * ln(a) + -((x-μ)/c)^2*** <br>
/// &emsp; ***2 * ln(y) = 2 * ln(a) + -((x-μ)^2 / c^2*** <br>
pub fn gaussian_1d_kernel(x: &Matrix, y: &Matrix) -> Result<(Matrix, Matrix), String> {
    if x.shape.0 != y.shape.0 {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let average: f64 = x.entries_sum() / x.shape.0 as f64;
    let mut kernel_matrix: Matrix = Matrix::ones(x.shape.0, 1);
    kernel_matrix = kernel_matrix.append(x, 1)?;
    for e in 0..kernel_matrix.shape.0 {
        kernel_matrix.real[e][1] = -(kernel_matrix.real[e][1] - average).powi(2);
    }

    let mut fx = y.clone();

    for e in 0..fx.shape.0 {
        fx.real[e][0] = 2.0 * fx.real[e][0].ln();
    }

    Ok((kernel_matrix, fx))
}

/// Return a vector [[***2 * ln(a), 1/c^2***]], such that ***a e^(-1/2 * ((x-μ)/c)^2) = y***.
///
/// Given a corresponding ***x*** and answer ***y***, using a Gaussian 1D function to do the
/// regression.
pub fn gaussian_1d_regression(x: &Matrix, y: &Matrix) -> Result<Matrix, String> {
    match gaussian_1d_kernel(x, y) {
        Err(error_msg) => Err(error_msg),
        Ok(tuple) => match least_squared_approximation(&tuple.0, &tuple.1) {
            Err(error_msg) => Err(error_msg),
            Ok(coefficients) => Ok(coefficients),
        },
    }
}
