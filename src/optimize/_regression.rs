use crate::{matrix::Matrix, vector::Vector};

/// ### Formula :
/// &emsp; ***x = (A^T @ A)^-1 @ A^T @ y***
/// 
/// Given a matrix ***A*** and a vecor of answer ***y***, return a vector ***x*** which |***Ax - y***| is minimized.
pub fn least_squared_approximation(kernel: &Matrix, y: &Vector) -> Result<Vector, String> {
    if kernel.shape.0 != y.size {
        return Err("Input Error: The size of vector does not match.".to_string());
    }

    let transposed_kernel: Matrix = kernel.transpose();
    let result: Result<Matrix, String> = (&transposed_kernel * kernel).inverse();
    // result.clone()?.display();
    // transposed_kernel.display();
    // (&result.clone()? * &transposed_kernel).display();
    (&(&result.clone()? * &transposed_kernel) * y).display();
    match result {
        Err(error_msg) => Err(error_msg),
        Ok(inverse) => Ok(&(&inverse * &transposed_kernel) * y),
    }
}

/// ### Formula :
/// &emsp; ***c0 + c1 x + ... + cn x^n = y***.
/// 
/// ### Return a tuple (***K, y***) :
/// &emsp; ***K*** : The kernel matrix which can be applied to **least_squared_approximation()**.
/// 
/// &emsp; ***y*** : The answer vector.
pub fn polynomial_kernel(
    x: &Vector,
    y: &Vector,
    degree: usize,
) -> Result<(Matrix, Vector), String> {
    if x.size != y.size {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let mut kernel_matrix: Matrix = Matrix::ones(x.size, 1);
    let mut powered_x: Vector = x.clone();
    for _ in 0..degree {
        kernel_matrix = kernel_matrix.append_vector(&powered_x, 1)?;
        for s in 0..x.size {
            powered_x.entries[s] *= x.entries[s];
        }
    }

    Ok((kernel_matrix, y.clone()))
}

/// ### Formula :
/// &emsp; ***c0 + c1 x + ... + cn x^n = y***.
/// 
/// ### Return a vector which contains the coefficients(c0, c1, c2, ..., cn).
pub fn polynomial_regression(x: &Vector, y: &Vector, degree: usize) -> Result<Vector, String> {
    match polynomial_kernel(x, y, degree) {
        Err(error_msg) => Err(error_msg),
        Ok((kernel, modified_y)) => match least_squared_approximation(&kernel, &modified_y) {
            Err(error_msg) => Err(error_msg),
            Ok(coefficients) => Ok(coefficients),
        },
    }
}


/// ### Formula :
/// &emsp; ***a e^(c x) = y***.
/// 
/// ### Return a tuple (***K, y***) :
/// &emsp; ***K*** : The kernel matrix which can be applied to **least_squared_approximation()**.
/// 
/// &emsp; ***y*** : The modified answer vector.
/// 
/// ### Notice :
/// &emsp; The output of least_squared_approximation() will be [[***ln(a), c***]].
pub fn exponential_kernel(x: &Vector, y: &Vector) -> Result<(Matrix, Vector), String> {
    if x.size != y.size {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let mut kernel_matrix: Matrix = Matrix::ones(x.size, 1);
    kernel_matrix = kernel_matrix.append_vector(x, 1)?;

    let mut fx = y.clone();
    for e in 0..fx.size {
        fx.entries[e] = fx.entries[e].ln();
    }

    Ok((kernel_matrix, fx))
}

/// ### Formula :
/// &emsp; ***a e^(c x) = y***.
/// 
/// ### Return a vector which contains the coefficients(a, c).
pub fn exponential_regression(x: &Vector, y: &Vector) -> Result<Vector, String> {
    match exponential_kernel(x, y) {
        Err(error_msg) => Err(error_msg),
        Ok((kernel, modified_y)) => match least_squared_approximation(&kernel, &modified_y) {
            Err(error_msg) => Err(error_msg),
            Ok(mut coefficients) => {
                coefficients.entries[0] = coefficients.entries[0].exp();
                
                Ok(coefficients)},
        },
    }
}


/// ### Formula :
/// &emsp; ***a e^((-1 / 2) * ((x - μ) / c)^2) = y***.
/// 
/// ### Return a tuple (***K, y***) :
/// &emsp; ***K*** : The kernel matrix which can be applied to **least_squared_approximation()**.
/// 
/// &emsp; ***y*** : The modified answer vector.
/// 
/// ### Notice :
/// &emsp; The output of least_squared_approximation() will be [[***ln(a), 1 / c^2***]].
pub fn gaussian_1d_kernel(x: &Vector, y: &Vector) -> Result<(Matrix, Vector), String> {
    if x.size != y.size {
        return Err("Input Error: The size of x and y do not match.".to_string());
    }

    let average: f64 = x.entries_sum().re / x.size as f64;
    let mut kernel_matrix: Matrix = 2.0 * &Matrix::ones(x.size, 1);
    kernel_matrix = kernel_matrix.append_vector(x, 1)?;
    for e in 0..kernel_matrix.shape.0 {
        kernel_matrix.entries[e][1] = -(kernel_matrix.entries[e][1] - average).powi(2);
    }

    let mut fx: Vector = y.clone();
    for e in 0..fx.size {
        fx.entries[e] = 2.0 * fx.entries[e].ln();
    }

    Ok((kernel_matrix, fx))
}

/// ### Formula :
/// &emsp; ***a e^((-1 / 2) * ((x - μ) / c)^2) = y***.
/// 
/// ### Return a matrix which contains the coefficients(a, \c).
pub fn gaussian_1d_regression(x: &Vector, y: &Vector) -> Result<Vector, String> {
    match gaussian_1d_kernel(x, y) {
        Err(error_msg) => Err(error_msg),
        Ok((kernel, modefied_y)) => match least_squared_approximation(&kernel, &modefied_y) {
            Err(error_msg) => Err(error_msg),
            Ok(mut coefficients) => {
                coefficients.entries[0] = (coefficients.entries[0]).exp();
                coefficients.entries[1] = (1.0 / coefficients.entries[1]).sqrt();
                
                Ok(coefficients)                
            }
        },
    }
}



/// ### Formula :
/// &emsp; ***c0 + c1 x + ... + cn x^n = y***.
pub fn polynomial_data(x: &Vector, coefficient: &Vector) -> Vector {
    let mut fx: Vector = coefficient.entries[0] * &Vector::ones(x.size);
    let mut powered_x: Vector = x.clone();
    for co in 1..coefficient.size {
        fx = &fx + &(&powered_x * coefficient.entries[co]);
        for i in 0..powered_x.size {
            powered_x.entries[i] *= x.entries[i];
        }
    }

    fx
}

/// ### Formula :
/// &emsp; ***a e^(c x) = y***.
pub fn exponential_data(x: &Vector, coefficient: &Vector) -> Vector {
    let mut fx: Vector = coefficient.entries[1] * x;
    for e in 0..x.size {
        fx.entries[e] = fx.entries[e].exp();
    }
    
    coefficient.entries[0] * &fx
}

/// ### Formula :
/// &emsp; ***a e^((-1 / 2) * ((x - μ) / c)^2) = y***.
pub fn gaussian_1d_data(x: &Vector, coefficient: &Vector) -> Vector {
    let mean: f64 = x.entries_sum().re / x.size as f64;
    let mut fx: Vector = &(x - mean) / coefficient.entries[1];
    for e in 0..x.size {
        fx.entries[e] = coefficient.entries[0] * (-0.5 * fx.entries[e].powi(2)).exp();
    }
    
    fx
}