use rand::Rng;

#[path = "basic/_vector.rs"]
mod vector;
#[path = "basic/_matrix.rs"]
mod matrix;
#[path = "basic/_solve.rs"]
mod solve;
#[path = "basic/_decomposition.rs"]
mod decomposition;
#[path = "basic/_process.rs"]
mod process;
#[path = "basic/_transform.rs"]
mod transform;
#[path = "basic/_eigen.rs"]
mod eigen;


#[path = "optimize/_regression.rs"]
mod regression;

#[path = "optimize/_preprocessing.rs"]
mod preprocessing;


fn main() {


    let mut mat = matrix::Matrix::random_matrix(3, 3, 0.0, 9.0).round(0);
    let mut b = vector::Vector::random_vector(3, 0.0, 5.0).round(0);

    
    // Parameters of simulated curve
    let mut generator: rand::prelude::ThreadRng = rand::rng();

    let exponential_c = generator.random_range(5.0..10.0);
    let exponential_a = generator.random_range(2.0..10.0);
    let gaussian_a = generator.random_range(1.0..10.0);
    let gaussian_c = generator.random_range(5.0..10.0);
    let x =  vector::Vector::fixed_size_arithmetic_sequence(-1.0, 1.0, 1000);

    // Get mean
    let mu: f64 = x.entries_sum() / x.size as f64;

    // Simulated curve
    let mut y = x.clone();
    for e in 0..x.size {
        // gaussian + exponential
        // y.entries[e] = gaussian_a *  (-1.0/2.0 * ((x.entries[e] - mu) / gaussian_c).powi(2)).exp() + 
        //                 exponential_c * (exponential_a * x.entries[e]).exp();
        
        // exponential
        // y.entries[e] = exponential_c * (exponential_a * x.entries[e]).exp();
        
        // gaussian
        // y.entries[e] = gaussian_a *  (-1.0/2.0 * ((x.entries[e] - mu) / gaussian_c).powi(2)).exp();
    }    
    // y = preprocessing::normalize(&y);

    // Get regression kernel
    let exp_kernel_tuple: (matrix::Matrix, vector::Vector) = regression::exponential_kernel(&x, &y).unwrap();   
    let gaussian_kernel_tuple: (matrix::Matrix, vector::Vector) = regression::gaussian_1d_kernel(&x, &y).unwrap();   

        
    // Optimize
    let learning_tuple = regression::combine_regression_kernel(&exp_kernel_tuple.0, &gaussian_kernel_tuple.0, &exp_kernel_tuple.1, &gaussian_kernel_tuple.1);
    let mut params = regression::least_squared_approximation(&learning_tuple.0, &learning_tuple.1).unwrap();

    params.entries[0] = params.entries[0].exp();
    params.entries[2] = (params.entries[2] / 2.0).exp();
    params.entries[3] = params.entries[3].powf(-0.5);
    
    println!("{} {} {} {}", exponential_c, exponential_a, gaussian_a, gaussian_c);
    params.display();
}
