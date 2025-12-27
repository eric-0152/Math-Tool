#[path = "basic/_matrix.rs"]
pub mod matrix;
#[path = "basic/_vector.rs"]
pub mod vector;
#[path = "basic/_polynomial.rs"]
pub mod polynomial;
#[path = "basic/_multipoly.rs"]
pub mod multipoly;
#[path = "basic/_io.rs"]
pub mod io;
#[path = "basic/_decomposition.rs"]
pub mod decomposition;
#[path = "basic/_transform.rs"]
pub mod transform;
#[path = "basic/_process.rs"]
pub mod process;
#[path = "basic/_solve.rs"]
pub mod solve;
#[path = "basic/_eigen.rs"]
pub mod eigen;

#[path = "optimize/_mcmc.rs"]
pub mod mcmc;
#[path = "optimize/_preprocessing.rs"]
pub mod preprocessing;
#[path = "optimize/_regression.rs"]
pub mod regression;


#[macro_export]
macro_rules! to_matrix {
    (
        $([$( $e:expr),*]), * 
    ) => {{
        let mut rows = Vec::new();
        $(
            let mut row = Vec::new();
            $(
                row.push(io::_parse_str(format!("{}", $e).as_str()).unwrap());
            )*
            rows.push(row);
        )*

        matrix::Matrix::new(&rows).unwrap()
    }};
}

#[macro_export]
macro_rules! to_polynomial {
    (
        [$( $e:expr),*] 
    ) => {{
        let mut elements = Vec::new();
        $(
            elements.push(io::_parse_str(format!("{}", $e).as_str()).unwrap());
        )*

        polynomial::Polynomial::new(&elements)
    }};
}

#[macro_export]
macro_rules! to_vector {
    (
        [$( $e:expr),*] 
    ) => {{
        let mut elements = Vec::new();
        $(
            elements.push(io::_parse_str(format!("{}", $e).as_str()).unwrap());
        )*

        vector::Vector::new(&elements)
    }};
}

