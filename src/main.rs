#![allow(dead_code, unused)]
#![allow(warnings)]


mod test;
use linalg::{decomposition, eigen, io, matrix::Matrix, polynomial::Polynomial, multipoly::MultiPoly, process, regression, solve, transform, vector::Vector, *};
use num_complex::Complex64;
use rand_distr::num_traits::ConstOne;
fn main() {    
    let mut mat = Matrix::random_matrix(5, 5, -100.0, 100.0, true).round(5);
    let mut t: Matrix = Matrix::random_matrix(3, 3, -9.0, 9.0, false).round(0);
    let m = Matrix::read_txt("./src/test.txt").unwrap();
    eigen::eigenvalue(&m).unwrap().round(8).display();
}

