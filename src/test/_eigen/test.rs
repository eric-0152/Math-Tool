use linalg::{matrix::Matrix, polynomial::Polynomial, vector::Vector, *};

#[test]
fn lambda_polynomial() {
    let matrix = Matrix::read_txt("./src/test/_eigen/lambda_polynomial/p1.txt").unwrap();
    let answer = Polynomial::read_txt("./src/test/_eigen/lambda_polynomial/a1.txt").unwrap();
    assert_eq!(eigen::lambda_polynomial(&matrix).coeff, answer.coeff);

    let matrix = Matrix::read_txt("./src/test/_eigen/lambda_polynomial/p2.txt").unwrap();
    let answer = Polynomial::read_txt("./src/test/_eigen/lambda_polynomial/a2.txt").unwrap();
    assert_eq!(eigen::lambda_polynomial(&matrix).coeff, answer.coeff);
}

#[test]
fn eigenvalue() {
    let matrix = Matrix::read_txt("./src/test/_eigen/eigenvalue/p1.txt").unwrap();
    let answer = Vector::read_txt("./src/test/_eigen/eigenvalue/a1.txt").unwrap();
    assert_eq!(eigen::eigenvalue(&matrix).unwrap().round(8).entries, answer.round(8).entries);
}
