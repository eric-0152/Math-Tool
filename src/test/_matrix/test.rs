use linalg::{to_matrix, matrix::Matrix};

#[test]
fn inverse() {    
    let matrix = Matrix::read_txt("./src/test/_matrix/inverse/p1.txt").unwrap();
    let answer = Matrix::read_txt("./src/test/_matrix/inverse/a1.txt").unwrap();
    assert_eq!(matrix.inverse().unwrap().entries, answer.entries);

    let matrix = Matrix::read_txt("./src/test/_matrix/inverse/p2.txt").unwrap();
    let answer = Matrix::read_txt("./src/test/_matrix/inverse/a2.txt").unwrap();
    assert_eq!(matrix.inverse().unwrap().round(8).entries, answer.round(8).entries);

    let matrix = Matrix::read_txt("./src/test/_matrix/inverse/p3.txt").unwrap();
    assert!(matrix.inverse().is_err());

    let matrix = Matrix::read_txt("./src/test/_matrix/inverse/p4.txt").unwrap();
    assert!(matrix.inverse().is_err());
}

