use linalg::{to_matrix, matrix::Matrix, solve};

#[test]
fn null_space() {
    let matrix = Matrix::read_txt("./src/test/_solve/null_space/p1.txt").unwrap();
    let answer = Matrix::read_txt("./src/test/_solve/null_space/a1.txt").unwrap();
    let output = solve::null_space(&matrix);
    assert_eq!(output.entries, answer.entries);
}

