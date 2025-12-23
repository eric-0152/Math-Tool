use crate::matrix::Matrix;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

impl Matrix {
    pub fn read_txt(path: &str) -> Result<Matrix, String> {
        let openfile: Result<File, std::io::Error> = File::open(path);

        match openfile {
            Err(erroe_msg) => Err(erroe_msg.to_string()),
            Ok(file) => {
                let reader: BufReader<File> = BufReader::new(file);
                let mut rows: Vec<Vec<f64>> = Vec::new();
                for line in reader.lines() {
                    let line: String = line.unwrap();
                    let line: std::str::SplitWhitespace<'_> = line.split_whitespace();
                    let elements: Result<Vec<f64>, std::num::ParseFloatError> =
                        line.map(|number| number.parse::<f64>()).collect();
                    match elements {
                        Err(error_msg) => return Err(error_msg.to_string()),
                        Ok(numbers) => {
                            if numbers.len() == 0 {continue;}
                            rows.push(numbers);
                        }
                    }
                }

                for r in 1..rows.len() {
                    if rows[0].len() != rows[r].len() {
                        return Err("Value Error: The size of rows are not match.".to_string());
                    }
                }

                Ok(Matrix::from_double_vec(&rows))
            }
        }
    }

    pub fn write_txt(matrix: &Matrix, path: &str) -> Result<File, String> {
        let openfile: Result<File, std::io::Error> = File::create(path);
        match openfile {
            Err(error_msg) => Err(format!("Operation Error: {error_msg}.")),
            Ok(mut file) => {
                for r in 0..matrix.shape.0 {
                    for c in 0..matrix.shape.1 {
                        write!(file, "{} ", matrix.real[r][c]).expect("Write entrie.");
                    }
                    write!(file, "\n").expect("Write new line.");
                }
                
                Ok(file)
            }
        }
    }
}
