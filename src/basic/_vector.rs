use std::ops::{Add, Mul, Sub, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg};
use rand::Rng;
use num_complex::Complex64;
use crate::matrix::Matrix;

#[derive(Clone, Debug)]
pub struct Vector {
    pub size: usize,
    pub entries: Vec<Complex64>,
}



impl Vector {
    pub fn new(vec: &Vec<Complex64>) -> Vector {        
        Vector {
            size: vec.len(),
            entries: vec.clone(),
        }
    }

    /// Return the size that round to the digit after decimal point.
    pub fn round(self: &Self, digit: usize) -> Vector {
        let mut result_vector: Vector = self.clone();
        let scale: f64 = 10_i32.pow(digit as u32) as f64;
        for e in 0..self.size {
            result_vector.entries[e].re = (scale * result_vector.entries[e].re).round();
            result_vector.entries[e].im = (scale * result_vector.entries[e].im).round();

            if result_vector.entries[e].re >= 1.0 || result_vector.entries[e].re <= -1.0 {
                result_vector.entries[e].re /= scale;
            } else if result_vector.entries[e].is_nan() {
                continue;
            } else {
                result_vector.entries[e].re = 0.0;
            }

            if result_vector.entries[e].im >= 1.0 || result_vector.entries[e].im <= -1.0 {
                result_vector.entries[e].im /= scale;
            } else if result_vector.entries[e].is_nan() {
                continue;
            } else {
                result_vector.entries[e].im = 0.0;
            } 
        }

        result_vector
    }

    pub fn replace_nan(self: &Self) -> Vector {
        let mut result_vector = self.clone();
        for e in 0..result_vector.size {
            if result_vector.entries[e].re.is_nan() {
                result_vector.entries[e].re = 0.0;
            } else if result_vector.entries[e].im.is_nan() {
                result_vector.entries[e].im = 0.0;
            }
        }

        result_vector
    }

    pub fn display(self: &Self) {
        let mut show_im = false;
        for e in 0..self.size {
            if self.entries[e].im != 0.0 {
                show_im = true;
                break;
            }
        }
        
        print!("[");
        if show_im {
            for e in 0..self.size {
                if self.entries[e].im >= 0.0 {
                    print!("{}", format!("{:>11?} {:>11} j", self.entries[e].re, format!("+ {:<?}", self.entries[e].im.abs())));
                } else {
                    print!("{}", format!("{:>11?} {:>11} j", self.entries[e].re, format!("- {:<?}", self.entries[e].im.abs())));
                }
                
                if e != (self.size - 1) {
                    print!(",");
                }
            }
        } else {
            for e in 0..self.size {
                print!("{}", format!("{:>11?}", self.entries[e].re));
                if e != (self.size - 1) {
                    print!(",")
                }
            }
        }
        println!("], size: {}", self.size);

    }

    /// Return a matrix contains all one entries with size m.
    pub fn ones(m: usize) -> Vector {
        Vector {
            size: m,
            entries: vec![Complex64::ONE; m],
        }
    }

    /// Return a matrix contains all zero entries with size m.
    pub fn zeros(m: usize) -> Vector {
        Vector {
            size: m,
            entries: vec![Complex64::ZERO; m],
        }
    }

    pub fn random_vector(m: usize, min: f64, max: f64, is_complex: bool) -> Vector {
        let mut result_vector: Vector = Self::zeros(m);
        let mut generator: rand::prelude::ThreadRng = rand::rng();
        if is_complex {
            for e in 0..m {
                result_vector.entries[e].re = generator.random_range(min..max);
                result_vector.entries[e].im = generator.random_range(min..max);
            }
        } else {
            for e in 0..m {
                result_vector.entries[e].re = generator.random_range(min..max);
            }
        }

        result_vector
    }

    pub fn arrange(start: f64, end: f64, step: f64) -> Result<Vector, String> {
        if start < end {
            if step < 0.0 {
                return Err("Input Error: Parameter step should be positive when start < end.".to_string());
            } else {
                let mut vector: Vec<Complex64> = Vec::new();
                let mut current = start;
                while current <= end {
                    vector.push(Complex64::new(current, 0.0));
                    current += step;
                }
        
                Ok(Vector::new(&vector))                
            }
        } else {
            if step > 0.0 {
                return Err("Input Error: Parameter step should be negative when start > end.".to_string());
            } else {
                let mut vector: Vec<Complex64> = Vec::new();
                let mut current = start;
                while current >= end {
                    vector.push(Complex64::new(current, 0.0));
                    current += step;
                }
                
                Ok(Vector::new(&vector))
            }
        }
    }
    
    pub fn linspace(start: f64, end: f64, size: usize) -> Vector {
        let mut vector: Vec<Complex64> = Vec::new();
        let mut current = start;
        let step = (end - start) / size as f64;
        for _ in 0..size {
            vector.push(Complex64::new(current, 0.0));
            current += step;
        }

        Vector::new(&vector)
    }
    
    /// Sum up all the entries in vector.
    pub fn entries_sum(self: &Self) -> Complex64 {
        let mut entries_sum: Complex64 = Complex64::ZERO;
        for e in 0..self.size {
            entries_sum += self.entries[e];
        }

        entries_sum
    }

    pub fn inner_product(self: &Self, vector: &Vector) -> Result<Complex64, String> {
        if self.size != vector.size {
            return Err("Input Error: The sizes of vector do not match".to_string());
        }
        
        let mut inner_product: Complex64 = Complex64::ZERO;
        for e in 0..self.size {
            inner_product += self.entries[e].conj() * vector.entries[e];
        }

        Ok(inner_product)
    }

    /// Append a vector to self.
    pub fn append(self: &Self, vector: &Vector) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..vector.size {
            result_vector.entries.push(vector.entries[e].clone());
        }
        
        result_vector.size += vector.size;
        result_vector
    }

    pub fn transpose(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Matrix::zeros(1, self.size);
        for e in 0..self.size {
            result_matrix.entries[0][e].re = self.entries[e].re;
            result_matrix.entries[0][e].im = -self.entries[e].im;
        }

        result_matrix
    }
    
    pub fn as_matrix(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Matrix::zeros(self.size, 1);
        for e in 0..self.size {
            result_matrix.entries[e][0] = self.entries[e];
        }

        result_matrix
    }
    
    /// Return a diagonal matrix which has the entries from vector.
    pub fn to_diagonal(self: &Self) -> Matrix {
        let mut result_matrix: Matrix = Matrix::identity(self.size);
        for d in 0..result_matrix.shape.0 {
            result_matrix.entries[d][d] = self.entries[d];
        }

        result_matrix
    }

    pub fn remove_element(self: &Self, i: usize) -> Result<Vector, String> {
        if i >= self.size {
            return Err("Input Error: Input row is out of bound".to_string());
        }

        let mut result_vector: Vector = self.clone();
        result_vector.entries.remove(i);

        Ok(result_vector)
    }

    pub fn swap_element(self: &Self, i: usize, j: usize) -> Result<Vector, String> {
        if i >= self.size || j >= self.size {
            return Err("Input Error: Input index is out of bound".to_string());
        }

        let mut result_vector: Vector = self.clone();
        result_vector.entries[i] = self.entries[j];
        result_vector.entries[j] = self.entries[i];

        Ok(result_vector)
    }

    /// Swap the elements according to the order of permutaion matrix.
    pub fn swap_with_permutation(self: &Self, permutation: &Matrix) -> Result<Vector, String> {
        if self.size != permutation.shape.0 {
            return Err(
                "Input Error: The row size of permutation matrix does not match".to_string(),
            );
        }
        
        Ok(permutation * self)
    }

    /// Return the length of self.
    pub fn norm(self: &Self) -> f64 {
        let mut distance: f64 = 0.0;
        for e in 0..self.size {
            distance += self.entries[e].re.powi(2);
            distance += self.entries[e].im.powi(2);
        } 

        distance.sqrt()
    }

    /// Normalize self to the length of 1.
    pub fn normalize(self: &Self) -> Vector {
        let mut result_vector: Vector = Vector::zeros(self.size);
        let distance: f64 = self.norm();     
        if distance > 0.0 {
            for e in 0..self.size {
                result_vector.entries[e] = self.entries[e] / distance;
            }
        }   

        result_vector
    }

    /// Return the vector that took square root on each element.
    pub fn square_root(self: &Self) -> Vector {
        let mut result_vector: Vector = Vector::zeros(self.size);
        for e in 0..self.size {
            result_vector.entries[e] = self.entries[e].sqrt();
        }

        result_vector
    }

    /// Return the vector that took power on each element.
    pub fn pow(self: &Self, power: f64) -> Vector {
        let mut result_vector: Vector = Vector::zeros(self.size);
        for e in 0..self.size {
            result_vector.entries[e] = self.entries[e].powf(power);
        }

        result_vector
    }
}


impl Add<&Vector> for &Vector {
    type Output = Vector;
    #[inline]
    fn add(self: Self, vector: &Vector) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..self.size {
            result_vector.entries[e] += vector.entries[e];
        }

        result_vector
    }
}
impl Add<f64> for &Vector {
    type Output = Vector;
    #[inline]
    fn add(self: Self, constant: f64) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..self.size {
            result_vector.entries[e] += constant;
        }

        result_vector
    }
}
impl Add<Complex64> for &Vector {
    type Output = Vector;
    #[inline]
    fn add(self: Self, constant: Complex64) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..self.size {
            result_vector.entries[e] += constant;
        }

        result_vector
    }
}
impl Add<&Vector> for f64 {
    type Output = Vector;
    #[inline]
    fn add(self: Self, vector: &Vector) -> Vector {
        let mut result_vector: Vector = vector.clone();
        for e in 0..vector.size {
            result_vector.entries[e] += self;
        }

        result_vector
    }
}
impl Add<&Vector> for Complex64 {
    type Output = Vector;
    #[inline]
    fn add(self: Self, vector: &Vector) -> Vector {
        let mut result_vector: Vector = vector.clone();
        for e in 0..vector.size {
            result_vector.entries[e] += self;
        }

        result_vector
    }
}

impl Sub for &Vector {
    type Output = Vector;
    #[inline]
    fn sub(self: Self, vector: &Vector) -> Vector {
        self + &(-1.0 * vector)
    }
}

impl Sub<f64> for &Vector {
    type Output = Vector;
    #[inline]
    fn sub(self: Self, constant: f64) -> Vector {
        self + -constant
    }
}
impl Sub<Complex64> for &Vector {
    type Output = Vector;
    #[inline]
    fn sub(self: Self, constant: Complex64) -> Vector {
        self + -constant
    }
}
impl Sub<&Vector> for f64 {
    type Output = Vector;
    #[inline]
    fn sub(self: Self, vector: &Vector) -> Vector {
        self + &(-1.0 * vector)
    }
}
impl Sub<&Vector> for Complex64 {
    type Output = Vector;
    #[inline]
    fn sub(self: Self, vector: &Vector) -> Vector {
        self + &(-1.0 * vector)
    }
}
impl Mul<f64> for &Vector {
    type Output = Vector;
    #[inline]
    fn mul(self: Self, scalar: f64) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..self.size {
            result_vector.entries[e] *= scalar;
        }

        result_vector
    }
}
impl Mul<Complex64> for &Vector {
    type Output = Vector;
    #[inline]
    fn mul(self: Self, scalar: Complex64) -> Vector {
        let mut result_vector: Vector = self.clone();
        for e in 0..self.size {
            result_vector.entries[e] *= scalar;
        }

        result_vector
    }
}
impl Mul<&Vector> for f64 {
    type Output = Vector;
    #[inline]
    fn mul(self: Self, vector: &Vector) -> Vector {
        let mut result_vector: Vector = vector.clone();
        for e in 0..result_vector.size {
            result_vector.entries[e] *= self;
        }

        result_vector
    }
}
impl Mul<&Vector> for Complex64 {
    type Output = Vector;
    #[inline]
    fn mul(self: Self, vector: &Vector) -> Vector {
        let mut result_vector: Vector = vector.clone();
        for e in 0..result_vector.size {
            result_vector.entries[e] *= self;
        }

        result_vector
    }
}

impl Div<f64> for &Vector {
    type Output = Vector;
    #[inline]
    fn div(self: Self, scalar: f64) -> Vector {
        if scalar != 0.0 {
            self * (1.0 / scalar)
        } else {
            panic!("Division by zero");
        }
    }
}
impl Div<Complex64> for &Vector {
    type Output = Vector;
    #[inline]
    fn div(self: Self, scalar: Complex64) -> Vector {
        if scalar != Complex64::ZERO {
            self * (1.0 / scalar)
        } else {
            panic!("Division by zero");
        }
    }
}
impl Div<&Vector> for f64 {
    type Output = Vector;
    #[inline]
    fn div(self: Self, vector: &Vector) -> Vector {
        if self != 0.0 {
            (1.0 / self) * vector
        } else {
            panic!("Division by zero");
        }
    }
}
impl Div<&Vector> for Complex64 {
    type Output = Vector;
    #[inline]
    fn div(self: Self, vector: &Vector) -> Vector {
        if self != Complex64::ZERO {
            (1.0 / self) * vector
        } else {
            panic!("Division by zero");
        }
    }
}


impl AddAssign for Vector {
    #[inline]
    fn add_assign(&mut self, other: Vector) {
        *self = &self.clone() + &other;
    }
}
impl AddAssign<f64> for Vector {
    #[inline]
    fn add_assign(&mut self, other: f64) {
        *self = &self.clone() + other;
    }
}
impl AddAssign<Complex64> for Vector {
    #[inline]
    fn add_assign(&mut self, other: Complex64) {
        *self = &self.clone() + other;
    }
}
impl SubAssign for Vector {
    #[inline]
    fn sub_assign(&mut self, other: Vector) {
        *self = &self.clone() - &other;
    }
}
impl SubAssign<f64> for Vector {
    #[inline]
    fn sub_assign(&mut self, other: f64) {
        *self = &self.clone() - other;
    }
}
impl SubAssign<Complex64> for Vector {
    #[inline]
    fn sub_assign(&mut self, other: Complex64) {
        *self = &self.clone() - other;
    }
}
impl Neg for Vector {
    type Output = Vector;
    #[inline]
    fn neg(self: Self) -> Vector {
        -1.0 * &self.clone()
    }
}
impl MulAssign<f64> for Vector {
    #[inline]
    fn mul_assign(&mut self, other: f64) {
        *self = &self.clone() * other;
    }
}
impl MulAssign<Complex64> for Vector {
    #[inline]
    fn mul_assign(&mut self, other: Complex64) {
        *self = &self.clone() * other;
    }
}
impl DivAssign<f64> for Vector {
    #[inline]
    fn div_assign(&mut self, other: f64) {
        *self = &self.clone() / other;
    }
}
impl DivAssign<Complex64> for Vector {
    #[inline]
    fn div_assign(&mut self, other: Complex64) {
        *self = &self.clone() / other;
    }
}