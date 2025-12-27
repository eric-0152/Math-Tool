use num_complex::Complex64;
use rand_distr::num_traits::ConstZero;
use std::{ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign}, thread::Thread};
use crate::{multipoly::{self, MultiPoly}, vector::Vector};

/// Coefficient of the poly is start from constant
#[derive(Clone, Debug)]
pub struct Polynomial {
    pub degree: usize,
    pub coeff: Vec<Complex64>,
}





impl Polynomial {
    pub fn new(coefficients: &Vec<Complex64>) -> Polynomial {
        if coefficients.len() == 0 {
            return Polynomial::new(&vec![Complex64::ZERO]);
        }

        let degree: usize = coefficients.len() - 1;
        Polynomial { degree, coeff: coefficients.clone() }.remove_redundant()
    }
    /// Return a poly with m 0.
    pub fn zero() -> Polynomial {
        Polynomial { degree: 0, coeff: vec![Complex64::ZERO] }
    }
    /// Return a poly with m 1.
    pub fn one() -> Polynomial {
        Polynomial { degree: 0, coeff: vec![Complex64::ONE] }
    }

    pub fn display(self: &Self) {
        let mut show_im = false;
        for e in 0..self.coeff.len() {
            if self.coeff[e].im != 0.0 {
                show_im = true;
                break;
            }
        }

        print!("[");
        if show_im {
            for e in 0..self.coeff.len() {
                if self.coeff[e].re != 0.0 || self.coeff[e].im != 0.0 {
                    if self.coeff[e].im >= 0.0 {
                        print!("{}", format!("({:>8?} {:>8} j)", self.coeff[e].re, format!("+ {:<?}", self.coeff[e].im.abs())));
                    } else {
                        print!("{}", format!("({:>8?} {:>8} j)", self.coeff[e].re, format!("- {:<?}", self.coeff[e].im.abs())));
                    }
                    if e != 0 {
                        print!("x^{}", e);
                    }
                    if e != (self.coeff.len() - 1) {
                        print!(" + ");
                    }
                }
            }
        } else {
            for e in 0..self.coeff.len() {
                if self.coeff[e].re != 0.0 {
                    print!("{}", format!("({:>8?})", self.coeff[e].re));
                    if e != 0 {
                        print!("x^{}", e);
                    }
                    if e != (self.coeff.len() - 1) {
                        print!(" + ")
                    }
                }
            }
        }
        println!("], degree: {}", self.degree);
    }

    #[inline]
    pub fn to_same_size(polynomial1: &Polynomial, polynomial2: &Polynomial) -> (Polynomial, Polynomial) {
        let p1 = polynomial1.remove_redundant();
        let p2 = polynomial2.remove_redundant();
        if p1.coeff.len() < p2.coeff.len() {
            let mut result_poly1: Polynomial = p1.clone();
            while result_poly1.coeff.len() < p2.coeff.len() {
                result_poly1.coeff.push(Complex64::ZERO);
            }

            return (result_poly1, p2.clone())
        } else if p1.coeff.len() > p2.coeff.len() {
            let mut result_poly2: Polynomial = p2.clone();
            while result_poly2.coeff.len() < p1.coeff.len() {
                result_poly2.coeff.push(Complex64::ZERO);
            }

            return (polynomial1.clone(), result_poly2)
        }

        (p1, p2)
    }

    #[inline]
    pub fn remove_redundant(self: &Self) -> Polynomial {
        let mut result_poly: Polynomial = self.clone();
        let mut row_index: usize = result_poly.coeff.len() - 1;
        while row_index > 0 && result_poly.coeff[row_index] == Complex64::ZERO {
            result_poly.coeff.pop();
            row_index -= 1;
        }

        result_poly.degree = result_poly.coeff.len() - 1;
        result_poly
    }

    #[inline]
    /// Return a tuple (quotient, remainder)
    pub fn divide_by(self: &Self, divider: &Polynomial) -> (Polynomial, Polynomial) {
        let divider = divider.remove_redundant();
        let mut remainder: Polynomial = self.clone().remove_redundant();
        let mut quotient: Polynomial = Polynomial::zero();
        let degree_one = Polynomial::new(&vec![Complex64::ZERO, Complex64::ONE]);
        let mut degree_diff: usize = remainder.degree - divider.degree;
        loop {
            let coefficient = remainder.coeff[remainder.degree] / divider.coeff[divider.degree];
            for d in 0..(divider.degree + 1) {
                remainder.coeff[d + degree_diff] -= coefficient * divider.coeff[d];
            }
            remainder = remainder.remove_redundant();
            quotient = &(&quotient * &degree_one) + &Polynomial::new(&vec![coefficient]);
            quotient.degree += 1;
            if degree_diff == 0 {
                break;
            } else {
                degree_diff -= 1;
            }
        }

        (quotient, remainder)
    }

    pub fn evaluate(self: &Self, value: &Complex64) -> Complex64 {        
        let mut result: Complex64 = Complex64::new(self.coeff[0].re, self.coeff[0].im);
        let mut power = 1.0;
        for d in 1..self.coeff.len() {
            let element: Complex64 = self.coeff[d] * value.powf(power);
            result += element;
            power += 1.0;
        }
        
        result
    }
    
    #[inline]
    pub fn differentiate(self: &Self) -> Polynomial {
        let mut result_poly: Polynomial = self.clone();
        for e in 0..self.coeff.len() - 1 {
            result_poly.coeff[e] = (e + 1) as f64 * self.coeff[e + 1];
        }

        result_poly
    }
}

fn newton_raphson(poly: &Polynomial) -> Result<Complex64, String> {
    // Build complex form of poly.
    let x_y: MultiPoly = MultiPoly::new(vec!["x".to_string(), "y".to_string()], vec![(Complex64::ONE, vec![1.0, 0.0]), (Complex64::new(0.0, 1.0), vec![0.0, 1.0])]).unwrap();
    let mut complex_poly: MultiPoly = MultiPoly::new(vec!["x".to_string(), "y".to_string()], vec![(poly.coeff[0], vec![0.0, 0.0])]).unwrap();
    let mut current_x_y: MultiPoly = x_y.clone();
    for e in 1..poly.coeff.len() {
        complex_poly = &complex_poly + &(poly.coeff[e] * &current_x_y);
        current_x_y = &current_x_y * &x_y;
    }

    // Separate to real and imginary parts.
    let mut real_poly: MultiPoly = MultiPoly::new(vec!["x".to_string(), "y".to_string()], vec![]).unwrap();
    let mut img_poly: MultiPoly = MultiPoly::new(vec!["x".to_string(), "y".to_string()], vec![]).unwrap();
    for coeff in complex_poly.coeff {
        if coeff.0.re != 0.0 {
            real_poly.coeff.push(coeff);
        } else {
            img_poly.coeff.push(coeff);
        }
    }

    // Newton-Raphson method
    img_poly = &img_poly * Complex64::new(0.0, -1.0);
    let real_partial_x: MultiPoly = real_poly.partial_derivative("x".to_string()).unwrap();
    let real_partial_y: MultiPoly = real_poly.partial_derivative("y".to_string()).unwrap();
    let img_partial_x: MultiPoly = img_poly.partial_derivative("x".to_string()).unwrap();
    let img_partial_y: MultiPoly = img_poly.partial_derivative("y".to_string()).unwrap();
    let mut param = Vector::random_vector(2, -1000.0, 1000.0, false);
    const THRESHOLD: f64 = 1e-10;
    const MAX_ITER: u32 = 100000000;
    let mut error: f64 = 1.0;
    let mut iter: u32 = 0;
    while error > THRESHOLD && iter < MAX_ITER {
        let real: f64 = real_poly.evaluate(&param).unwrap().re;
        let img: f64 = img_poly.evaluate(&param).unwrap().re;
        let real_dx: f64 = real_partial_x.evaluate(&param).unwrap().re;
        let real_dy: f64 = real_partial_y.evaluate(&param).unwrap().re;
        let img_dx: f64 = img_partial_x.evaluate(&param).unwrap().re;
        let img_dy: f64 = img_partial_y.evaluate(&param).unwrap().re;
        let denominator: f64 = (real_dx * img_dy) - (real_dy * real_dx);
        let dx: f64 = ((real * img_dy) - (real_dy * img)) / denominator;  
        let dy: f64 = ((real_dx * img) - (real * img_dx)) / denominator;  
        param.entries[0].re -= 0.9 * dx;
        param.entries[1].re -= 0.9 * dy;
        error = (&poly.evaluate(&Complex64::new(param.entries[0].re, param.entries[1].re)).re).abs();
        if error.is_nan() {
            return Err("Failed to converge".to_string());
        }
        iter += 1;
    }

    if error > THRESHOLD {
        return Err("Failed to converge".to_string());
    }
    
    Ok(Complex64::new(param.entries[0].re, param.entries[1].re))
}

/// Return a complex root using Newton-Raphson method.
pub fn find_root(poly: &Polynomial) -> Vector {
    let mut roots = Vector::new(&vec![]);
    let mut current_poly = poly.clone();
    let mut last_residual = Complex64::ZERO;
    while current_poly.coeff.len() > 1 {
        let find_result = newton_raphson(&current_poly);
        if find_result.is_err() {
            continue;
        }
        
        let mut new_root = find_result.unwrap();
        if !new_root.is_infinite() && !new_root.is_nan() {
            roots = roots.append(&Vector { size: 1, entries: vec![new_root] });
            let (quotient, remainder) = current_poly.divide_by(&Polynomial::new(&vec![-new_root, Complex64::ONE]));
            current_poly = quotient;
            last_residual = remainder.coeff[0];
        }
        
    }
    
    roots.entries[roots.size - 1] -= last_residual;
    roots.entries.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    roots.entries.reverse();
    roots
}









impl Add<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, float: f64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        result_poly.coeff[0] += float;

        result_poly.remove_redundant()
    }
}
impl Add<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn add(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        result_poly.coeff[0] += self;

        result_poly.remove_redundant()
    }
}
impl Add<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, complex: Complex64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        result_poly.coeff[0] += complex;

        result_poly.remove_redundant()
    }
}
impl Add<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn add(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        result_poly.coeff[0] += self;

        result_poly.remove_redundant()
    }
}
impl Add<&Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn add(self, poly: &Polynomial) -> Self::Output {
        let (mut p1, p2): (Polynomial, Polynomial) = Polynomial::to_same_size(self, poly);
        for d in 0..p1.coeff.len() {
            p1.coeff[d] += p2.coeff[d]
        }

        p1.degree = p1.coeff.len() - 1;
        p1.remove_redundant()
    }
}

impl Sub<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, float: f64) -> Self::Output {
        self + (-float)
    }
}
impl Sub<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn sub(self, poly: &Polynomial) -> Self::Output {
        self + &(-1.0 * poly)
    }
}
impl Sub<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, complex: Complex64) -> Self::Output {
        self + (-complex)
    }
}
impl Sub<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn sub(self, poly: &Polynomial) -> Self::Output {
        self + &(-1.0 * poly)
    }
}
impl Sub<&Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn sub(self, poly: &Polynomial) -> Self::Output {
        self + &(-1.0 * poly)
    }
}

impl Mul<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn mul(self, scalar: f64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= scalar;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn mul(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= self;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn mul(self, scalar: Complex64) -> Self::Output {
        let mut result_poly: Polynomial = self.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= scalar;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn mul(self, poly: &Polynomial) -> Self::Output {
        let mut result_poly: Polynomial = poly.clone();
        for d in 0..result_poly.coeff.len() {
            result_poly.coeff[d] *= self;
        }

        result_poly.remove_redundant()
    }
}
impl Mul<&Polynomial> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn mul(self, poly: &Polynomial) -> Self::Output {
        let p1: Polynomial = self.clone();
        let p2: Polynomial = poly.clone();

        if p1.degree == 0 {
            if p1.coeff.len() == 0 {
                return Polynomial::new(&vec![Complex64::ZERO]);
            } else {
                return p1.coeff[0] * &p2;
            }
        } else if p2.degree == 0 {
            if p2.coeff.len() == 0 {
                return Polynomial::new(&vec![Complex64::ZERO]);
            } else {
                return p2.coeff[0] * &p1;
            }
        }


        let mut result_poly: Polynomial = Polynomial::new(&vec![Complex64::ZERO]);
        while result_poly.degree < p1.degree + p2.degree {
            result_poly.coeff.push(Complex64::ZERO);
            result_poly.degree += 1;
        }
        for i in 0..p1.coeff.len() {
            for j in 0..p2.coeff.len() {
                result_poly.coeff[i + j] += p1.coeff[i] * p2.coeff[j];
            }
        }

        result_poly.remove_redundant()
    }
}
impl Div<&Polynomial> for f64 {
    type Output = Polynomial;
    #[inline]
    fn div(self, poly: &Polynomial) -> Self::Output {
        poly * (1.0 / self)
    }
}
impl Div<&Polynomial> for Complex64 {
    type Output = Polynomial;
    #[inline]
    fn div(self, poly: &Polynomial) -> Self::Output {
        poly * (1.0 / self)
    }
}
impl Div<f64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn div(self, float: f64) -> Self::Output {
        self * (1.0 / float)
    }
}
impl Div<Complex64> for &Polynomial {
    type Output = Polynomial;
    #[inline]
    fn div(self, complex: Complex64) -> Self::Output {
        self * (1.0 / complex)
    }
}


impl AddAssign for Polynomial {
    #[inline]
    fn add_assign(&mut self, other: Polynomial) {
        *self = &self.clone() + &other;
    }
}
impl AddAssign<f64> for Polynomial {
    #[inline]
    fn add_assign(&mut self, other: f64) {
        *self = &self.clone() + other;
    }
}
impl AddAssign<Complex64> for Polynomial {
    #[inline]
    fn add_assign(&mut self, other: Complex64) {
        *self = &self.clone() + other;
    }
}
impl SubAssign for Polynomial {
    #[inline]
    fn sub_assign(&mut self, other: Polynomial) {
        *self = &self.clone() - &other;
    }
}
impl SubAssign<f64> for Polynomial {
    #[inline]
    fn sub_assign(&mut self, other: f64) {
        *self = &self.clone() - other;
    }
}
impl SubAssign<Complex64> for Polynomial {
    #[inline]
    fn sub_assign(&mut self, other: Complex64) {
        *self = &self.clone() - other;
    }
}
impl Neg for Polynomial {
    type Output = Polynomial;
    #[inline]
    fn neg(self: Self) -> Polynomial {
        -1.0 * &self.clone()
    }
}
impl MulAssign for Polynomial {
    #[inline]
    fn mul_assign(&mut self, other: Polynomial) {
        *self = &self.clone() * &other;
    }
}
impl MulAssign<f64> for Polynomial {
    #[inline]
    fn mul_assign(&mut self, other: f64) {
        *self = &self.clone() * other;
    }
}
impl MulAssign<Complex64> for Polynomial {
    #[inline]
    fn mul_assign(&mut self, other: Complex64) {
        *self = &self.clone() * other;
    }
}
impl DivAssign<f64> for Polynomial {
    #[inline]
    fn div_assign(&mut self, other: f64) {
        *self = &self.clone() / other;
    }
}
impl DivAssign<Complex64> for Polynomial {
    #[inline]
    fn div_assign(&mut self, other: Complex64) {
        *self = &self.clone() / other;
    }
}
