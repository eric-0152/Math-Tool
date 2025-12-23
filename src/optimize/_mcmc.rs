use std::clone;

use crate::matrix::Matrix;
use rand::Rng;
use rand_distr::{Distribution, Normal};

pub struct GuassianStep {
    pub mean: f64,
    pub std: f64,
    pub max_iter: u32,
    pub function: fn(&Matrix) -> f64,
    pub kernel: Normal<f64>,
    pub init_param: Matrix,
}

impl GuassianStep {
    pub fn init(
        mean: f64,
        std: f64,
        max_iter: u32,
        init_param: Matrix,
        function: fn(&Matrix) -> f64,
    ) -> Result<Self, String> {
        let kernel: Result<Normal<f64>, rand_distr::NormalError> = Normal::new(mean, std);
        match kernel {
            Err(error_msg) => Err(error_msg.to_string()),
            Ok(normal) => Ok(GuassianStep {
                mean,
                std,
                max_iter,
                function,
                kernel: normal,
                init_param: init_param.clone(),
            }),
        }
    }

    pub fn sample(self: &Self) -> f64 {
        self.kernel.sample(&mut rand::rng())
    }

    fn generate_new_param(self: &Self, last_param: Matrix, last_fx: f64) -> (Matrix, f64) {
        let mut new_param: Matrix = last_param.clone();

        for p in 0..new_param.shape.0 {
            new_param.real[p][0] += self.sample();
        }

        let mut new_fx: f64 = (self.function)(&new_param);
        if new_fx < last_fx {
            let mut generator: rand::prelude::ThreadRng = rand::rng();
            if (new_fx / last_fx) <= generator.random_range(0.0..=1.0) {
                new_param = last_param;
                new_fx = last_fx;
            }
        }

        (new_param, new_fx)
    }

    fn iter(self: &Self) -> Matrix {
        let mut current_param: Matrix = self.init_param.clone();
        let mut param_evo: Matrix = current_param.clone();
        let mut current_fx: f64 = (self.function)(&current_param);
        for _ in 0..self.max_iter {
            let tuple = Self::generate_new_param(self, current_param, current_fx);
            current_param = tuple.0;
            current_fx = tuple.1;
            param_evo = param_evo.append(&current_param.transpose(), 0).unwrap();
        }

        param_evo
    }
}

fn burn_in(param_evo: &Matrix, burn_in_step: u32) -> Result<Matrix, String> {
    if burn_in_step > param_evo.shape.0 as u32 {
        return Err("Input Error: Parameter burn_in_step should less than iteration".to_string());
    }

    let mut result_param_evo: Matrix = param_evo.clone();
    for _ in 0..burn_in_step {
        result_param_evo = result_param_evo.remove_row(0).unwrap();
    }

    Ok(result_param_evo)
}
