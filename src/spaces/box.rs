//! Implementation of a space that represents closed boxes in euclidean space.
use crate::spaces::space::Space;
use crate::tensor::{DType, Device, Tensor};
use crate::utils::seeding::{rs_random, Generator};

fn _short_repr(arr: Tensor) -> String {
    let arr_size = arr.elem_count();
    if arr_size == 0 {
        return "[]".to_string();
    } else {
        let flattened_arr: Tensor = arr.flatten_all().unwrap().to_dtype(DType::F32).unwrap();
        let min: f32 = flattened_arr.min(0).unwrap().to_scalar().unwrap();
        let max: f32 = flattened_arr.max(0).unwrap().to_scalar().unwrap();
        if min == max {
            return format!("[{}]", min);
        } else {
            return arr.to_string();
        }
    }
}

/// Struct that is used to define observation and action spaces.
#[derive(Debug, Clone)]
pub struct Box {
    pub shape: Option<Vec<usize>>,
    pub dtype: Option<String>,
    pub rs_random: Generator,
    pub device: Option<Device>,
    pub low: Tensor,
    pub high: Tensor,
}

impl Space<Tensor> for Box {
    fn is_flattenable(&self) -> bool {
        true
    }

    fn sample<T>(&self, mask: Option<T>) -> Tensor {
        todo!()
    }

    fn seed(&mut self, seed: Option<usize>) -> Vec<usize> {
        let rs_seed;
        (self.rs_random, rs_seed) = rs_random(seed);

        // Return the seed in a vec
        vec![rs_seed]
    }

    fn contains<T>(&self, x: T) -> bool {
        true
    }
}
