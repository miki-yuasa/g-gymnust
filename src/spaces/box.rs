//! Implementation of a space that represents closed boxes in euclidean space.
use crate::spaces::space::{Bound, Space};
use crate::tensor::{DType, Device, Tensor};
use crate::utils::seeding::{rs_random, Generator, Seed};

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
    pub dtype: DType,
    pub rs_random: Generator,
    pub device: Option<Device>,
    pub low: Tensor,
    pub high: Tensor,
    pub low_repr: String,
    pub high_repr: String,
    pub bounded_below: Tensor,
    pub bounded_above: Tensor,
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

impl Box {
    pub fn new(
        low: Bound,
        high: Bound,
        shape: Option<Vec<usize>>,
        dtype: DType,
        seed: Option<Seed>,
        device: Option<Device>,
    ) -> Self {
        let device = match device {
            Some(device) => device,
            None => Device::Cpu,
        };

        // Determine the shape if it is not provided directly
        let shape = match shape {
            Some(shape) => shape,
            None => {
                let low = match low {
                    Bound::F64(_) => panic!("Low must be a tensor"),
                    Bound::Tensor(ref tensor) => tensor,
                };
                let high = match high {
                    Bound::F64(_) => panic!("High must be a tensor"),
                    Bound::Tensor(ref tensor) => tensor,
                };
                if low.shape() != high.shape() {
                    panic!("Low and high must have the same shape");
                }
                low.shape().to_owned().into_dims()
            }
        };

        // Capture the boundedness information before replacing inf values with the largest DType value.
        let _low: Tensor = match low {
            Bound::F64(low) => Tensor::full(low, shape.clone(), &device).unwrap(),
            Bound::Tensor(low) => low,
        };
        let bounded_below: Tensor = _low.gt(-f32::INFINITY).unwrap().to_dtype(dtype).unwrap();

        let _high: Tensor = match high {
            Bound::F64(high) => Tensor::full(high, shape.clone(), &device).unwrap(),
            Bound::Tensor(high) => high,
        };
        let bounded_above: Tensor = _high.lt(f32::INFINITY).unwrap().to_dtype(dtype).unwrap();

        let low = _broadcast(_low.clone());
        let high = _broadcast(_high.clone());

        let _rs_random: Generator = match seed {
            Some(seed) => match seed {
                Seed::USize(seed) => rs_random(Some(seed)).0,
                Seed::Generator(generator) => generator,
            },
            None => rs_random(None).0,
        };

        let low_repr = _short_repr(_low.clone());
        let high_repr = _short_repr(_high.clone());

        Box {
            shape: Some(shape),
            dtype: dtype,
            rs_random: _rs_random,
            device: Some(device),
            low: low,
            high: high,
            low_repr: low_repr,
            high_repr: high_repr,
            bounded_below: bounded_below,
            bounded_above: bounded_above,
        }
    }
}

/// Handle infinite bounds and broadcast at the same time if needed.
fn _broadcast(value: Tensor) -> Tensor {
    value.clamp(-f32::INFINITY, f32::INFINITY).unwrap()
}
