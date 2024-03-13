use crate::common::array::Array;
use crate::common::tensor::Tensor;

pub enum NDArray<A, D> {
    Tensor(Tensor),
    Array(Array<A, D>),
}
