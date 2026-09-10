use core::ops::{Add, Mul, Range, Sub};

/// Lambda parameter for generating a geometric distribution where each trial has probability `p`.
pub fn geometric_lambda(p: f64) -> f64 {
    if p <= 0.0 || p > 1.0 {
        panic!("Invalid probability: {p}");
    }
    (1.0 - p).ln()
}

/// Deterministic RNG used by current upstream ntsc-rs. It advances by the
/// SplitMix64 Weyl increment and applies upstream's selected finalizer for the
/// requested output type.
#[derive(Clone)]
pub struct SplitMix64 {
    state: u64,
}

const PHI: u64 = 0x9e3779b97f4a7c15;

impl SplitMix64 {
    pub fn random<T: FromState>(&mut self) -> T {
        self.state = self.state.wrapping_add(PHI);
        T::finalize(self.state)
    }

    /// Uniform float in `[low, high)`.
    #[inline]
    pub fn random_range<T: Rangeable>(&mut self, range: Range<T>) -> T {
        range.start + self.random::<T>() * (range.end - range.start)
    }

    pub fn random_geometric(&mut self, lambda: f64) -> usize {
        (self.random::<f64>().ln() / lambda) as usize
    }
}

pub trait Rangeable:
    Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Copy + FromState
{
}
impl Rangeable for f32 {}
impl Rangeable for f64 {}

/// Convert one SplitMix64 state using the finalizer selected by upstream.
pub trait FromState {
    fn finalize(state: u64) -> Self;
}

impl FromState for u32 {
    fn finalize(mut state: u64) -> Self {
        // David Stafford's Mix4 variant of the MurmurHash3 64-bit finalizer.
        state = (state ^ (state >> 33)).wrapping_mul(0x62A9D9ED799705F5);
        state = (state ^ (state >> 28)).wrapping_mul(0xCB24D0A5C88C35B3);
        (state >> 32) as u32
    }
}

impl FromState for i32 {
    fn finalize(state: u64) -> Self {
        u32::finalize(state) as i32
    }
}

impl FromState for u64 {
    fn finalize(mut state: u64) -> Self {
        state = (state ^ (state >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        state = (state ^ (state >> 27)).wrapping_mul(0x94d049bb133111eb);
        state ^ (state >> 31)
    }
}

impl FromState for i64 {
    fn finalize(state: u64) -> Self {
        u64::finalize(state) as i64
    }
}

impl FromState for f32 {
    fn finalize(state: u64) -> Self {
        (u32::finalize(state) >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

impl FromState for f64 {
    fn finalize(state: u64) -> Self {
        (u64::finalize(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Mix an independent seed into the RNG and finalize it once.
    pub fn mix(mut self, input: u64) -> Self {
        self.state = self.state.wrapping_add(input);
        Self {
            state: self.random(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_seed_sequence_matches_upstream() {
        let mut rng = SplitMix64::new(0x0123_4567_89ab_cdef);
        assert_eq!(rng.random::<u64>(), 0x157a_3807_a48f_aa9d);
        assert_eq!(rng.random::<u64>(), 0xd573_529b_34a1_d093);
        assert_eq!(rng.random::<u64>(), 0x2f90_b72e_996d_ccbe);
    }

    #[test]
    fn mix4_and_float_sequences_are_fixed() {
        let mut ints = SplitMix64::new(47).mix(6).mix(13);
        assert_eq!(ints.random::<u32>(), 0x7241_ffb0);
        assert_eq!(ints.random::<u32>(), 0x7c04_a0eb);

        let mut floats = SplitMix64::new(47).mix(6).mix(13);
        assert_eq!(floats.random::<f32>().to_bits(), 0x3ee4_83fe);
        assert_eq!(floats.random::<f32>().to_bits(), 0x3ef8_0940);
    }
}
