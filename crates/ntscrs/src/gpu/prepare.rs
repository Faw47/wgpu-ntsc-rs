//! Compact stochastic control data. The CPU generates the reference RNG sequence;
//! shaders evaluate noise and transform image pixels. No image is processed here.
use crate::{
    noise::{Fbm, Simplex, Simplex1d, Simplex2d, sample_noise_1d, sample_noise_2d},
    noise_seeds,
    random::{Geometric, Seeder},
    settings::standard::*,
};
use fearless_simd::Level;
use rand::{Rng, RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Row {
    pub shift: f32,
    pub noise_seed: i32,
    pub offset: f32,
    pub intensity: f32,
    pub frequency: f32,
    pub octaves: u32,
    pub phase: f32,
    pub start: u32,
    pub transient_len: f32,
    pub transient_intensity: f32,
    pub extend: u32,
    pub loss: u32,
}

pub fn hash_row(seed: u64, row: usize) -> u64 {
    let mut h = seed.wrapping_add(row as u64);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
    h ^ (h >> 33)
}

pub fn seeder(seed: i32, frame: usize, tag: u64) -> Seeder {
    Seeder::new(seed as u32 as u64).mix(tag).mix(frame)
}

fn noise_row(
    seeder: &Seeder,
    index: usize,
    width: usize,
    frequency: f32,
    intensity: f32,
    detail: i32,
) -> Row {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seeder.clone().mix(index as u64).finalize());
    Row {
        noise_seed: rng.next_u32() as i32,
        offset: rng.random::<f32>() * width as f32,
        frequency,
        intensity: intensity * 0.25,
        octaves: (detail.max(0) as u32).clamp(1, 5),
        ..Row::default()
    }
}

pub fn noise(
    seed: i32,
    frame: usize,
    tag: u64,
    width: usize,
    rows: usize,
    scale: f32,
    settings: &FbmNoiseSettings,
) -> Vec<Row> {
    let seeder = seeder(seed, frame, tag);
    (0..rows)
        .map(|row| {
            noise_row(
                &seeder,
                row,
                width,
                settings.frequency / scale,
                settings.intensity,
                settings.detail,
            )
        })
        .collect()
}

pub fn phase(seed: i32, frame: usize, rows: usize, intensity: f32) -> Vec<Row> {
    let seed = seeder(seed, frame, noise_seeds::VIDEO_CHROMA_PHASE).finalize::<u64>();
    (0..rows)
        .map(|row| {
            let val = hash_row(seed, row) as f32 / (u64::MAX as f32 + 1.0);
            let (sin, cos) =
                (((val - 0.5) * 2.0 * intensity) * std::f32::consts::PI * 2.0).sin_cos();
            Row {
                phase: sin,
                frequency: cos,
                ..Row::default()
            }
        })
        .collect()
}

pub fn head(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &HeadSwitchingSettings,
) -> Vec<Row> {
    let mut result = vec![
        Row {
            start: width as u32,
            ..Row::default()
        };
        rows
    ];
    let count = (settings.height.max(0) as f32 * sy).round() as usize;
    let offset = (settings.offset.max(0) as f32 * sy).round() as usize;
    if offset >= count {
        return result;
    }
    let affected = count - offset;
    let start = rows.saturating_sub(affected);
    let cutoff = affected.saturating_sub(rows);
    let seeder = seeder(seed, frame, noise_seeds::HEAD_SWITCHING);
    for (index, row) in result[start..].iter_mut().enumerate() {
        let index = affected - (index + cutoff);
        let shift = settings.horiz_shift * ((index + offset) as f32 / count as f32).powf(1.5);
        row.shift = (shift + seeder.clone().mix(index).finalize::<f32>() - 0.5) * sx;
        row.start = 0;
        if index == affected
            && let Some(mid) = &settings.mid_line
        {
            let seeder = self::seeder(seed, frame, noise_seeds::HEAD_SWITCHING_MID_LINE_JITTER);
            let random = (seeder.clone().mix(0).finalize::<f32>()
                + seeder.clone().mix(1).finalize::<f32>())
                * 0.5;
            row.start = (width as f32 * (mid.position + (random - 0.5) * mid.jitter)) as u32;
            row.transient_len = 16.0 * sx;
            row.transient_intensity = (seeder.mix(0).finalize::<f32>() + 0.5) * 0.5;
        }
    }
    result
}

pub fn wave(
    seed: i32,
    frame: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &VHSEdgeWaveSettings,
) -> Vec<Row> {
    let seeder = Seeder::new(seed as u32 as u64).mix(noise_seeds::EDGE_WAVE);
    let noise = Fbm {
        seed: seeder.clone().mix(0).finalize(),
        octaves: settings.detail.clamp(1, 5) as usize,
        gain: std::f32::consts::FRAC_1_SQRT_2,
        lacunarity: 2.0,
        frequency: settings.frequency / sy,
    };
    let offset = seeder.mix(1).finalize::<f32>() * rows as f32;
    let mut shifts = vec![0.0; rows];
    sample_noise_2d::<Simplex2d, _>(
        Level::new(),
        &noise,
        [offset, frame as f32 * settings.speed],
        [rows, 1],
        &mut shifts,
    );
    shifts
        .into_iter()
        .map(|v| Row {
            shift: (v / 0.022) * settings.intensity * 0.5 * sx,
            extend: 1,
            ..Row::default()
        })
        .collect()
}

pub fn loss(seed: i32, frame: usize, rows: usize, intensity: f32) -> Vec<Row> {
    let mut result = vec![Row::default(); rows];
    let mut rng =
        Xoshiro256PlusPlus::seed_from_u64(seeder(seed, frame, noise_seeds::CHROMA_LOSS).finalize());
    let dist = Geometric::new(intensity as f64);
    let mut row = 0usize;
    loop {
        row = row.saturating_add(rng.sample(&dist));
        if row >= rows {
            break;
        }
        result[row].loss = 1;
        row += 1;
    }
    result
}

pub fn tracking(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &TrackingNoiseSettings,
) -> (Vec<Row>, Vec<u32>) {
    let count = (settings.height.max(0) as f32 * sy).round() as usize;
    let start = rows.saturating_sub(count);
    let cutoff = count.saturating_sub(rows);
    let seeder = seeder(seed, frame, noise_seeds::TRACKING_NOISE);
    let noise = Simplex {
        seed: seeder.clone().mix(0).finalize(),
        frequency: 0.5,
    };
    let offset = seeder.clone().mix(1).finalize::<f32>() * rows as f32;
    let seeder = seeder.mix(2);
    let base_seed = seeder.clone().finalize();
    let mut shifts = vec![0.0; count.min(rows)];
    sample_noise_1d::<Simplex1d, _>(Level::new(), &noise, [offset], [shifts.len()], &mut shifts);
    let mut result = vec![
        Row {
            start: width as u32,
            ..Row::default()
        };
        rows
    ];
    let mut snow = Snow::new(width, rows);
    for (local, row) in result[start..].iter_mut().enumerate() {
        let index = local + cutoff;
        let intensity = index as f32 / count as f32;
        *row = noise_row(
            &seeder,
            index,
            width,
            0.25 / sx,
            intensity.powi(2) * settings.noise_intensity * 4.0,
            1,
        );
        row.shift = shifts[local] * intensity * settings.wave_intensity * 0.25 * sx;
        snow.row(
            start + local,
            hash_row(base_seed, index),
            settings.snow_intensity * intensity.powi(2),
            settings.snow_anisotropy,
            sx,
        );
    }
    (result, snow.finish())
}

// Snow events are binned into 32-pixel tiles. Each pixel evaluates only overlapping
// events in reference order, without float atomics or a serial whole-row shader.
struct Snow {
    width: usize,
    tiles: Vec<Vec<u32>>,
    events: Vec<[u32; 4]>,
    random: Vec<u32>,
}
impl Snow {
    fn new(width: usize, rows: usize) -> Self {
        Self {
            width,
            tiles: vec![Vec::new(); width.div_ceil(32) * rows],
            events: Vec::new(),
            random: Vec::new(),
        }
    }
    fn row(&mut self, row: usize, seed: u64, intensity: f32, anisotropy: f32, scale: f32) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let intensity = intensity as f64;
        let anisotropy = anisotropy as f64;
        let logistic = ((rng.random::<f64>() - intensity)
            / (intensity * (1.0 - intensity) * (1.0 - anisotropy)))
            .exp();
        let probability = ((anisotropy / (1.0 + logistic) + intensity * (1.0 - anisotropy))
            * 0.125)
            .clamp(0.0, 1.0);
        if probability <= 0.0 {
            return;
        }
        let dist = Geometric::new(probability);
        let mut start = -64isize;
        loop {
            start = start.saturating_add(rng.sample(&dist).min(isize::MAX as usize) as isize);
            if start >= self.width as isize {
                break;
            }
            let len = rng.random_range(8.0..=64.0) * scale;
            let freq = rng.random_range(len * 3.0..=len * 5.0);
            let end = start
                .saturating_add(len.ceil() as isize)
                .clamp(0, self.width as isize) as usize;
            rng.jump();
            let mut event_rng = rng.clone();
            let visible = start.clamp(0, self.width as isize) as usize;
            if visible < end {
                let event = self.events.len() as u32;
                self.events.push([
                    start as i32 as u32,
                    len.to_bits(),
                    freq.to_bits(),
                    self.random.len() as u32,
                ]);
                for _ in visible..end {
                    self.random
                        .push(event_rng.random_range(-1.0f32..2.0).to_bits());
                }
                for tile in visible / 32..=(end - 1) / 32 {
                    self.tiles[row * self.width.div_ceil(32) + tile].push(event);
                }
            }
            start += 1;
        }
    }
    fn finish(self) -> Vec<u32> {
        if self.events.is_empty() {
            return Vec::new();
        }
        let mut words = vec![0u32; 4 + self.tiles.len() + 1];
        for (tile, events) in self.tiles.iter().enumerate() {
            words[4 + tile] = words.len() as u32;
            words.extend_from_slice(events);
        }
        words[4 + self.tiles.len()] = words.len() as u32;
        words[0] = words.len() as u32;
        for event in self.events {
            words.extend_from_slice(&event);
        }
        words[1] = words.len() as u32;
        words.extend(self.random);
        words
    }
}

pub fn snow(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    intensity: f32,
    anisotropy: f32,
    scale: f32,
) -> Vec<u32> {
    let seed = seeder(seed, frame, noise_seeds::SNOW).finalize();
    let mut result = Snow::new(width, rows);
    for row in 0..rows {
        result.row(row, hash_row(seed, row), intensity, anisotropy, scale);
    }
    result.finish()
}
