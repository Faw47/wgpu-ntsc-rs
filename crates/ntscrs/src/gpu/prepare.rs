//! Compact stochastic control data. The CPU generates the reference RNG sequence;
//! shaders evaluate noise and transform image pixels. No image is processed here.
use crate::{
    noise::{Fbm, Simplex, Simplex1d, Simplex2d, sample_noise_1d, sample_noise_2d},
    noise_seeds,
    random::{Geometric, Seeder},
    settings::standard::*,
    thread_pool::{self, ZipChunks, with_thread_pool},
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
    let mut result = vec![Row::default(); rows];
    ZipChunks::new([result.as_mut_slice()], 1).par_for_each(|row, [slot]| {
        slot[0] = noise_row(
            &seeder,
            row,
            width,
            settings.frequency / scale,
            settings.intensity,
            settings.detail,
        );
    });
    result
}

pub fn phase(seed: i32, frame: usize, rows: usize, intensity: f32) -> Vec<Row> {
    let seed = seeder(seed, frame, noise_seeds::VIDEO_CHROMA_PHASE).finalize::<u64>();
    let mut result = vec![Row::default(); rows];
    ZipChunks::new([result.as_mut_slice()], 1).par_for_each(|row, [slot]| {
        let val = hash_row(seed, row) as f32 / (u64::MAX as f32 + 1.0);
        let (sin, cos) =
            (((val - 0.5) * 2.0 * intensity) * std::f32::consts::PI * 2.0).sin_cos();
        slot[0] = Row {
            phase: sin,
            frequency: cos,
            ..Row::default()
        };
    });
    result
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
    ZipChunks::new([&mut result[start..]], 1).par_for_each(|local, [slot]| {
        let row = &mut slot[0];
        let index = affected - (local + cutoff);
        let shift = settings.horiz_shift * ((index + offset) as f32 / count as f32).powf(1.5);
        row.shift = (shift + seeder.clone().mix(index).finalize::<f32>() - 0.5) * sx;
        row.start = 0;
        if index == affected
            && let Some(mid) = &settings.mid_line
        {
            let seeder = self::seeder(
                seed,
                frame,
                noise_seeds::HEAD_SWITCHING_MID_LINE_JITTER,
            );
            let random = (seeder.clone().mix(0).finalize::<f32>()
                + seeder.clone().mix(1).finalize::<f32>())
                * 0.5;
            row.start = (width as f32 * (mid.position + (random - 0.5) * mid.jitter)) as u32;
            row.transient_len = 16.0 * sx;
            row.transient_intensity = (seeder.mix(0).finalize::<f32>() + 0.5) * 0.5;
        }
    });
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

#[derive(Default)]
struct SnowRow {
    events: Vec<SnowEvent>,
    random: Vec<u32>,
}

struct SnowEvent {
    start: isize,
    len: f32,
    frequency: f32,
    visible: usize,
    end: usize,
    random_start: usize,
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
        self.append_row(
            row,
            generate_snow_row(self.width, seed, intensity, anisotropy, scale),
        );
    }
    fn append_row(&mut self, row: usize, generated: SnowRow) {
        let random_base = self.random.len();
        for event in generated.events {
            let event_idx = self.events.len() as u32;
            self.events.push([
                event.start as i32 as u32,
                event.len.to_bits(),
                event.frequency.to_bits(),
                (random_base + event.random_start) as u32,
            ]);
            for tile in event.visible / 32..=(event.end - 1) / 32 {
                self.tiles[row * self.width.div_ceil(32) + tile].push(event_idx);
            }
        }
        self.random.extend(generated.random);
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

fn generate_snow_row(
    width: usize,
    seed: u64,
    intensity: f32,
    anisotropy: f32,
    scale: f32,
) -> SnowRow {
    let mut generated = SnowRow::default();
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
        return generated;
    }
    let dist = Geometric::new(probability);
    let mut start = -64isize;
    loop {
        start = start.saturating_add(rng.sample(&dist).min(isize::MAX as usize) as isize);
        if start >= width as isize {
            break;
        }
        let len = rng.random_range(8.0..=64.0) * scale;
        let freq = rng.random_range(len * 3.0..=len * 5.0);
        let end = start
            .saturating_add(len.ceil() as isize)
            .clamp(0, width as isize) as usize;
        rng.jump();
        let mut event_rng = rng.clone();
        let visible = start.clamp(0, width as isize) as usize;
        if visible < end {
            let random_start = generated.random.len();
            for _ in visible..end {
                generated
                    .random
                    .push(event_rng.random_range(-1.0f32..2.0).to_bits());
            }
            generated.events.push(SnowEvent {
                start,
                len,
                frequency: freq,
                visible,
                end,
                random_start,
            });
        }
        start += 1;
    }
    generated
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
    let mut generated: Vec<SnowRow> = (0..rows).map(|_| SnowRow::default()).collect();
    ZipChunks::new([generated.as_mut_slice()], 1).par_for_each(|row, [slot]| {
        slot[0] = generate_snow_row(
            width,
            hash_row(seed, row),
            intensity,
            anisotropy,
            scale,
        );
    });
    let mut result = Snow::new(width, rows);
    for (row, generated) in generated.into_iter().enumerate() {
        result.append_row(row, generated);
    }
    result.finish()
}

/// All CPU-generated control data for one field. Each stream uses its own reference seed, so
/// streams can be prepared concurrently without changing RNG advancement or effect ordering.
#[derive(Default)]
pub struct Controls {
    pub composite_noise: Option<Vec<Row>>,
    pub snow: Option<Vec<u32>>,
    pub head: Option<Vec<Row>>,
    pub tracking: Option<(Vec<Row>, Vec<u32>)>,
    pub luma_noise: Option<Vec<Row>>,
    pub chroma_noise: Option<(Vec<Row>, Vec<Row>)>,
    pub phase_noise: Option<Vec<Row>>,
    pub edge_wave: Option<Vec<Row>>,
    pub chroma_loss: Option<Vec<Row>>,
}

pub fn controls(
    effect: &NtscEffect,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
) -> Controls {
    let vhs = effect.vhs_settings.as_ref();
    let has_controls = effect.composite_noise.is_some()
        || (effect.snow_intensity > 0.0 && sx > 0.0)
        || effect.head_switching.is_some()
        || effect.tracking_noise.is_some()
        || effect.luma_noise.is_some()
        || effect.chroma_noise.is_some()
        || effect.chroma_phase_noise_intensity > 0.0
        || vhs
            .and_then(|settings| settings.edge_wave.as_ref())
            .is_some_and(|settings| settings.intensity > 0.0)
        || vhs.is_some_and(|settings| settings.chroma_loss > 0.0);
    if !has_controls {
        return Controls::default();
    }

    with_thread_pool(|| {
        let (
            ((composite_noise, snow), (head, tracking)),
            ((luma_noise, chroma_noise), (phase_noise, (edge_wave, chroma_loss))),
        ) = thread_pool::join(
            || {
                thread_pool::join(
                    || {
                        thread_pool::join(
                            || {
                                effect.composite_noise.as_ref().map(|settings| {
                                    noise(
                                        effect.random_seed,
                                        frame,
                                        noise_seeds::VIDEO_COMPOSITE,
                                        width,
                                        rows,
                                        sx,
                                        settings,
                                    )
                                })
                            },
                            || {
                                (effect.snow_intensity > 0.0 && sx > 0.0).then(|| {
                                    snow(
                                        effect.random_seed,
                                        frame,
                                        width,
                                        rows,
                                        effect.snow_intensity * 0.01,
                                        effect.snow_anisotropy,
                                        sx,
                                    )
                                })
                            },
                        )
                    },
                    || {
                        thread_pool::join(
                            || {
                                effect.head_switching.as_ref().map(|settings| {
                                    head(
                                        effect.random_seed,
                                        frame,
                                        width,
                                        rows,
                                        sx,
                                        sy,
                                        settings,
                                    )
                                })
                            },
                            || {
                                effect.tracking_noise.as_ref().map(|settings| {
                                    tracking(
                                        effect.random_seed,
                                        frame,
                                        width,
                                        rows,
                                        sx,
                                        sy,
                                        settings,
                                    )
                                })
                            },
                        )
                    },
                )
            },
            || {
                thread_pool::join(
                    || {
                        thread_pool::join(
                            || {
                                effect.luma_noise.as_ref().map(|settings| {
                                    noise(
                                        effect.random_seed,
                                        frame,
                                        noise_seeds::VIDEO_LUMA,
                                        width,
                                        rows,
                                        sx,
                                        settings,
                                    )
                                })
                            },
                            || {
                                effect.chroma_noise.as_ref().map(|settings| {
                                    thread_pool::join(
                                        || {
                                            noise(
                                                effect.random_seed,
                                                frame,
                                                noise_seeds::VIDEO_CHROMA_I,
                                                width,
                                                rows,
                                                sx,
                                                settings,
                                            )
                                        },
                                        || {
                                            noise(
                                                effect.random_seed,
                                                frame,
                                                noise_seeds::VIDEO_CHROMA_Q,
                                                width,
                                                rows,
                                                sx,
                                                settings,
                                            )
                                        },
                                    )
                                })
                            },
                        )
                    },
                    || {
                        thread_pool::join(
                            || {
                                (effect.chroma_phase_noise_intensity > 0.0).then(|| {
                                    phase(
                                        effect.random_seed,
                                        frame,
                                        rows,
                                        effect.chroma_phase_noise_intensity,
                                    )
                                })
                            },
                            || {
                                thread_pool::join(
                                    || {
                                        effect
                                            .vhs_settings
                                            .as_ref()
                                            .and_then(|vhs| vhs.edge_wave.as_ref())
                                            .filter(|settings| settings.intensity > 0.0)
                                            .map(|settings| {
                                                wave(
                                                    effect.random_seed,
                                                    frame,
                                                    rows,
                                                    sx,
                                                    sy,
                                                    settings,
                                                )
                                            })
                                    },
                                    || {
                                        effect.vhs_settings.as_ref().and_then(|vhs| {
                                            (vhs.chroma_loss > 0.0).then(|| {
                                                loss(
                                                    effect.random_seed,
                                                    frame,
                                                    rows,
                                                    vhs.chroma_loss,
                                                )
                                            })
                                        })
                                    },
                                )
                            },
                        )
                    },
                )
            },
        );

        Controls {
            composite_noise,
            snow,
            head,
            tracking,
            luma_noise,
            chroma_noise,
            phase_noise,
            edge_wave,
            chroma_loss,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_noise_rows_preserve_serial_control_bytes() {
        let settings = FbmNoiseSettings {
            frequency: 0.173,
            intensity: 0.41,
            detail: 3,
        };
        let seeder = seeder(-47, 13, noise_seeds::VIDEO_LUMA);
        let expected: Vec<Row> = (0..257)
            .map(|row| noise_row(&seeder, row, 641, settings.frequency / 1.25, settings.intensity, 3))
            .collect();
        let actual = noise(
            -47,
            13,
            noise_seeds::VIDEO_LUMA,
            641,
            257,
            1.25,
            &settings,
        );
        assert_eq!(
            bytemuck::cast_slice::<Row, u8>(&actual),
            bytemuck::cast_slice::<Row, u8>(&expected)
        );
    }

    #[test]
    fn parallel_snow_merge_preserves_serial_control_bytes() {
        for (width, rows, intensity, anisotropy, scale) in [
            (1, 1, 0.01, 0.0, 1.0),
            (65, 33, 0.5, 0.5, 1.25),
            (257, 67, 1.0, 1.0, 0.75),
        ] {
            let seed = seeder(-47, 13, noise_seeds::SNOW).finalize();
            let mut serial = Snow::new(width, rows);
            for row in 0..rows {
                serial.row(
                    row,
                    hash_row(seed, row),
                    intensity,
                    anisotropy,
                    scale,
                );
            }
            assert_eq!(
                snow(-47, 13, width, rows, intensity, anisotropy, scale),
                serial.finish()
            );
        }
    }
}
