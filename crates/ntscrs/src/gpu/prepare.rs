//! Compact stochastic control data. The CPU generates the reference RNG sequence;
//! shaders evaluate noise and transform image pixels. No image is processed here.
use crate::{
    noise::{
        Fbm, Simplex, Simplex1d, Simplex2d, active_f32_lane_count, sample_noise_1d, sample_noise_2d,
    },
    noise_seeds,
    random::{SplitMix64, geometric_lambda},
    settings::standard::*,
    thread_pool::{ZipChunks, with_thread_pool},
};
use fearless_simd::Level;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
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
    /// SIMD lane count for noise rows; chroma-loss flag for loss rows. These stages
    /// never share a row payload, so the slot is intentionally reused.
    pub loss: u32,
}

pub fn stage_rng(seed: i32, frame: usize, tag: u64) -> SplitMix64 {
    SplitMix64::new(seed as u32 as u64)
        .mix(tag)
        .mix(frame as u64)
}

fn noise_row(
    rng: &SplitMix64,
    index: usize,
    width: usize,
    frequency: f32,
    intensity: f32,
    detail: i32,
    lane_count: usize,
) -> Row {
    let mut rng = rng.clone().mix(index as u64);
    Row {
        noise_seed: rng.random::<i32>(),
        offset: rng.random::<f32>() * width as f32,
        frequency,
        intensity: intensity * 0.25,
        octaves: (detail.max(0) as u32).clamp(1, 5),
        loss: lane_count as u32,
        ..Row::default()
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn noise(
    seed: i32,
    frame: usize,
    tag: u64,
    width: usize,
    rows: usize,
    scale: f32,
    settings: &FbmNoiseSettings,
) -> Vec<Row> {
    let mut result = Vec::new();
    noise_into(seed, frame, tag, width, rows, scale, settings, &mut result);
    result
}

pub fn noise_into(
    seed: i32,
    frame: usize,
    tag: u64,
    width: usize,
    rows: usize,
    scale: f32,
    settings: &FbmNoiseSettings,
    mut result: &mut Vec<Row>,
) {
    let rng = stage_rng(seed, frame, tag);
    let lane_count = active_f32_lane_count();
    result.clear();
    result.resize(rows, Row::default());
    with_thread_pool(|| {
        ZipChunks::new([&mut result], 1).par_for_each(|row, [slot]| {
            slot[0] = noise_row(
                &rng,
                row,
                width,
                settings.frequency / scale,
                settings.intensity,
                settings.detail,
                lane_count,
            );
        });
    });
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn phase(seed: i32, frame: usize, rows: usize, intensity: f32) -> Vec<Row> {
    let mut result = Vec::new();
    phase_into(seed, frame, rows, intensity, &mut result);
    result
}

pub fn phase_into(seed: i32, frame: usize, rows: usize, intensity: f32, mut result: &mut Vec<Row>) {
    let rng = stage_rng(seed, frame, noise_seeds::VIDEO_CHROMA_PHASE);
    result.clear();
    result.resize(rows, Row::default());
    with_thread_pool(|| {
        ZipChunks::new([&mut result], 1).par_for_each(|row, [slot]| {
            let val = rng.clone().mix(row as u64).random::<f32>();
            let (sin, cos) =
                (((val - 0.5) * 2.0 * intensity) * std::f32::consts::PI * 2.0).sin_cos();
            slot[0].phase = sin;
            slot[0].frequency = cos;
        });
    });
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn head(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &HeadSwitchingSettings,
) -> Vec<Row> {
    let mut result = Vec::new();
    head_into(seed, frame, width, rows, sx, sy, settings, &mut result);
    result
}

pub fn head_into(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &HeadSwitchingSettings,
    result: &mut Vec<Row>,
) {
    result.clear();
    result.resize(
        rows,
        Row {
            start: width as u32,
            ..Row::default()
        },
    );
    let count = (settings.height.max(0) as f32 * sy).round() as usize;
    let offset = (settings.offset.max(0) as f32 * sy).round() as usize;
    if offset >= count {
        return;
    }
    let affected = count - offset;
    let start = rows.saturating_sub(affected);
    let cutoff = affected.saturating_sub(rows);
    let rng = stage_rng(seed, frame, noise_seeds::HEAD_SWITCHING);
    for (index, row) in result[start..].iter_mut().enumerate() {
        let index = affected - (index + cutoff);
        let shift = settings.horiz_shift * ((index + offset) as f32 / count as f32).powf(1.5);
        row.shift = (shift + (rng.clone().mix(index as u64).random::<f32>() - 0.5)) * sx;
        row.start = 0;
        if index == affected
            && let Some(mid) = &settings.mid_line
        {
            let mut rng = stage_rng(seed, frame, noise_seeds::HEAD_SWITCHING_MID_LINE_JITTER);
            let random = (rng.random::<f32>() + rng.random::<f32>()) * 0.5;
            row.start = (width as f32 * (mid.position + (random - 0.5) * mid.jitter)) as u32;
            row.transient_len = 16.0 * sx;
            row.transient_intensity = (rng.random::<f32>() + 0.5) * 0.5;
        }
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn wave(
    seed: i32,
    frame: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &VHSEdgeWaveSettings,
) -> Vec<Row> {
    let mut result = Vec::new();
    wave_into(seed, frame, rows, sx, sy, settings, &mut result);
    result
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn wave_into(
    seed: i32,
    frame: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &VHSEdgeWaveSettings,
    result: &mut Vec<Row>,
) {
    let mut shifts = Vec::new();
    wave_into_with_scratch(seed, frame, rows, sx, sy, settings, &mut shifts, result);
}

pub fn wave_into_with_scratch(
    seed: i32,
    frame: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &VHSEdgeWaveSettings,
    shifts: &mut Vec<f32>,
    result: &mut Vec<Row>,
) {
    let mut rng = SplitMix64::new(seed as u32 as u64).mix(noise_seeds::EDGE_WAVE);
    let noise = Fbm {
        seed: rng.random::<i32>(),
        octaves: settings.detail.clamp(1, 5) as usize,
        gain: std::f32::consts::FRAC_1_SQRT_2,
        lacunarity: 2.0,
        frequency: settings.frequency / sy,
    };
    let offset = rng.random::<f32>() * rows as f32;
    shifts.clear();
    shifts.resize(rows, 0.0);
    sample_noise_2d::<Simplex2d, _>(
        Level::new(),
        &noise,
        [offset, frame as f32 * settings.speed],
        [rows, 1],
        shifts,
    );
    result.clear();
    result.extend(shifts.iter().copied().map(|v| Row {
        shift: (v / 0.022) * settings.intensity * 0.5 * sx,
        extend: 1,
        ..Row::default()
    }));
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn loss(seed: i32, frame: usize, rows: usize, intensity: f32) -> Vec<Row> {
    let mut result = Vec::new();
    loss_into(seed, frame, rows, intensity, &mut result);
    result
}

pub fn loss_into(seed: i32, frame: usize, rows: usize, intensity: f32, result: &mut Vec<Row>) {
    result.clear();
    result.resize(rows, Row::default());
    let mut rng = stage_rng(seed, frame, noise_seeds::CHROMA_LOSS);
    let dist = geometric_lambda(intensity as f64);
    let mut row = 0usize;
    loop {
        row = row.saturating_add(rng.random_geometric(dist));
        if row >= rows {
            break;
        }
        result[row].loss = 1;
        row += 1;
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn tracking(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &TrackingNoiseSettings,
) -> (Vec<Row>, Vec<u32>) {
    let mut result = Vec::new();
    let mut snow = Vec::new();
    tracking_into(
        seed,
        frame,
        width,
        rows,
        sx,
        sy,
        settings,
        &mut result,
        &mut snow,
    );
    (result, snow)
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn tracking_into(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &TrackingNoiseSettings,
    result: &mut Vec<Row>,
    events: &mut Vec<u32>,
) {
    let mut shifts = Vec::new();
    let mut snow = SnowScratch::new(width, rows);
    tracking_into_with_scratch(
        seed,
        frame,
        width,
        rows,
        sx,
        sy,
        settings,
        &mut shifts,
        &mut snow,
        result,
        events,
    );
}

pub fn tracking_into_with_scratch(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    sx: f32,
    sy: f32,
    settings: &TrackingNoiseSettings,
    shifts: &mut Vec<f32>,
    snow: &mut SnowScratch,
    result: &mut Vec<Row>,
    events: &mut Vec<u32>,
) {
    let count = (settings.height.max(0) as f32 * sy).round() as usize;
    let start = rows.saturating_sub(count);
    let cutoff = count.saturating_sub(rows);
    let mut rng = stage_rng(seed, frame, noise_seeds::TRACKING_NOISE);
    let noise = Simplex {
        seed: rng.random::<i32>(),
        frequency: 0.5,
    };
    let offset = rng.random::<f32>() * rows as f32;
    shifts.clear();
    shifts.resize(count.min(rows), 0.0);
    sample_noise_1d::<Simplex1d, _>(Level::new(), &noise, [offset], [shifts.len()], shifts);
    result.clear();
    result.resize(
        rows,
        Row {
            start: width as u32,
            ..Row::default()
        },
    );
    let lane_count = active_f32_lane_count();
    snow.0.reset(width, rows);
    for (local, row) in result[start..].iter_mut().enumerate() {
        let index = local + cutoff;
        let intensity = index as f32 / count as f32;
        *row = noise_row(
            &rng,
            index,
            width,
            0.25 / sx,
            intensity.powi(2) * settings.noise_intensity * 4.0,
            1,
            lane_count,
        );
        row.shift = shifts[local] * intensity * settings.wave_intensity * 0.25 * sx;
        snow.0.row(
            start + local,
            rng.clone().mix(index as u64),
            settings.snow_intensity * intensity.powi(2),
            settings.snow_anisotropy,
            sx,
        );
    }
    events.clear();
    snow.0.finish_into(events);
}

// Snow events are binned into 32-pixel tiles. Each pixel evaluates only overlapping
// events in reference order, without float atomics or a serial whole-row shader.
struct Snow {
    width: usize,
    tiles: Vec<Vec<u32>>,
    events: Vec<[u32; 4]>,
    random: Vec<u32>,
}

/// Reusable storage for sparse snow events. A backend can keep one instance per
/// worker and avoid rebuilding the tile index and event vectors for every frame.
pub struct SnowScratch(Snow);

impl SnowScratch {
    pub fn new(width: usize, rows: usize) -> Self {
        Self(Snow::new(width, rows))
    }
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

    fn reset(&mut self, width: usize, rows: usize) {
        let tile_count = width.div_ceil(32) * rows;
        if self.width != width || self.tiles.len() != tile_count {
            self.width = width;
            self.tiles = vec![Vec::new(); tile_count];
        } else {
            self.tiles.iter_mut().for_each(Vec::clear);
        }
        self.events.clear();
        self.random.clear();
    }
    fn row(
        &mut self,
        row: usize,
        mut rng: SplitMix64,
        intensity: f32,
        anisotropy: f32,
        scale: f32,
    ) {
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
        let dist = geometric_lambda(probability);
        let mut start = -64isize;
        loop {
            start =
                start.saturating_add(rng.random_geometric(dist).min(isize::MAX as usize) as isize);
            if start >= self.width as isize {
                break;
            }
            let len = rng.random_range(8.0..64.0) * scale;
            let freq = rng.random_range(len * 3.0..len * 5.0);
            let end = start
                .saturating_add(len.ceil() as isize)
                .clamp(0, self.width as isize) as usize;
            let mut event_rng = SplitMix64::new(rng.random());
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
    fn finish_into(&self, words: &mut Vec<u32>) {
        words.clear();
        if self.events.is_empty() {
            return;
        }
        words.resize(4 + self.tiles.len() + 1, 0);
        for (tile, events) in self.tiles.iter().enumerate() {
            words[4 + tile] = words.len() as u32;
            words.extend_from_slice(events);
        }
        words[4 + self.tiles.len()] = words.len() as u32;
        words[0] = words.len() as u32;
        for event in &self.events {
            words.extend_from_slice(event);
        }
        words[1] = words.len() as u32;
        words.extend_from_slice(&self.random);
    }

    #[allow(dead_code)]
    fn finish(self) -> Vec<u32> {
        let mut words = Vec::new();
        self.finish_into(&mut words);
        words
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn snow(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    intensity: f32,
    anisotropy: f32,
    scale: f32,
) -> Vec<u32> {
    let mut result = Vec::new();
    snow_into(
        seed,
        frame,
        width,
        rows,
        intensity,
        anisotropy,
        scale,
        &mut result,
    );
    result
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn snow_into(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    intensity: f32,
    anisotropy: f32,
    scale: f32,
    result: &mut Vec<u32>,
) {
    let mut scratch = SnowScratch::new(width, rows);
    snow_into_with_scratch(
        seed,
        frame,
        width,
        rows,
        intensity,
        anisotropy,
        scale,
        &mut scratch,
        result,
    );
}

pub fn snow_into_with_scratch(
    seed: i32,
    frame: usize,
    width: usize,
    rows: usize,
    intensity: f32,
    anisotropy: f32,
    scale: f32,
    scratch: &mut SnowScratch,
    result: &mut Vec<u32>,
) {
    let rng = stage_rng(seed, frame, noise_seeds::SNOW);
    scratch.0.reset(width, rows);
    for row in 0..rows {
        scratch.0.row(
            row,
            rng.clone().mix(row as u64),
            intensity,
            anisotropy,
            scale,
        );
    }
    scratch.0.finish_into(result);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_fingerprint(rows: &[Row]) -> u64 {
        word_fingerprint(bytemuck::cast_slice(rows))
    }

    fn word_fingerprint(words: &[u32]) -> u64 {
        words.iter().fold(0xcbf2_9ce4_8422_2325, |hash, word| {
            (hash ^ u64::from(*word)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    fn noise_reference(
        seed: i32,
        frame: usize,
        tag: u64,
        width: usize,
        rows: usize,
        scale: f32,
        settings: &FbmNoiseSettings,
    ) -> Vec<Row> {
        let rng = stage_rng(seed, frame, tag);
        let lane_count = active_f32_lane_count();
        (0..rows)
            .map(|row| {
                noise_row(
                    &rng,
                    row,
                    width,
                    settings.frequency / scale,
                    settings.intensity,
                    settings.detail,
                    lane_count,
                )
            })
            .collect()
    }

    fn phase_reference(seed: i32, frame: usize, rows: usize, intensity: f32) -> Vec<Row> {
        let rng = stage_rng(seed, frame, noise_seeds::VIDEO_CHROMA_PHASE);
        (0..rows)
            .map(|row| {
                let val = rng.clone().mix(row as u64).random::<f32>();
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

    #[test]
    fn independent_row_preparation_matches_sequential_reference() {
        let settings = FbmNoiseSettings {
            frequency: 1.7,
            intensity: 0.42,
            detail: 4,
        };
        let parallel = noise(-47, 13, noise_seeds::VIDEO_LUMA, 65, 33, 1.25, &settings);
        let sequential = noise_reference(-47, 13, noise_seeds::VIDEO_LUMA, 65, 33, 1.25, &settings);
        assert_eq!(parallel, sequential);

        let parallel = phase(-47, 13, 33, 0.075);
        let sequential = phase_reference(-47, 13, 33, 0.075);
        assert_eq!(parallel, sequential);
    }

    #[test]
    fn head_shift_uses_pinned_upstream_expression_grouping() {
        let settings = HeadSwitchingSettings {
            height: 12,
            offset: 2,
            horiz_shift: 5.0,
            mid_line: None,
        };
        let sx = 0.125;
        let sy = 1.0;
        let width = 65;
        let rows = 33;
        let actual = head(-47, 13, width, rows, sx, sy, &settings);

        let count = (settings.height.max(0) as f32 * sy).round() as usize;
        let offset = (settings.offset.max(0) as f32 * sy).round() as usize;
        let affected = count - offset;
        let start = rows.saturating_sub(affected);
        let cutoff = affected.saturating_sub(rows);
        let rng = stage_rng(-47, 13, noise_seeds::HEAD_SWITCHING);
        for (local, row) in actual[start..].iter().enumerate() {
            let index = affected - (local + cutoff);
            let shift = settings.horiz_shift * ((index + offset) as f32 / count as f32).powf(1.5);
            let expected = (shift + (rng.clone().mix(index as u64).random::<f32>() - 0.5)) * sx;
            assert_eq!(
                row.shift.to_bits(),
                expected.to_bits(),
                "head row {local} does not match pinned upstream grouping"
            );
        }
    }

    #[test]
    fn fixed_seed_stochastic_control_fingerprints() {
        let noise_settings = FbmNoiseSettings {
            frequency: 1.7,
            intensity: 0.42,
            detail: 4,
        };
        for (tag, expected) in [
            (noise_seeds::VIDEO_COMPOSITE, 3_485_735_891_285_444_892),
            (noise_seeds::VIDEO_LUMA, 12_447_564_389_534_145_545),
            (noise_seeds::VIDEO_CHROMA_I, 11_125_566_137_780_981_240),
            (noise_seeds::VIDEO_CHROMA_Q, 15_709_688_246_788_731_364),
        ] {
            let rows = noise(-47, 13, tag, 65, 33, 1.25, &noise_settings);
            assert!(
                rows.iter()
                    .all(|row| row.loss as usize == active_f32_lane_count())
            );
            let mut portable_rows = rows;
            portable_rows.iter_mut().for_each(|row| row.loss = 0);
            assert_eq!(row_fingerprint(&portable_rows), expected, "noise tag {tag}");
        }

        assert_eq!(
            row_fingerprint(&phase(-47, 13, 33, 0.075)),
            13_864_946_719_853_334_478
        );
        assert_eq!(
            row_fingerprint(&head(
                -47,
                13,
                65,
                33,
                1.25,
                0.75,
                &HeadSwitchingSettings::default(),
            )),
            7_299_772_395_079_915_787
        );
        let (mut tracking_rows, tracking_snow) = tracking(
            -47,
            13,
            65,
            33,
            1.25,
            0.75,
            &TrackingNoiseSettings::default(),
        );
        assert!(
            tracking_rows
                .iter()
                .any(|row| row.loss as usize == active_f32_lane_count())
        );
        tracking_rows.iter_mut().for_each(|row| row.loss = 0);
        assert_eq!(row_fingerprint(&tracking_rows), 7_011_883_219_807_750_076);
        assert_eq!(word_fingerprint(&tracking_snow), 14_695_981_039_346_656_037);
        assert_eq!(
            row_fingerprint(&wave(
                -47,
                13,
                33,
                1.25,
                0.75,
                &VHSEdgeWaveSettings::default(),
            )),
            3_755_879_939_263_884_841
        );
        assert_eq!(
            row_fingerprint(&loss(-47, 13, 33, 0.35)),
            7_927_523_142_538_073_797
        );
        assert_eq!(
            word_fingerprint(&snow(-47, 13, 65, 33, 0.25, 0.5, 1.25)),
            8_365_921_260_893_723_034
        );
    }
}
