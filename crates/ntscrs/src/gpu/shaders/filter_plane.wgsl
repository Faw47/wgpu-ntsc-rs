@group(0) @binding(0) var<storage, read_write> y_plane: array<f32>;
@group(0) @binding(1) var<storage, read_write> i_plane: array<f32>;
@group(0) @binding(2) var<storage, read_write> q_plane: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch_plane: array<f32>;

struct ShaderParams {
    width: u32,
    frame_num: u32,
    seed: u32,
    noise_idx: u32,

    noise_frequency: f32,
    noise_intensity: f32,
    noise_detail: u32,
    snow_anisotropy: f32,

    phase_shift: u32,
    phase_offset: i32,
    filter_mode: u32,
    chroma_delay_horizontal: f32,

    chroma_delay_vertical: i32,
    horizontal_scale: f32,
    vertical_scale: f32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}
@group(1) @binding(0) var<uniform> params: ShaderParams;

struct FilterCoeffs {
    num: vec4<f32>,
    den: vec4<f32>,
    z_initial: vec4<f32>,
    delay: u32,
    filter_len: u32,
    plane_idx: u32, // 0: y, 1: i, 2: q
    // Bit 0: FirstSample initial condition. Bit 1: fused host SIMD arithmetic.
    initial_condition_mode: u32,
}
@group(2) @binding(0) var<uniform> filter_coeffs: FilterCoeffs;

// A four-word unsigned integer, least-significant word first. WGSL permits its
// fma builtin to be evaluated as a multiply followed by an add, so it cannot
// reproduce AVX2/AVX512/NEON filter feedback reliably. These helpers evaluate
// the finite binary32 operation with integer significands and one final
// round-to-nearest-even step.
fn u128_word(value: vec4<u32>, index: u32) -> u32 {
    if (index == 0u) { return value.x; }
    if (index == 1u) { return value.y; }
    if (index == 2u) { return value.z; }
    if (index == 3u) { return value.w; }
    return 0u;
}

fn u128_shift_left(value: vec4<u32>, shift: u32) -> vec4<u32> {
    if (shift >= 128u) { return vec4<u32>(0u); }
    let whole = shift / 32u;
    let part = shift % 32u;
    var result = vec4<u32>(0u);
    for (var dst = 0u; dst < 4u; dst++) {
        if (dst >= whole) {
            result[dst] = u128_word(value, dst - whole) << part;
            if (part != 0u && dst > whole) {
                result[dst] |= u128_word(value, dst - whole - 1u) >> (32u - part);
            }
        }
    }
    return result;
}

fn u128_add(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var result = vec4<u32>(0u);
    var carry = 0u;
    for (var index = 0u; index < 4u; index++) {
        let partial = a[index] + b[index];
        let carry_a = u32(partial < a[index]);
        result[index] = partial + carry;
        let carry_b = u32(result[index] < partial);
        carry = carry_a | carry_b;
    }
    return result;
}

// Requires a >= b.
fn u128_subtract(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
    var result = vec4<u32>(0u);
    var borrow = 0u;
    for (var index = 0u; index < 4u; index++) {
        let partial = a[index] - b[index];
        let borrow_a = u32(a[index] < b[index]);
        result[index] = partial - borrow;
        let borrow_b = u32(partial < borrow);
        borrow = borrow_a | borrow_b;
    }
    return result;
}

fn u128_compare(a: vec4<u32>, b: vec4<u32>) -> i32 {
    if (a.w != b.w) { return select(-1, 1, a.w > b.w); }
    if (a.z != b.z) { return select(-1, 1, a.z > b.z); }
    if (a.y != b.y) { return select(-1, 1, a.y > b.y); }
    if (a.x != b.x) { return select(-1, 1, a.x > b.x); }
    return 0;
}

fn u128_high_bit(value: vec4<u32>) -> i32 {
    if (value.w != 0u) { return 127 - i32(countLeadingZeros(value.w)); }
    if (value.z != 0u) { return 95 - i32(countLeadingZeros(value.z)); }
    if (value.y != 0u) { return 63 - i32(countLeadingZeros(value.y)); }
    if (value.x != 0u) { return 31 - i32(countLeadingZeros(value.x)); }
    return -1;
}

fn u128_bit(value: vec4<u32>, index: u32) -> bool {
    if (index >= 128u) { return false; }
    return ((u128_word(value, index / 32u) >> (index % 32u)) & 1u) != 0u;
}

fn u128_any_below(value: vec4<u32>, count: u32) -> bool {
    if (count >= 128u) {
        return any(value != vec4<u32>(0u));
    }
    let whole = count / 32u;
    for (var index = 0u; index < whole; index++) {
        if (u128_word(value, index) != 0u) { return true; }
    }
    let part = count % 32u;
    if (part != 0u) {
        let mask = (1u << part) - 1u;
        return (u128_word(value, whole) & mask) != 0u;
    }
    return false;
}

fn u128_shift_right_low(value: vec4<u32>, shift: u32) -> u32 {
    if (shift >= 128u) { return 0u; }
    let whole = shift / 32u;
    let part = shift % 32u;
    var result = u128_word(value, whole) >> part;
    if (part != 0u && whole < 3u) {
        result |= u128_word(value, whole + 1u) << (32u - part);
    }
    return result;
}

fn multiply_u24(a: u32, b: u32) -> vec4<u32> {
    let a_low = a & 0xffffu;
    let a_high = a >> 16u;
    let b_low = b & 0xffffu;
    let b_high = b >> 16u;
    let product_low = a_low * b_low;
    let product_mid = a_low * b_high + a_high * b_low;
    let low = product_low + (product_mid << 16u);
    let carry = u32(low < product_low);
    let high = a_high * b_high + (product_mid >> 16u) + carry;
    return vec4<u32>(low, high, 0u, 0u);
}

struct NormalizedF32 {
    mantissa: u32,
    exponent: i32,
}

fn normalize_f32(bits: u32) -> NormalizedF32 {
    let exponent_bits = (bits >> 23u) & 0xffu;
    var mantissa = bits & 0x7fffffu;
    var exponent = -149;
    if (exponent_bits != 0u) {
        mantissa |= 0x800000u;
        exponent = i32(exponent_bits) - 150;
    }
    let shift = countLeadingZeros(mantissa) - 8u;
    return NormalizedF32(mantissa << shift, exponent - i32(shift));
}

// `tie_direction` is zero for an exact represented magnitude, positive when
// an omitted same-sign term moves an exact tie away from zero, and negative
// when an omitted opposite-sign term moves it toward zero.
fn round_finite_f32(
    sign: u32,
    magnitude: vec4<u32>,
    exponent: i32,
    tie_direction: i32,
) -> f32 {
    let high_bit = u128_high_bit(magnitude);
    if (high_bit < 0) { return 0.0; }
    var top_exponent = exponent + high_bit;
    var shift: i32;
    var normal = top_exponent >= -126;
    if (normal) {
        shift = high_bit - 23;
    } else {
        shift = -149 - exponent;
    }

    var rounded: u32;
    var guard = false;
    var sticky = false;
    if (shift > 0) {
        rounded = u128_shift_right_low(magnitude, u32(shift));
        guard = u128_bit(magnitude, u32(shift - 1));
        sticky = u128_any_below(magnitude, u32(shift - 1));
    } else {
        rounded = u128_shift_left(magnitude, u32(-shift)).x;
    }

    if (guard && (sticky || tie_direction > 0 || (tie_direction == 0 && (rounded & 1u) != 0u))) {
        rounded += 1u;
    }

    if (!normal) {
        // Rounding the largest subnormal upward naturally produces the bit
        // pattern for the smallest normal number.
        return bitcast<f32>(sign | rounded);
    }

    if (rounded == 0x1000000u) {
        rounded >>= 1u;
        top_exponent += 1;
    }
    if (top_exponent > 127) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    let exponent_bits = u32(top_exponent + 127) << 23u;
    return bitcast<f32>(sign | exponent_bits | (rounded & 0x7fffffu));
}

fn exact_add_f32(a: f32, b: f32) -> f32 {
    let a_bits = bitcast<u32>(a);
    let b_bits = bitcast<u32>(b);
    let a_exp = (a_bits >> 23u) & 0xffu;
    let b_exp = (b_bits >> 23u) & 0xffu;
    if (a_exp == 0xffu || b_exp == 0xffu) {
        return a + b;
    }
    if ((a_bits & 0x7fffffffu) == 0u || (b_bits & 0x7fffffffu) == 0u) {
        return a + b;
    }

    let a_normal = normalize_f32(a_bits);
    let b_normal = normalize_f32(b_bits);
    let exponent_difference = abs(a_normal.exponent - b_normal.exponent);
    if (exponent_difference > 64) {
        return select(b, a, a_normal.exponent > b_normal.exponent);
    }

    let common_exponent = min(a_normal.exponent, b_normal.exponent);
    let a_magnitude = u128_shift_left(
        vec4<u32>(a_normal.mantissa, 0u, 0u, 0u),
        u32(a_normal.exponent - common_exponent),
    );
    let b_magnitude = u128_shift_left(
        vec4<u32>(b_normal.mantissa, 0u, 0u, 0u),
        u32(b_normal.exponent - common_exponent),
    );
    let a_sign = a_bits & 0x80000000u;
    let b_sign = b_bits & 0x80000000u;
    var result_sign = a_sign;
    var result_magnitude: vec4<u32>;
    if (a_sign == b_sign) {
        result_magnitude = u128_add(a_magnitude, b_magnitude);
    } else {
        let ordering = u128_compare(a_magnitude, b_magnitude);
        if (ordering == 0) { return 0.0; }
        if (ordering > 0) {
            result_magnitude = u128_subtract(a_magnitude, b_magnitude);
        } else {
            result_sign = b_sign;
            result_magnitude = u128_subtract(b_magnitude, a_magnitude);
        }
    }
    return round_finite_f32(result_sign, result_magnitude, common_exponent, 0);
}

fn exact_fma_f32(a: f32, b: f32, c: f32) -> f32 {
    let a_bits = bitcast<u32>(a);
    let b_bits = bitcast<u32>(b);
    let c_bits = bitcast<u32>(c);
    let a_exp = (a_bits >> 23u) & 0xffu;
    let b_exp = (b_bits >> 23u) & 0xffu;
    let c_exp = (c_bits >> 23u) & 0xffu;

    // Filter inputs are finite. Preserve IEEE special-value behavior if a
    // future setting violates that invariant.
    if (a_exp == 0xffu || b_exp == 0xffu || c_exp == 0xffu) {
        return fma(a, b, c);
    }
    if ((a_bits & 0x7fffffffu) == 0u || (b_bits & 0x7fffffffu) == 0u) {
        return a * b + c;
    }

    let a_normal = normalize_f32(a_bits);
    let b_normal = normalize_f32(b_bits);
    var product = multiply_u24(a_normal.mantissa, b_normal.mantissa);
    var product_exponent = a_normal.exponent + b_normal.exponent;
    // The product of two normalized 24-bit significands has its leading bit
    // at 46 or 47. Normalize it to 47 so exponent comparison is sufficient.
    if ((product.y & 0x8000u) == 0u) {
        product = u128_shift_left(product, 1u);
        product_exponent -= 1;
    }
    let product_sign = (a_bits ^ b_bits) & 0x80000000u;

    if ((c_bits & 0x7fffffffu) == 0u) {
        return round_finite_f32(product_sign, product, product_exponent, 0);
    }

    let c_normal = normalize_f32(c_bits);
    // Shift c's 24-bit significand to the same fixed leading-bit position as
    // the product.
    let c_magnitude = vec4<u32>(
        c_normal.mantissa << 24u,
        c_normal.mantissa >> 8u,
        0u,
        0u,
    );
    let c_exponent = c_normal.exponent - 24;
    let c_sign = c_bits & 0x80000000u;
    let exponent_difference = abs(product_exponent - c_exponent);

    if (exponent_difference > 64) {
        if (product_exponent > c_exponent) {
            let tie_direction = select(-1, 1, product_sign == c_sign);
            return round_finite_f32(
                product_sign,
                product,
                product_exponent,
                tie_direction,
            );
        }
        // c is already an exactly representable binary32 value, and the
        // product is much less than half of its least-significant bit.
        return c;
    }

    let common_exponent = min(product_exponent, c_exponent);
    let aligned_product = u128_shift_left(product, u32(product_exponent - common_exponent));
    let aligned_c = u128_shift_left(c_magnitude, u32(c_exponent - common_exponent));
    var result_sign = product_sign;
    var result_magnitude: vec4<u32>;
    if (product_sign == c_sign) {
        result_magnitude = u128_add(aligned_product, aligned_c);
    } else {
        let ordering = u128_compare(aligned_product, aligned_c);
        if (ordering == 0) { return 0.0; }
        if (ordering > 0) {
            result_magnitude = u128_subtract(aligned_product, aligned_c);
        } else {
            result_sign = c_sign;
            result_magnitude = u128_subtract(aligned_c, aligned_product);
        }
    }
    return round_finite_f32(result_sign, result_magnitude, common_exponent, 0);
}

fn upstream_mul_add(a: f32, b: f32, c: f32, fused: bool) -> f32 {
    if (fused) { return exact_fma_f32(a, b, c); }
    // A source-level `a * b + c` is not a portable no-contraction barrier in
    // GPU toolchains. Round the product first, then round the addition.
    let rounded_product = exact_fma_f32(a, b, 0.0);
    return exact_add_f32(rounded_product, c);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row_idx = global_id.x;
    let width = params.width;
    let height = arrayLength(&y_plane) / width;
    
    if (row_idx >= height) {
        return;
    }

    let row_start = row_idx * width;
    var z = filter_coeffs.z_initial;
    let num = filter_coeffs.num;
    let den = filter_coeffs.den;
    let filter_len = filter_coeffs.filter_len;
    let delay = filter_coeffs.delay;
    let plane_idx = filter_coeffs.plane_idx;

    // Matches TransferFunction::initial_condition_into (scipy) for FirstSample, like the CPU path.
    if ((filter_coeffs.initial_condition_mode & 1u) != 0u) {
        var initial_val: f32;
        if (plane_idx == 0u) { initial_val = y_plane[row_start]; }
        else if (plane_idx == 1u) { initial_val = i_plane[row_start]; }
        else if (plane_idx == 2u) { initial_val = q_plane[row_start]; }
        else { initial_val = scratch_plane[row_start]; }

        z *= initial_val;
    }

    // We process width + delay iterations.
    for (var i = 0u; i < width + delay; i++) {
        let read_idx = row_start + min(i, width - 1u);
        var sample_val: f32;
        if (plane_idx == 0u) { sample_val = y_plane[read_idx]; }
        else if (plane_idx == 1u) { sample_val = i_plane[read_idx]; }
        else if (plane_idx == 2u) { sample_val = q_plane[read_idx]; }
        else { sample_val = scratch_plane[read_idx]; }

        let fused = (filter_coeffs.initial_condition_mode & 2u) != 0u;
        let filt_sample = upstream_mul_add(num.x, sample_val, z.x, fused);

        if (filter_len > 1u) {
            let filtered_y = upstream_mul_add(num.y, sample_val, z.y, fused);
            z.x = upstream_mul_add(-den.x, filt_sample, filtered_y, fused);
        }
        if (filter_len > 2u) {
            let filtered_z = upstream_mul_add(num.z, sample_val, z.z, fused);
            z.y = upstream_mul_add(-den.y, filt_sample, filtered_z, fused);
        }
        if (filter_len > 3u) {
            let filtered_w = upstream_mul_add(num.w, sample_val, z.w, fused);
            z.z = upstream_mul_add(-den.z, filt_sample, filtered_w, fused);
        }

        if (i >= delay) {
            let write_idx = row_start + i - delay;
            if (plane_idx == 0u) { y_plane[write_idx] = filt_sample; }
            else if (plane_idx == 1u) { i_plane[write_idx] = filt_sample; }
            else if (plane_idx == 2u) { q_plane[write_idx] = filt_sample; }
            else { scratch_plane[write_idx] = filt_sample; }
        }
    }
}
