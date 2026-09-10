fn hash_1d(val: u32) -> u32 {
    var h = (val ^ 2747636419u) * 2654435769u;
    h = (h ^ (h >> 16u)) * 2654435769u;
    h = (h ^ (h >> 16u)) * 2654435769u;
    return h;
}

fn pcg_3d(v: vec3<u32>) -> vec3<u32> {
    var vx = v.x * 1664525u + 1013904223u;
    var vy = v.y * 1664525u + 1013904223u;
    var vz = v.z * 1664525u + 1013904223u;

    vx += vy * vz;
    vy += vz * vx;
    vz += vx * vy;

    vx ^= (vx >> 16u);
    vy ^= (vy >> 16u);
    vz ^= (vz >> 16u);

    vx += vy * vz;
    vy += vz * vx;
    vz += vx * vy;

    return vec3<u32>(vx, vy, vz);
}

fn gradient_1d(hash: u32) -> f32 {
    let h = hash >> 28u;
    var v = f32((h & 7u) + 1u);
    if ((h & 8u) != 0u) {
        v = -v;
    }
    return v;
}

fn gradient_2d(hash: u32) -> vec2<f32> {
    let h = hash & 7u;
    let x_mag = select(2.0, 1.0, h < 4u);
    let y_mag = select(1.0, 2.0, h < 4u);

    let gx = select(
        select(x_mag, -x_mag, (h & 2u) != 0u),
        select(x_mag, -x_mag, (h & 1u) != 0u),
        h < 4u
    );
    let gy = select(
        select(y_mag, -y_mag, (h & 1u) != 0u),
        select(y_mag, -y_mag, (h & 2u) != 0u),
        h < 4u
    );
    return vec2<f32>(gx, gy);
}

fn simplex_1d(x: f32, seed: i32) -> f32 {
    let i = floor(x);
    let i0 = i32(i);
    let i1 = i0 + 1;
    let x0 = x - i;
    let x1 = x0 - 1.0;

    let gi0 = hash_1d(bitcast<u32>(i0 ^ seed));
    let gi1 = hash_1d(bitcast<u32>(i1 ^ seed));

    let t0 = 1.0 - (x0 * x0);
    let t20 = t0 * t0;
    let t40 = t20 * t20;
    let gx0 = gradient_1d(gi0);

    let t1 = 1.0 - (x1 * x1);
    let t21 = t1 * t1;
    let t41 = t21 * t21;
    let gx1 = gradient_1d(gi1);

    return (t40 * gx0 * x0) + (t41 * gx1 * x1);
}

fn simplex_2d(point: vec2<f32>, seed: i32) -> f32 {
    let skew = 0.36602540378;
    let unskew = 0.2113248654;

    let s = (point.x + point.y) * skew;
    let ips = floor(point.x + s);
    let jps = floor(point.y + s);

    let i = i32(ips);
    let j = i32(jps);

    let t = f32(i + j) * unskew;

    let x0 = point.x - (ips - t);
    let y0 = point.y - (jps - t);

    let i1 = i32(x0 >= y0);
    let j1 = i32(y0 > x0);

    let x1 = x0 - f32(i1) + unskew;
    let y1 = y0 - f32(j1) + unskew;
    let x2 = x0 - 1.0 + 2.0 * unskew;
    let y2 = y0 - 1.0 + 2.0 * unskew;

    let gi0 = pcg_3d(vec3<u32>(bitcast<u32>(i), bitcast<u32>(j), bitcast<u32>(seed))).x;
    let gi1 = pcg_3d(vec3<u32>(bitcast<u32>(i + i1), bitcast<u32>(j + j1), bitcast<u32>(seed))).x;
    let gi2 = pcg_3d(vec3<u32>(bitcast<u32>(i + 1), bitcast<u32>(j + 1), bitcast<u32>(seed))).x;

    let t0 = max(0.0, 0.5 - x0*x0 - y0*y0);
    let t1 = max(0.0, 0.5 - x1*x1 - y1*y1);
    let t2 = max(0.0, 0.5 - x2*x2 - y2*y2);

    let t20 = t0 * t0; let t40 = t20 * t20;
    let t21 = t1 * t1; let t41 = t21 * t21;
    let t22 = t2 * t2; let t42 = t22 * t22;

    let g0 = gradient_2d(gi0); let n0 = t40 * dot(g0, vec2<f32>(x0, y0));
    let g1 = gradient_2d(gi1); let n1 = t41 * dot(g1, vec2<f32>(x1, y1));
    let g2 = gradient_2d(gi2); let n2 = t42 * dot(g2, vec2<f32>(x2, y2));

    return n0 + n1 + n2;
}

fn fbm_1d(seed: i32, octaves: u32, gain: f32, lacunarity: f32, freq: f32, x: f32) -> f32 {
    var p = x * freq;
    var result = simplex_1d(p, seed);
    var amplitude = gain;
    for (var i = 1u; i < octaves; i++) {
        p *= lacunarity;
        result += simplex_1d(p, seed) * amplitude;
        amplitude *= gain;
    }
    return result;
}

fn fbm_2d(seed: i32, octaves: u32, gain: f32, lacunarity: f32, freq: f32, point: vec2<f32>) -> f32 {
    var p = point * freq;
    var result = simplex_2d(p, seed);
    var amplitude = gain;
    for (var i = 1u; i < octaves; i++) {
        p *= lacunarity;
        result += simplex_2d(p, seed) * amplitude;
        amplitude *= gain;
    }
    return result;
}
