use super::k_quants::{BlockQ2K, BlockQ4_0, BlockQ4K, BlockQ6K, BlockQ8_0, BlockQ8K, QK_K, QK8_0};

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    // Fallback to unoptimized version for now
    let nb = n / QK8_0;
    let mut sumf = 0f32;
    for i in 0..nb {
        let x0 = &xs[i];
        let y0 = &ys[i];

        let mut sum_i = 0i32;
        for j in 0..QK8_0 / 2 {
            let v0 = (x0.qs[j] & 0x0F) as i32 - 8;
            let v1 = (x0.qs[j] >> 4) as i32 - 8;
            sum_i += v0 * y0.qs[j] as i32 + v1 * y0.qs[j + QK8_0 / 2] as i32;
        }
        sumf += sum_i as f32 * x0.d.to_f32() * y0.d.to_f32();
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    let nb = n / QK8_0;
    let mut sumf = 0f32;
    for i in 0..nb {
        let x0 = &xs[i];
        let y0 = &ys[i];

        let sum_i = x0
            .qs
            .iter()
            .zip(y0.qs.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum::<i32>();
        sumf += sum_i as f32 * x0.d.to_f32() * y0.d.to_f32();
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q2k_q8k(n: usize, xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    // Fallback to unoptimized
    let mut sumf = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let mut q2: &[_] = &x.qs;
        let mut q8: &[_] = &y.qs;
        let sc = &x.scales;

        let mut summs = 0;
        for (bsum, scale) in y.bsums.iter().zip(sc) {
            summs += *bsum as i32 * ((scale >> 4) as i32);
        }

        let dall = y.d * x.d.to_f32();
        let dmin = y.d * x.dmin.to_f32();

        let mut isum = 0;
        let mut is = 0;
        for _ in 0..(QK_K / 128) {
            let mut shift = 0;
            for _ in 0..4 {
                let d = (sc[is] & 0xF) as i32;
                is += 1;
                let mut isuml = 0;
                for l in 0..16 {
                    isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                }
                isum += d * isuml;
                let d = (sc[is] & 0xF) as i32;
                is += 1;
                isuml = 0;
                for l in 16..32 {
                    isuml += q8[l] as i32 * (((q2[l] >> shift) & 3) as i32);
                }
                isum += d * isuml;
                shift += 2;
                q8 = &q8[32..];
            }
            q2 = &q2[32..];
        }
        sumf += dall * isum as f32 - dmin * summs as f32;
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    // Fallback
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    let mut utmp: [u32; 4] = [0; 4];
    let mut scales: [u8; 8] = [0; 8];
    let mut mins: [u8; 8] = [0; 8];

    let mut sumf = 0.0;
    for (y, x) in ys.iter().zip(xs.iter()) {
        let q4 = &x.qs;
        let q8 = &y.qs;
        let mut aux8: [i8; QK_K] = [0; QK_K];
        let mut aux16: [i16; 8] = [0; 8];
        let mut aux32: [i32; 8] = [0; 8];

        let mut a = &mut aux8[..];
        let mut q4 = &q4[..];
        for _ in 0..QK_K / 64 {
            for l in 0..32 {
                a[l] = (q4[l] & 0xF) as i8;
            }
            a = &mut a[32..];
            for l in 0..32 {
                a[l] = (q4[l] >> 4) as i8;
            }
            a = &mut a[32..];
            q4 = &q4[32..];
        }

        super::k_quants::little_endian_read_u32_into(&x.scales, &mut utmp[0..3]);

        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;

        super::k_quants::little_endian_write_u32_into(&utmp[0..2], &mut scales);
        super::k_quants::little_endian_write_u32_into(&utmp[2..4], &mut mins);

        let mut sumi = 0;
        for j in 0..QK_K / 16 {
            sumi += y.bsums[j] as i32 * mins[j / 2] as i32;
        }

        let mut a = &mut aux8[..];
        let mut q8 = &q8[..];

        for scale in scales {
            let scale = scale as i32;
            for _ in 0..4 {
                for l in 0..8 {
                    aux16[l] = q8[l] as i16 * a[l] as i16;
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8 = &q8[8..];
                a = &mut a[8..];
            }
        }
        let d = x.d.to_f32() * y.d;
        for l in 0..8 {
            sumf += d * aux32[l] as f32;
        }
        let dmin = x.dmin.to_f32() * y.d;
        sumf -= dmin * sumi as f32;
    }
    sumf
}

#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    // Fallback
    let mut aux8 = [0i8; QK_K];
    let mut aux16 = [0i16; 8];
    let mut sums = [0f32; 8];
    let mut aux32 = [0f32; 8];

    for (x, y) in xs.iter().zip(ys.iter()) {
        let q4 = &x.ql;
        let qh = &x.qh;
        let q8 = &y.qs;
        aux32.fill(0f32);

        for j in (0..QK_K).step_by(128) {
            let aux8 = &mut aux8[j..];
            let q4 = &q4[j / 2..];
            let qh = &qh[j / 4..];
            for l in 0..32 {
                aux8[l] = (((q4[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                aux8[l + 32] = (((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                aux8[l + 64] = (((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                aux8[l + 96] = (((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
            }
        }

        for (j, &scale) in x.scales.iter().enumerate() {
            let scale = scale as f32;
            let q8 = &q8[16 * j..];
            let aux8 = &aux8[16 * j..];
            for l in 0..8 {
                aux16[l] = q8[l] as i16 * aux8[l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as f32;
            }
            let q8 = &q8[8..];
            let aux8 = &aux8[8..];
            for l in 0..8 {
                aux16[l] = q8[l] as i16 * aux8[l] as i16;
            }
            for l in 0..8 {
                aux32[l] += scale * aux16[l] as f32;
            }
        }

        let d = x.d.to_f32() * y.d;
        for (sum, &a) in sums.iter_mut().zip(aux32.iter()) {
            *sum += a * d;
        }
    }
    sums.iter().sum()
}

#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k(n: usize, xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q8k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut sumf = 0f32;
    for (xs, ys) in xs.iter().zip(ys.iter()) {
        let sum_i = xs
            .qs
            .iter()
            .zip(ys.qs.iter())
            .map(|(&x, &y)| x as i32 * y as i32)
            .sum::<i32>();
        sumf += sum_i as f32 * xs.d * ys.d;
    }
    sumf
}
