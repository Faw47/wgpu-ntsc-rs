use std::{cell::RefCell, sync::OnceLock, time::Instant};

use gstreamer::{BufferRef, ClockTime, FlowError};
use gstreamer_video::{VideoFormat, VideoFrameExt, VideoFrameRef, VideoInterlaceMode};
use ntsc_rs::{
    BackendPreference, apply_effect_to_rgba8_with_backend_preference,
    apply_effect_to_yiq_with_backend_preference,
    settings::standard::NtscEffect,
    yiq_fielding::{
        Bgrx, BlitInfo, DeinterlaceMode, Normalize, PixelFormat, Rect, Rgbx, Xbgr, Xrgb, YiqField,
        YiqOwned, YiqView,
    },
};

thread_local! {
    static YIQ_SCRATCH: RefCell<Option<YiqOwned>> = const { RefCell::new(None) };
}

fn host_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NTSC_GPU_PROFILE_HOST").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

fn direct_rgba8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("NTSC_WGPU_DIRECT_RGBA8").as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

fn elapsed_ms(start: Option<Instant>) -> Option<f64> {
    start.map(|start| start.elapsed().as_secs_f64() * 1000.0)
}

fn with_frame_yiq<R>(
    in_frame: &VideoFrameRef<&BufferRef>,
    field: YiqField,
    f: impl FnOnce(&mut YiqOwned) -> Result<R, FlowError>,
) -> Result<R, FlowError> {
    let width = in_frame.width() as usize;
    let height = in_frame.height() as usize;
    let in_stride = in_frame.plane_stride()[0] as usize;
    let in_data = in_frame.plane_data(0).or(Err(FlowError::Error))?;
    let in_format = in_frame.format();

    YIQ_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        let yiq = scratch.get_or_insert_with(|| YiqOwned::new((width, height), field));
        match in_format {
            VideoFormat::Rgbx | VideoFormat::Rgba => {
                yiq.set_from_strided_buffer::<Rgbx, u8>(in_data, in_stride, width, height, field)
            }
            VideoFormat::Bgrx | VideoFormat::Bgra => {
                yiq.set_from_strided_buffer::<Bgrx, u8>(in_data, in_stride, width, height, field)
            }
            VideoFormat::Xrgb | VideoFormat::Argb => {
                yiq.set_from_strided_buffer::<Xrgb, u8>(in_data, in_stride, width, height, field)
            }
            VideoFormat::Xbgr | VideoFormat::Abgr => {
                yiq.set_from_strided_buffer::<Xbgr, u8>(in_data, in_stride, width, height, field)
            }
            VideoFormat::Argb64 => {
                let data_16 = unsafe { in_data.align_to::<u16>() }.1;
                yiq.set_from_strided_buffer::<Xrgb, u16>(data_16, in_stride, width, height, field);
            }
            _ => return Err(FlowError::NotSupported),
        }
        f(yiq)
    })
}

/// Preview-specialized entry point that can use the adapter-side YIQ to RGBA8
/// conversion when the input is progressive, tightly packed, and the effect
/// requests a full-frame `Both` view. All other cases use the general path so
/// fielding, crops, high-bit-depth output, and deinterlacing remain unchanged.
pub fn process_gst_frame_rgbx_u8(
    in_frame: &VideoFrameRef<&BufferRef>,
    out_frame: &mut [u8],
    out_stride: usize,
    settings: &NtscEffect,
    backend_preference: Option<BackendPreference>,
) -> Result<(), FlowError> {
    let backend_preference = backend_preference.unwrap_or_default();
    let width = in_frame.width() as usize;
    let height = in_frame.height() as usize;
    let timestamp = in_frame.buffer().pts().ok_or(FlowError::Error)?.nseconds();
    let frame = (in_frame.info().fps().numer() as u128 * (timestamp + 100) as u128
        / in_frame.info().fps().denom() as u128) as u64
        / ClockTime::SECOND.nseconds();
    let direct = direct_rgba8_enabled()
        && in_frame.info().interlace_mode() == VideoInterlaceMode::Progressive
        && out_stride == width * 4
        && settings.use_field.to_yiq_field(frame as usize) == YiqField::Both
        && matches!(
            backend_preference,
            BackendPreference::Auto | BackendPreference::Wgpu
        );
    if direct {
        let field = YiqField::Both;
        let result = with_frame_yiq(in_frame, field, |yiq| {
            let view = YiqView::from(yiq);
            apply_effect_to_rgba8_with_backend_preference(
                settings,
                &view,
                frame as usize,
                [1.0, 1.0],
                backend_preference,
                out_frame,
            )
            .map_err(|error| {
                eprintln!("ntsc-rs: direct RGBA8 output failed: {error}");
                FlowError::Error
            })
        });
        match result {
            Ok(_) => return Ok(()),
            Err(error) if backend_preference == BackendPreference::Wgpu => return Err(error),
            Err(_) => {
                // Automatic selection is allowed to retry through the general
                // path, which can fall back to the CPU backend.
            }
        }
    }
    process_gst_frame::<Rgbx, u8>(
        in_frame,
        out_frame,
        out_stride,
        None,
        settings,
        Some(backend_preference),
    )
}

pub fn process_gst_frame<S: PixelFormat, T: Normalize>(
    in_frame: &VideoFrameRef<&BufferRef>,
    out_frame: &mut [T],
    out_stride: usize,
    out_rect: Option<Rect>,
    settings: &NtscEffect,
    backend_preference: Option<BackendPreference>,
) -> Result<(), FlowError> {
    let info = in_frame.info();

    let timestamp = in_frame.buffer().pts().ok_or(FlowError::Error)?.nseconds();
    let frame = (info.fps().numer() as u128 * (timestamp + 100) as u128
        / info.fps().denom() as u128) as u64
        / ClockTime::SECOND.nseconds();

    let blit_info = out_rect
        .map(|rect| {
            BlitInfo::new(
                rect,
                (rect.left, rect.top),
                out_stride,
                in_frame.height() as usize,
                false,
            )
        })
        .unwrap_or_else(|| {
            BlitInfo::from_full_frame(
                in_frame.width() as usize,
                in_frame.height() as usize,
                out_stride,
            )
        });
    let backend_preference = backend_preference.unwrap_or_default();

    match in_frame.info().interlace_mode() {
        VideoInterlaceMode::Progressive => {
            let field = settings.use_field.to_yiq_field(frame as usize);
            let profile = host_profile_enabled();
            let total_start = profile.then(Instant::now);
            let input_start = profile.then(Instant::now);
            let (execution, input_ms, backend_ms, output_ms) =
                with_frame_yiq(in_frame, field, |yiq| {
                    let input_ms = elapsed_ms(input_start);
                    let mut view = YiqView::from(yiq);
                    let backend_start = profile.then(Instant::now);
                    let execution = apply_effect_to_yiq_with_backend_preference(
                        settings,
                        &mut view,
                        frame as usize,
                        [1.0, 1.0],
                        backend_preference,
                    )
                    .map_err(|error| {
                        eprintln!("ntsc-rs: {error}");
                        FlowError::Error
                    })?;
                    let backend_ms = elapsed_ms(backend_start);
                    let output_start = profile.then(Instant::now);
                    view.write_to_strided_buffer::<S, T, _>(
                        out_frame,
                        blit_info,
                        DeinterlaceMode::Bob,
                        (),
                    );
                    let output_ms = elapsed_ms(output_start);
                    Ok((execution, input_ms, backend_ms, output_ms))
                })?;
            if let Some(reason) = execution.fallback_reason.as_ref() {
                eprintln!("ntsc-rs: {reason}");
            }
            if profile {
                eprintln!(
                    "ntsc-rs: frame host timings: input_yiq={:.3} ms, backend={:.3} ms, output_rgb={:.3} ms, total={:.3} ms, actual={:?}",
                    input_ms.unwrap_or_default(),
                    backend_ms.unwrap_or_default(),
                    output_ms.unwrap_or_default(),
                    elapsed_ms(total_start).unwrap_or_default(),
                    execution.actual,
                );
            }
        }
        VideoInterlaceMode::Interleaved | VideoInterlaceMode::Mixed => {
            let field = match (in_frame.is_tff(), in_frame.is_onefield()) {
                (true, true) => YiqField::Upper,
                (false, true) => YiqField::Lower,
                (true, false) => YiqField::InterleavedUpper,
                (false, false) => YiqField::InterleavedLower,
            };

            let profile = host_profile_enabled();
            let total_start = profile.then(Instant::now);
            let input_start = profile.then(Instant::now);
            let (execution, input_ms, backend_ms, output_ms) =
                with_frame_yiq(in_frame, field, |yiq| {
                    let input_ms = elapsed_ms(input_start);
                    let mut view = YiqView::from(yiq);
                    let backend_start = profile.then(Instant::now);
                    let execution = apply_effect_to_yiq_with_backend_preference(
                        settings,
                        &mut view,
                        if in_frame.is_onefield() {
                            frame as usize * 2
                        } else {
                            frame as usize
                        },
                        [1.0, 1.0],
                        backend_preference,
                    )
                    .map_err(|error| {
                        eprintln!("ntsc-rs: {error}");
                        FlowError::Error
                    })?;
                    let backend_ms = elapsed_ms(backend_start);
                    let output_start = profile.then(Instant::now);
                    view.write_to_strided_buffer::<S, T, _>(
                        out_frame,
                        blit_info,
                        DeinterlaceMode::Skip,
                        (),
                    );
                    let output_ms = elapsed_ms(output_start);
                    Ok((execution, input_ms, backend_ms, output_ms))
                })?;
            if let Some(reason) = execution.fallback_reason.as_ref() {
                eprintln!("ntsc-rs: {reason}");
            }
            if profile {
                eprintln!(
                    "ntsc-rs: frame host timings: input_yiq={:.3} ms, backend={:.3} ms, output_rgb={:.3} ms, total={:.3} ms, actual={:?}",
                    input_ms.unwrap_or_default(),
                    backend_ms.unwrap_or_default(),
                    output_ms.unwrap_or_default(),
                    elapsed_ms(total_start).unwrap_or_default(),
                    execution.actual,
                );
            }
        }
        _ => Err(FlowError::NotSupported)?,
    }

    Ok(())
}
