use gstreamer::{BufferRef, ClockTime, FlowError};
use gstreamer_video::{VideoFormat, VideoFrameExt, VideoFrameRef, VideoInterlaceMode};
use ntsc_rs::{
    BackendPreference, apply_effect_to_yiq_with_backend_preference,
    settings::standard::NtscEffect,
    yiq_fielding::{
        Bgrx, BlitInfo, DeinterlaceMode, Normalize, PixelFormat, Rect, Rgbx, Xbgr, Xrgb, YiqField,
        YiqView,
    },
};
use std::cell::RefCell;

thread_local! {
    // GStreamer normally keeps a stream on one worker. Retain its conversion buffer just like the
    // WGPU runner retains device-side frame buffers, while preserving the old zeroed scratch plane.
    static YIQ_BUFFER: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn frame_to_yiq(
    in_frame: &VideoFrameRef<&BufferRef>,
    view: &mut YiqView,
) -> Result<(), FlowError> {
    let width = in_frame.width() as usize;
    let height = in_frame.height() as usize;
    let in_stride = in_frame.plane_stride()[0] as usize;
    let in_data = in_frame.plane_data(0).or(Err(FlowError::Error))?;
    let in_format = in_frame.format();
    match in_format {
        VideoFormat::Rgbx | VideoFormat::Rgba => {
            view.set_from_strided_buffer::<Rgbx, u8, _>(
                in_data,
                BlitInfo::from_full_frame(width, height, in_stride),
                (),
            );
        }
        VideoFormat::Bgrx | VideoFormat::Bgra => {
            view.set_from_strided_buffer::<Bgrx, u8, _>(
                in_data,
                BlitInfo::from_full_frame(width, height, in_stride),
                (),
            );
        }
        VideoFormat::Xrgb | VideoFormat::Argb => {
            view.set_from_strided_buffer::<Xrgb, u8, _>(
                in_data,
                BlitInfo::from_full_frame(width, height, in_stride),
                (),
            );
        }
        VideoFormat::Xbgr | VideoFormat::Abgr => {
            view.set_from_strided_buffer::<Xbgr, u8, _>(
                in_data,
                BlitInfo::from_full_frame(width, height, in_stride),
                (),
            );
        }

        VideoFormat::Argb64 => {
            let data_16 = unsafe { in_data.align_to::<u16>() }.1;
            view.set_from_strided_buffer::<Xrgb, u16, _>(
                data_16,
                BlitInfo::from_full_frame(width, height, in_stride),
                (),
            );
        }
        _ => return Err(FlowError::NotSupported),
    }
    Ok(())
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

    let (field, effect_frame, deinterlace_mode) = match in_frame.info().interlace_mode() {
        VideoInterlaceMode::Progressive => {
            let field = settings.use_field.to_yiq_field(frame as usize);
            (field, frame as usize, DeinterlaceMode::Bob)
        }
        VideoInterlaceMode::Interleaved | VideoInterlaceMode::Mixed => {
            let field = match (in_frame.is_tff(), in_frame.is_onefield()) {
                (true, true) => YiqField::Upper,
                (false, true) => YiqField::Lower,
                (true, false) => YiqField::InterleavedUpper,
                (false, false) => YiqField::InterleavedLower,
            };
            let effect_frame = if in_frame.is_onefield() {
                frame as usize * 2
            } else {
                frame as usize
            };
            (field, effect_frame, DeinterlaceMode::Skip)
        }
        _ => return Err(FlowError::NotSupported),
    };

    YIQ_BUFFER.with(|storage| {
        let mut storage = storage.borrow_mut();
        let required = YiqView::buf_length_for((info.width() as usize, info.height() as usize), field);
        storage.resize(required, 0.0);
        storage.fill(0.0);
        let mut view = YiqView::from_parts(
            storage.as_mut_slice(),
            (info.width() as usize, info.height() as usize),
            field,
        );
        frame_to_yiq(in_frame, &mut view)?;
        apply_effect_to_yiq_with_backend_preference(
            settings,
            &mut view,
            effect_frame,
            [1.0, 1.0],
            backend_preference,
        );
        view.write_to_strided_buffer::<S, T, _>(
            out_frame,
            blit_info,
            deinterlace_mode,
            (),
        );
        Ok(())
    })
}
