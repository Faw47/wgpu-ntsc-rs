//! Opt-in GPU timestamps. These measure dispatch execution, not host preparation or transfers.
use std::cell::RefCell;

use super::wgpu_backend::WgpuBackendError;

const MAX_PASSES: u32 = 256;

#[derive(Debug, Clone)]
pub struct PassTiming {
    pub stage: &'static str,
    pub milliseconds: f64,
}

pub(crate) struct GpuProfiler {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    stages: RefCell<Vec<&'static str>>,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device) -> Self {
        let size = u64::from(MAX_PASSES) * 2 * 8;
        Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("NTSC pass timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_PASSES * 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NTSC timestamp resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("NTSC timestamp readback"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            stages: RefCell::new(Vec::new()),
        }
    }

    pub fn reset(&self) {
        self.stages.borrow_mut().clear();
    }

    pub fn writes(&self, stage: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let mut stages = self.stages.borrow_mut();
        // The current effect has far fewer passes. Never silently return a partial profile.
        assert!(
            stages.len() < MAX_PASSES as usize,
            "GPU timestamp capacity exceeded"
        );
        let index = stages.len() as u32 * 2;
        stages.push(stage);
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: Some(index + 1),
        })
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.stages.borrow().len() as u32 * 2;
        if count != 0 {
            encoder.resolve_query_set(&self.queries, 0..count, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &self.resolve,
                0,
                &self.readback,
                0,
                u64::from(count) * 8,
            );
        }
    }

    pub fn read(
        &self,
        device: &wgpu::Device,
        period: f32,
    ) -> Result<Vec<PassTiming>, WgpuBackendError> {
        let stages = self.stages.borrow();
        if stages.is_empty() {
            return Ok(Vec::new());
        }
        let slice = self.readback.slice(..(stages.len() * 16) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let result = (|| {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| WgpuBackendError::Readback(e.to_string()))?;
            rx.recv()
                .map_err(|e| WgpuBackendError::Readback(e.to_string()))?
                .map_err(|e| WgpuBackendError::Readback(e.to_string()))?;
            let mapped = slice.get_mapped_range();
            let ticks: &[u64] = bytemuck::cast_slice(&mapped);
            Ok(stages
                .iter()
                .enumerate()
                .map(|(i, &stage)| PassTiming {
                    stage,
                    milliseconds: ticks[2 * i + 1].wrapping_sub(ticks[2 * i]) as f64
                        * f64::from(period)
                        / 1_000_000.0,
                })
                .collect())
        })();
        self.readback.unmap();
        result
    }
}
