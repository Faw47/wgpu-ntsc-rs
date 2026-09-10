#![cfg(feature = "gpu-wgpu")]

fn validate(label: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{label} WGSL parse failed: {error}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|error| panic!("{label} WGSL validation failed: {error}"));
}

#[test]
fn every_live_wgsl_module_parses_and_validates_without_an_adapter() {
    for (label, source) in [
        (
            "chroma delay",
            include_str!("../src/gpu/shaders/chroma_delay.wgsl").to_owned(),
        ),
        (
            "chroma into luma",
            include_str!("../src/gpu/shaders/chroma_into_luma.wgsl").to_owned(),
        ),
        (
            "chroma loss and vertical blend",
            include_str!("../src/gpu/shaders/chroma_loss_blend.wgsl").to_owned(),
        ),
        (
            "chroma phase",
            include_str!("../src/gpu/shaders/chroma_phase.wgsl").to_owned(),
        ),
        (
            "filter plane",
            include_str!("../src/gpu/shaders/filter_plane.wgsl").to_owned(),
        ),
        (
            "luma box",
            include_str!("../src/gpu/shaders/luma_box.wgsl").to_owned(),
        ),
        (
            "luma into chroma",
            include_str!("../src/gpu/shaders/luma_into_chroma.wgsl").to_owned(),
        ),
        (
            "row effects",
            format!(
                "{}\n{}",
                include_str!("../src/gpu/shaders/simplex.wgsl"),
                include_str!("../src/gpu/shaders/row_effects.wgsl")
            ),
        ),
    ] {
        validate(label, &source);
    }
}
