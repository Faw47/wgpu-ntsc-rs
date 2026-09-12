//! Visual system and navigation state for the standalone editing workspace.
//!
//! The effect schema remains the source of truth for settings. This module only
//! owns presentation state, search filtering, and the small amount of chrome that
//! makes the existing operations discoverable.

use eframe::egui::{self, Color32, FontId, Stroke, vec2};
use ntsc_rs::settings::{SettingDescriptor, SettingKind, Settings};

#[derive(Debug, Default)]
pub struct WorkspaceState {
    pub search: String,
    pub section: EffectSection,
    pub focus_preview: bool,
    pub focus_search: bool,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum EffectSection {
    #[default]
    All,
    Signal,
    Chroma,
    Tape,
}

impl EffectSection {
    pub const VALUES: [Self; 4] = [Self::All, Self::Signal, Self::Chroma, Self::Tape];

    pub fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Signal => "Signal",
            Self::Chroma => "Chroma",
            Self::Tape => "Tape & noise",
        }
    }

    pub fn includes(self, label: &str) -> bool {
        let lower = label.to_ascii_lowercase();
        let tape = [
            "vhs",
            "head switching",
            "tracking",
            "snow",
            "noise",
            "ringing",
        ]
        .iter()
        .any(|term| lower.contains(term));
        let chroma = lower.contains("chroma") && !tape;
        match self {
            Self::All => true,
            Self::Signal => !tape && !chroma,
            Self::Chroma => chroma,
            Self::Tape => tape,
        }
    }
}

/// Clone only the descriptors that can produce visible results for a query.
/// `SettingDescriptor` is intentionally schema-owned, so keeping this helper here
/// prevents the UI from maintaining a second list of effect controls.
pub fn filter_descriptors<T: Settings + Clone>(
    descriptors: &[SettingDescriptor<T>],
    section: EffectSection,
    query: &str,
) -> Vec<SettingDescriptor<T>> {
    fn filter_one<T: Settings + Clone>(
        descriptor: &SettingDescriptor<T>,
        query: &str,
    ) -> Option<SettingDescriptor<T>> {
        let direct = query.is_empty()
            || descriptor.label.to_ascii_lowercase().contains(query)
            || descriptor
                .description
                .is_some_and(|description| description.to_ascii_lowercase().contains(query));
        if direct {
            return Some(descriptor.clone());
        }

        if let SettingKind::Group { children } = &descriptor.kind {
            let children = children
                .iter()
                .filter_map(|child| filter_one(child, query))
                .collect::<Vec<_>>();
            if !children.is_empty() {
                let mut group = descriptor.clone();
                group.kind = SettingKind::Group { children };
                return Some(group);
            }
        }
        None
    }

    let query = query.trim().to_ascii_lowercase();
    descriptors
        .iter()
        .filter(|descriptor| section.includes(descriptor.label))
        .filter_map(|descriptor| filter_one(descriptor, &query))
        .collect()
}

pub fn configure_style(ctx: &egui::Context) {
    for theme in [egui::Theme::Dark, egui::Theme::Light] {
        ctx.style_mut_of(theme, |style| {
            let dark = theme == egui::Theme::Dark;
            style
                .text_styles
                .insert(egui::TextStyle::Body, FontId::proportional(13.0));
            style
                .text_styles
                .insert(egui::TextStyle::Button, FontId::proportional(13.0));
            style
                .text_styles
                .insert(egui::TextStyle::Small, FontId::proportional(11.0));
            style
                .text_styles
                .insert(egui::TextStyle::Heading, FontId::proportional(18.0));
            style
                .text_styles
                .insert(egui::TextStyle::Monospace, FontId::monospace(12.0));
            style.spacing.item_spacing = vec2(8.0, 8.0);
            style.spacing.button_padding = vec2(10.0, 6.0);
            style.spacing.interact_size = vec2(36.0, 28.0);
            style.spacing.window_margin = egui::Margin::same(12);
            style.animation_time = 0.08;

            let (canvas, panel, control, text, _muted, line, accent) = if dark {
                (
                    Color32::from_rgb(21, 23, 25),
                    Color32::from_rgb(32, 35, 38),
                    Color32::from_rgb(44, 48, 52),
                    Color32::from_rgb(232, 234, 236),
                    Color32::from_rgb(169, 176, 183),
                    Color32::from_rgb(64, 70, 76),
                    Color32::from_rgb(141, 188, 179),
                )
            } else {
                (
                    Color32::from_rgb(227, 230, 232),
                    Color32::from_rgb(244, 245, 246),
                    Color32::WHITE,
                    Color32::from_rgb(28, 34, 38),
                    Color32::from_rgb(80, 89, 96),
                    Color32::from_rgb(184, 193, 199),
                    Color32::from_rgb(34, 99, 89),
                )
            };
            style.visuals.panel_fill = panel;
            style.visuals.window_fill = panel;
            style.visuals.extreme_bg_color = canvas;
            style.visuals.faint_bg_color = control;
            style.visuals.override_text_color = Some(text);
            style.visuals.hyperlink_color = accent;
            style.visuals.selection.bg_fill = if dark {
                Color32::from_rgb(50, 78, 73)
            } else {
                Color32::from_rgb(201, 225, 219)
            };
            style.visuals.selection.stroke = Stroke::new(1.0, accent);
            style.visuals.window_stroke = Stroke::new(1.0, line);
            style.visuals.window_corner_radius = egui::CornerRadius::same(6);
            style.visuals.widgets.inactive.bg_fill = control;
            style.visuals.widgets.inactive.weak_bg_fill = control;
            style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, line);
            style.visuals.widgets.inactive.fg_stroke = Stroke::new(1.0, text);
            style.visuals.widgets.hovered.bg_fill = if dark {
                Color32::from_rgb(59, 65, 70)
            } else {
                Color32::from_rgb(220, 228, 230)
            };
            style.visuals.widgets.hovered.weak_bg_fill = style.visuals.widgets.hovered.bg_fill;
            style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, accent);
            style.visuals.widgets.active.bg_stroke = Stroke::new(1.5, accent);
            for widget in [
                &mut style.visuals.widgets.inactive,
                &mut style.visuals.widgets.hovered,
                &mut style.visuals.widgets.active,
                &mut style.visuals.widgets.noninteractive,
            ] {
                widget.corner_radius = egui::CornerRadius::same(4);
            }
        });
    }
}

pub fn brand(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(vec2(18.0, 18.0), egui::Sense::hover());
    let colors = [
        Color32::from_rgb(183, 194, 184),
        Color32::from_rgb(126, 163, 161),
        Color32::from_rgb(168, 153, 176),
    ];
    for (index, color) in colors.into_iter().enumerate() {
        let x = rect.min.x + index as f32 * 6.0;
        ui.painter().rect_filled(
            egui::Rect::from_min_size(egui::pos2(x, rect.min.y), vec2(5.0, 18.0)),
            1.0,
            color,
        );
    }
    ui.label(egui::RichText::new("ntsc-rs").strong().size(16.0));
}

#[cfg(test)]
mod tests {
    use super::*;
    use ntsc_rs::settings::{SettingsList, standard::NtscEffectFullSettings};

    #[test]
    fn filtering_preserves_groups_and_supports_empty_and_nested_queries() {
        let list = SettingsList::<NtscEffectFullSettings>::new();
        let all = filter_descriptors(&list.setting_descriptors, EffectSection::All, "");
        assert_eq!(all.len(), list.setting_descriptors.len());
        assert!(filter_descriptors(&all, EffectSection::All, "not-real").is_empty());
        assert!(!filter_descriptors(&all, EffectSection::All, "tracking").is_empty());
        assert!(
            filter_descriptors(&all, EffectSection::All, "intensity")
                .iter()
                .any(|descriptor| matches!(descriptor.kind, SettingKind::Group { .. }))
        );
    }
}
