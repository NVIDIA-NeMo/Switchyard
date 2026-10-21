// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module configures a nonactivating tray panel to join other apps' fullscreen Spaces.

use dispatch2::MainThreadBound;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send, rc::Retained};
use objc2_app_kit::{
    NSApplication, NSBackingStoreType, NSColor, NSEvent, NSEventType, NSPanel,
    NSPopUpMenuWindowLevel, NSWindow, NSWindowCollectionBehavior, NSWindowStyleMask,
};
use objc2_foundation::{MainThreadMarker, NSProcessInfo, NSRect};
use std::cell::Cell;

/// MENU_GAP matches the space macOS leaves between the menu bar and a status item's menu.
const MENU_GAP: f64 = 5.0;

/// CORNER_RADIUS matches the corners of macOS 26 menus; the glass and tray.css use it too.
pub(super) const CORNER_RADIUS: f64 = 12.0;

// Like a status item's menu, the panel's left edge follows the icon and its top edge sits
// just below the menu bar. AppKit supplies points, so the frame needs no scale conversion.
fn popup_frame(anchor: NSRect, area: NSRect, height: f64) -> NSRect {
    let mut frame = area;
    frame.size.width = frame.size.width.min(super::POPUP_WIDTH);
    frame.size.height = frame.size.height.min(height);
    frame.origin.x = anchor.origin.x.clamp(
        area.origin.x,
        area.origin.x + area.size.width - frame.size.width,
    );
    let top = anchor.origin.y.min(area.origin.y + area.size.height) - MENU_GAP;
    frame.origin.y = (top - frame.size.height).clamp(
        area.origin.y,
        area.origin.y + area.size.height - frame.size.height,
    );
    frame
}

/// This function returns the status item's frame and its screen's visible frame in points.
pub(super) fn status_item_frames(tray: &tauri::tray::TrayIcon) -> Option<(NSRect, NSRect)> {
    tray.with_inner_tray_icon(|inner| {
        let mtm = MainThreadMarker::new()?;
        let window = inner.ns_status_item()?.button(mtm)?.window()?;
        Some((window.frame(), window.screen()?.visibleFrame()))
    })
    .ok()
    .flatten()
}

#[derive(Default)]
struct PopupState {
    anchor: Cell<NSRect>,
    area: Cell<NSRect>,
    height: Cell<f64>,
}

define_class!(
    // SAFETY: The panel is initialized as an NSPanel, and every method runs on the main thread.
    #[unsafe(super = NSPanel)]
    #[thread_kind = MainThreadOnly]
    #[ivars = PopupState]
    struct Popup;

    impl Popup {
        #[unsafe(method(canBecomeKeyWindow))]
        fn can_become_key(&self) -> bool { true }

        #[unsafe(method(canBecomeMainWindow))]
        fn can_become_main(&self) -> bool { false }

        #[unsafe(method(resignKeyWindow))]
        fn resign_key(&self) {
            // SAFETY: These inherited AppKit methods have no additional requirements.
            unsafe {
                let _: () = msg_send![super(self), resignKeyWindow];
                let mouse = NSEvent::mouseLocation();
                let anchor = self.ivars().anchor.get();
                // A left-click on the tray icon must reach toggle_tray before focus loss hides the panel.
                let tray_click = NSApplication::sharedApplication(self.mtm()).currentEvent()
                    .is_some_and(|event| matches!(event.r#type(), NSEventType::LeftMouseDown | NSEventType::LeftMouseUp));
                if !tray_click || mouse.x < anchor.origin.x || mouse.x > anchor.origin.x + anchor.size.width
                    || mouse.y < anchor.origin.y || mouse.y > anchor.origin.y + anchor.size.height {
                    self.orderOut(None);
                }
            }
        }
    }
);

// MainThreadBound also releases the panel on the main thread during shutdown.
pub(super) struct TrayPanel(MainThreadBound<Retained<Popup>>);

impl TrayPanel {
    pub(super) fn new(window: &tauri::WebviewWindow) -> Result<Self, String> {
        let mtm = MainThreadMarker::new().ok_or("The tray panel must start on the main thread.")?;
        let pointer = window.ns_window().map_err(|error| error.to_string())?;
        // SAFETY: Tauri supplies a live NSWindow; its hidden window retains the webview and IPC registration.
        unsafe {
            let source = &*pointer.cast::<NSWindow>();
            let view = source
                .contentView()
                .ok_or("The tray webview has no content view.")?;
            let responder = source.firstResponder();
            let allocated = Popup::alloc(mtm).set_ivars(PopupState {
                anchor: Cell::new(NSRect::default()),
                area: Cell::new(source.frame()),
                height: Cell::new(620.0),
            });
            let panel: Retained<Popup> = msg_send![super(allocated),
                initWithContentRect: source.frame(),
                styleMask: NSWindowStyleMask::NonactivatingPanel,
                backing: NSBackingStoreType::Buffered,
                defer: false
            ];
            panel.setReleasedWhenClosed(false);
            panel.setOpaque(false);
            panel.setBackgroundColor(Some(&NSColor::clearColor()));
            panel.setHasShadow(true);
            panel.setTitle(&source.title());
            panel.setHidesOnDeactivate(false);
            panel.setBecomesKeyOnlyIfNeeded(false);
            panel.setLevel(NSPopUpMenuWindowLevel);
            let mut behavior = NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::FullScreenAuxiliary
                | NSWindowCollectionBehavior::Stationary
                | NSWindowCollectionBehavior::IgnoresCycle;
            if NSProcessInfo::processInfo()
                .operatingSystemVersion()
                .majorVersion
                >= 13
            {
                behavior |= NSWindowCollectionBehavior::CanJoinAllApplications;
            }
            panel.setCollectionBehavior(behavior);
            source.setContentView(None);
            panel.setContentView(Some(&view));
            // Clipping the glass and the webview to one rounded shape also rounds the shadow.
            view.setWantsLayer(true);
            if let Some(layer) = view.layer() {
                layer.setCornerRadius(CORNER_RADIUS);
                layer.setMasksToBounds(true);
            }
            panel.makeFirstResponder(responder.as_deref());
            Ok(Self(MainThreadBound::new(panel, mtm)))
        }
    }

    pub(super) fn dismiss(&self) {
        self.0.get_on_main(|panel| panel.orderOut(None));
    }

    pub(super) fn resize(&self, height: f64) {
        self.0.get_on_main(|panel| {
            let state = panel.ivars();
            state.height.set(height);
            panel.setFrame_display(
                popup_frame(state.anchor.get(), state.area.get(), height),
                true,
            );
            panel.invalidateShadow();
        });
    }

    /// This function shows the panel below `anchor`, the status item's frame, within `area`,
    /// the visible frame of the status item's screen. It returns false when it hid the panel.
    pub(super) fn toggle(&self, anchor: NSRect, area: NSRect) -> bool {
        self.0.get_on_main(|panel| {
            // A panel visible on another Space must be shown here rather than hidden by this click.
            if panel.isVisible() && panel.isOnActiveSpace() {
                panel.orderOut(None);
                return false;
            }
            let state = panel.ivars();
            state.anchor.set(anchor);
            state.area.set(area);
            panel.setFrame_display(popup_frame(anchor, area, state.height.get()), true);
            panel.orderFrontRegardless();
            panel.makeKeyWindow();
            panel.invalidateShadow();
            true
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_foundation::{NSPoint, NSSize};

    #[test]
    fn native_popup_follows_the_icon_and_fits_the_work_area() {
        let screen = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(1512.0, 945.0));
        let icon = |x| NSRect::new(NSPoint::new(x, 945.0), NSSize::new(30.0, 37.0));
        assert_eq!(
            popup_frame(icon(1000.0), screen, 400.0),
            NSRect::new(NSPoint::new(1000.0, 540.0), NSSize::new(320.0, 400.0))
        );
        // An icon near the right edge moves the panel left so it stays on screen.
        assert_eq!(popup_frame(icon(1400.0), screen, 400.0).origin.x, 1192.0);
        let area = NSRect::new(NSPoint::new(-1920.0, -1080.0), NSSize::new(1920.0, 1055.0));
        let anchor = NSRect::new(NSPoint::new(-20.0, -25.0), NSSize::new(22.0, 22.0));
        assert_eq!(
            popup_frame(anchor, area, 500.0),
            NSRect::new(NSPoint::new(-320.0, -530.0), NSSize::new(320.0, 500.0))
        );
        let small = NSRect::new(NSPoint::new(-300.0, 50.0), NSSize::new(200.0, 400.0));
        assert_eq!(popup_frame(anchor, small, 620.0), small);
    }
}
