// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module configures a nonactivating tray panel to join other apps' fullscreen Spaces.

use dispatch2::MainThreadBound;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send, rc::Retained};
use objc2_app_kit::{
    NSApplication, NSBackingStoreType, NSColor, NSEvent, NSEventType, NSPanel,
    NSPopUpMenuWindowLevel, NSScreen, NSWindow, NSWindowCollectionBehavior, NSWindowStyleMask,
};
use objc2_foundation::{MainThreadMarker, NSPoint, NSProcessInfo, NSRect};
use std::cell::Cell;

// AppKit supplies coordinates in points, so this frame needs no display-scale conversion.
fn popup_frame(anchor: NSRect, area: NSRect, height: f64) -> NSRect {
    let mut frame = area;
    frame.size.width = frame.size.width.min(360.0);
    frame.size.height = frame.size.height.min(height);
    frame.origin.x = (anchor.origin.x + anchor.size.width / 2.0 - frame.size.width / 2.0).clamp(
        area.origin.x,
        area.origin.x + area.size.width - frame.size.width,
    );
    frame.origin.y = (anchor.origin.y - 6.0 - frame.size.height).clamp(
        area.origin.y,
        area.origin.y + area.size.height - frame.size.height,
    );
    frame
}

#[derive(Default)]
struct PopupState {
    anchor: Cell<NSRect>,
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
            panel.makeFirstResponder(responder.as_deref());
            Ok(Self(MainThreadBound::new(panel, mtm)))
        }
    }

    pub(super) fn dismiss(&self) {
        self.0.get_on_main(|panel| panel.orderOut(None));
    }

    pub(super) fn resize(&self, height: f64) {
        self.0.get_on_main(|panel| {
            panel.ivars().height.set(height);
            if let Some(screen) = panel.screen() {
                panel.setFrame_display(
                    popup_frame(panel.ivars().anchor.get(), screen.visibleFrame(), height),
                    true,
                );
            }
        });
    }

    pub(super) fn toggle(&self, rect: tauri::Rect) -> bool {
        self.0.get_on_main(|panel| {
            // A panel visible on another Space must be shown here rather than hidden by this click.
            if panel.isVisible() && panel.isOnActiveSpace() {
                panel.orderOut(None);
                return false;
            }
            let mtm = panel.mtm();
            let mouse = NSEvent::mouseLocation();
            if let Some(screen) = NSScreen::screens(mtm).iter().find(|screen| {
                let frame = screen.frame();
                mouse.x >= frame.origin.x
                    && mouse.x < frame.origin.x + frame.size.width
                    && mouse.y >= frame.origin.y
                    && mouse.y < frame.origin.y + frame.size.height
            }) {
                let scale = screen.backingScaleFactor();
                let origin = rect.position.to_logical::<f64>(scale);
                let size = rect.size.to_logical::<f64>(scale);
                let primary_top = NSScreen::screens(mtm)
                    .iter()
                    .next()
                    .map(|screen| {
                        let frame = screen.frame();
                        frame.origin.y + frame.size.height
                    })
                    .unwrap_or(0.0);
                let anchor = NSRect::new(
                    NSPoint::new(origin.x, primary_top - origin.y - size.height),
                    objc2_foundation::NSSize::new(size.width, size.height),
                );
                panel.ivars().anchor.set(anchor);
                panel.setFrame_display(
                    popup_frame(anchor, screen.visibleFrame(), panel.ivars().height.get()),
                    true,
                );
            }
            panel.orderFrontRegardless();
            panel.makeKeyWindow();
            true
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_foundation::NSSize;

    #[test]
    fn native_popup_fits_negative_origins_and_small_work_areas() {
        let area = NSRect::new(NSPoint::new(-1920.0, -1080.0), NSSize::new(1920.0, 1055.0));
        let anchor = NSRect::new(NSPoint::new(-20.0, -25.0), NSSize::new(22.0, 22.0));
        assert_eq!(
            popup_frame(anchor, area, 500.0),
            NSRect::new(NSPoint::new(-360.0, -531.0), NSSize::new(360.0, 500.0))
        );
        let small = NSRect::new(NSPoint::new(-300.0, 50.0), NSSize::new(200.0, 400.0));
        assert_eq!(popup_frame(anchor, small, 620.0), small);
    }
}
