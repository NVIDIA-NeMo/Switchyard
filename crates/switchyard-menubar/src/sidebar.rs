// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The route list at the left of the routes window: one entry per route, in
//! two groups.
//!
//! The table view asks [`SidebarSource`] for its rows. A pick goes into a
//! queue that the window reads by calling [`SidebarSource::take_picks`], so a
//! pick that AppKit sends does not reenter the window. Selection changes that
//! the window makes itself, when it redraws the rows, are not picks.

use std::cell::{Cell, RefCell};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::{DefinedClass, MainThreadOnly, define_class, msg_send};
use objc2_app_kit::{
    NSColor, NSControlTextEditingDelegate, NSFont, NSLineBreakMode, NSScrollView, NSTableCellView,
    NSTableColumn, NSTableView, NSTableViewDataSource, NSTableViewDelegate, NSTableViewStyle,
    NSTextField, NSView, NSVisualEffectBlendingMode, NSVisualEffectMaterial, NSVisualEffectState,
    NSVisualEffectView,
};
use objc2_foundation::{
    MainThreadMarker, NSIndexSet, NSInteger, NSNotification, NSObject, NSObjectProtocol, NSString,
    ns_string,
};

use crate::picker::rect;

pub const WIDTH: f64 = 210.0;
const ROW_HEIGHT: f64 = 28.0;

/// One row of the sidebar.
pub enum SidebarRow {
    /// A group title. The user cannot pick it.
    Header(&'static str),
    /// A route: its index in the window's route list, and the text to show.
    Route { route: usize, title: String },
}

/// One route as the sidebar lists it.
pub struct Entry<'a> {
    pub title: &'a str,
    /// Whether the route sends every request to one model.
    pub single_model: bool,
    /// Whether the user changed the route and has not applied the change.
    pub edited: bool,
}

/// Builds the rows: the routes that choose between models, then the routes
/// that send every request to one model. The group titles show only when both
/// groups have routes, because a list of one kind needs no titles. An edited
/// route gets a dot.
pub fn rows(entries: &[Entry]) -> Vec<SidebarRow> {
    let group = |single_model: bool| {
        entries
            .iter()
            .enumerate()
            .filter(move |(_, entry)| entry.single_model == single_model)
            .map(|(route, entry)| SidebarRow::Route {
                route,
                title: if entry.edited {
                    format!("{} •", entry.title)
                } else {
                    entry.title.to_string()
                },
            })
            .collect::<Vec<_>>()
    };
    let (routers, singles) = (group(false), group(true));
    let titled = !routers.is_empty() && !singles.is_empty();
    let mut rows = Vec::with_capacity(entries.len() + 2);
    for (title, group) in [
        ("Multi-model routes", routers),
        ("Single-model routes", singles),
    ] {
        if titled {
            rows.push(SidebarRow::Header(title));
        }
        rows.extend(group);
    }
    rows
}

/// What the table view reads, and what it reports.
#[derive(Default)]
pub struct State {
    rows: RefCell<Vec<SidebarRow>>,
    picks: RefCell<Vec<usize>>,
    /// Whether the window is redrawing the rows. The table view reports the
    /// selection changes that the redraw causes, and they are not picks.
    redrawing: Cell<bool>,
}

define_class!(
    /// Gives the route table its rows and queues the row that the user picks.
    // SAFETY: NSObject has no subclassing requirements, and `SidebarSource`
    // does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "SwitchyardSidebarSource"]
    #[ivars = State]
    pub struct SidebarSource;

    impl SidebarSource {
        // SAFETY: the signature matches the protocol method.
        #[unsafe(method(numberOfRowsInTableView:))]
        fn number_of_rows(&self, _table: &NSTableView) -> NSInteger {
            NSInteger::try_from(self.ivars().rows.borrow().len()).unwrap_or(0)
        }

        // SAFETY: the signature matches the protocol method.
        #[unsafe(method_id(tableView:viewForTableColumn:row:))]
        fn view_for_row(
            &self,
            _table: &NSTableView,
            _column: Option<&NSTableColumn>,
            row: NSInteger,
        ) -> Option<Retained<NSView>> {
            self.view_at(row)
        }

        // SAFETY: the signature matches the protocol method.
        #[unsafe(method(tableView:isGroupRow:))]
        fn is_group_row(&self, _table: &NSTableView, row: NSInteger) -> bool {
            self.is_header(row)
        }

        // SAFETY: the signature matches the protocol method.
        #[unsafe(method(tableView:shouldSelectRow:))]
        fn should_select_row(&self, _table: &NSTableView, row: NSInteger) -> bool {
            !self.is_header(row)
        }

        // SAFETY: the signature matches the protocol method.
        #[unsafe(method(tableViewSelectionDidChange:))]
        fn selection_did_change(&self, notification: &NSNotification) {
            if self.ivars().redrawing.get() {
                return;
            }
            let table = notification
                .object()
                .and_then(|object| object.downcast::<NSTableView>().ok());
            if let Some(row) = table.and_then(|table| usize::try_from(table.selectedRow()).ok()) {
                self.ivars().picks.borrow_mut().push(row);
            }
        }
    }

    unsafe impl NSObjectProtocol for SidebarSource {}
    unsafe impl NSControlTextEditingDelegate for SidebarSource {}
    unsafe impl NSTableViewDataSource for SidebarSource {}
    unsafe impl NSTableViewDelegate for SidebarSource {}
);

impl SidebarSource {
    pub fn new(mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(State::default());
        // SAFETY: NSObject's `init` has this signature.
        unsafe { msg_send![super(this), init] }
    }

    /// Shows `rows` in `table` and selects the row of `route`. The table
    /// view clears the selection when it reloads, and it selects the first
    /// route because it does not allow an empty selection. The window must
    /// not read either change as a pick.
    pub fn show(&self, table: &NSTableView, rows: Vec<SidebarRow>, route: usize) {
        *self.ivars().rows.borrow_mut() = rows;
        self.ivars().redrawing.set(true);
        table.reloadData();
        if let Some(row) = self.row_of(route) {
            table.selectRowIndexes_byExtendingSelection(&NSIndexSet::indexSetWithIndex(row), false);
            table.scrollRowToVisible(isize::try_from(row).unwrap_or(0));
        }
        self.ivars().redrawing.set(false);
    }

    /// Returns the rows that the user picked since the last call.
    pub fn take_picks(&self) -> Vec<usize> {
        std::mem::take(&mut *self.ivars().picks.borrow_mut())
    }

    /// Builds the view for a row, or `None` when the table asks for a row
    /// that does not exist.
    fn view_at(&self, row: NSInteger) -> Option<Retained<NSView>> {
        let rows = self.ivars().rows.borrow();
        let row = rows.get(usize::try_from(row).ok()?)?;
        Some(row_view(self.mtm(), row))
    }
    /// Returns the route that a row shows, or `None` for a group title.
    pub fn route_at(&self, row: usize) -> Option<usize> {
        match self.ivars().rows.borrow().get(row)? {
            SidebarRow::Route { route, .. } => Some(*route),
            SidebarRow::Header(_) => None,
        }
    }

    /// Returns the row that shows a route.
    pub fn row_of(&self, route: usize) -> Option<usize> {
        self.ivars().rows.borrow().iter().position(
            |row| matches!(row, SidebarRow::Route { route: shown, .. } if *shown == route),
        )
    }

    fn is_header(&self, row: NSInteger) -> bool {
        usize::try_from(row).is_ok_and(|row| {
            matches!(
                self.ivars().rows.borrow().get(row),
                Some(SidebarRow::Header(_))
            )
        })
    }
}

fn row_view(mtm: MainThreadMarker, row: &SidebarRow) -> Retained<NSView> {
    let (text, header) = match row {
        SidebarRow::Header(title) => ((*title).to_string(), true),
        SidebarRow::Route { title, .. } => (title.clone(), false),
    };
    let label = NSTextField::labelWithString(&NSString::from_str(&text), mtm);
    if header {
        label.setFont(Some(&NSFont::boldSystemFontOfSize(
            NSFont::smallSystemFontSize(),
        )));
        label.setTextColor(Some(&NSColor::secondaryLabelColor()));
    }
    label.setLineBreakMode(NSLineBreakMode::ByTruncatingTail);
    label.setFrame(rect(10.0, 5.0, WIDTH - 36.0, 18.0));
    // SAFETY: NSView's `initWithFrame:` has this signature.
    let cell: Retained<NSTableCellView> = unsafe {
        msg_send![NSTableCellView::alloc(mtm), initWithFrame: rect(0.0, 0.0, WIDTH, ROW_HEIGHT)]
    };
    cell.addSubview(&label);
    // The cell view recolors its text field when the row is selected.
    // SAFETY: the label is a text field, which `textField` takes.
    unsafe { cell.setTextField(Some(&label)) };
    Retained::into_super(cell)
}

/// Builds the sidebar: a translucent panel that holds the route table.
/// `source` must live as long as the returned views.
pub fn build(
    mtm: MainThreadMarker,
    source: &SidebarSource,
    height: f64,
) -> (Retained<NSVisualEffectView>, Retained<NSTableView>) {
    let frame = rect(0.0, 0.0, WIDTH, height);
    let panel = NSVisualEffectView::initWithFrame(NSVisualEffectView::alloc(mtm), frame);
    panel.setMaterial(NSVisualEffectMaterial::Sidebar);
    panel.setBlendingMode(NSVisualEffectBlendingMode::BehindWindow);
    panel.setState(NSVisualEffectState::FollowsWindowActiveState);

    let table = NSTableView::initWithFrame(NSTableView::alloc(mtm), frame);
    let column = NSTableColumn::initWithIdentifier(NSTableColumn::alloc(mtm), ns_string!("route"));
    column.setWidth(WIDTH - 20.0);
    table.addTableColumn(&column);
    table.setHeaderView(None);
    table.setStyle(NSTableViewStyle::SourceList);
    table.setBackgroundColor(&NSColor::clearColor());
    table.setRowHeight(ROW_HEIGHT);
    table.setAllowsEmptySelection(false);
    // SAFETY: the caller keeps `source` alive as long as the table.
    unsafe {
        table.setDataSource(Some(ProtocolObject::from_ref(source)));
        table.setDelegate(Some(ProtocolObject::from_ref(source)));
    }

    // The list starts a little below the title bar.
    let scroll = NSScrollView::initWithFrame(
        NSScrollView::alloc(mtm),
        rect(0.0, 0.0, WIDTH, height - 10.0),
    );
    scroll.setDrawsBackground(false);
    scroll.setHasVerticalScroller(true);
    scroll.setAutohidesScrollers(true);
    scroll.setDocumentView(Some(&table));
    panel.addSubview(&scroll);
    (panel, table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(title: &str, single_model: bool, edited: bool) -> Entry<'_> {
        Entry {
            title,
            single_model,
            edited,
        }
    }

    /// Describes the rows as text: `#` for a group title, `-` for a route.
    fn describe(rows: &[SidebarRow]) -> Vec<String> {
        rows.iter()
            .map(|row| match row {
                SidebarRow::Header(title) => format!("# {title}"),
                SidebarRow::Route { route, title } => format!("- {route} {title}"),
            })
            .collect()
    }

    #[test]
    fn lists_the_multi_model_routes_first_and_keeps_each_route_index() {
        let rows = rows(&[
            entry("gpt-6-sol", true, false),
            entry("switchyard", false, true),
            entry("gpt-6-luna", true, false),
        ]);

        assert_eq!(
            describe(&rows),
            [
                "# Multi-model routes",
                "- 1 switchyard •",
                "# Single-model routes",
                "- 0 gpt-6-sol",
                "- 2 gpt-6-luna",
            ]
        );
    }

    #[test]
    fn a_list_of_one_kind_has_no_group_titles() {
        let singles = rows(&[entry("a", true, false), entry("b", true, false)]);
        let routers = rows(&[entry("a", false, false)]);

        assert_eq!(describe(&singles), ["- 0 a", "- 1 b"]);
        assert_eq!(describe(&routers), ["- 0 a"]);
    }
}
