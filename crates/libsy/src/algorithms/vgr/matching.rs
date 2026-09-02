// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, anchored agreement between a witness answer and an attempt's final
//! message.
//!
//! Answering "do these two say the same thing?" mechanically is where a
//! verification-gated router is easiest to fool, so this module is deliberately
//! strict. Everything it accepts, a careful reader would call agreement; it
//! rejects a great deal that a looser matcher would accept.
//!
//! Three properties do most of the work:
//!
//! - **Quantities are typed.** A unit is part of the value, so `$5`, `€5`, `5`
//!   and `5%` are four different answers. Numbers compare by value, never by
//!   prefix or substring.
//! - **Matching is anchored.** A candidate must appear in the span the message
//!   actually concludes with — an explicit answer marker, else the last
//!   non-empty line. A number mentioned and then discarded earlier in the
//!   reasoning is not the answer.
//! - **Ambiguity is not agreement.** Same-line corrections resolve to the last
//!   segment, and a span still carrying negation or contradiction language
//!   cannot be resolved mechanically, so it yields [`Tri::Unknown`] rather than
//!   a guess. Indeterminate never commits.

use regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

use super::rules::Tri;

/// Words too common or too syntactic to identify an answer on their own.
///
/// A candidate made only of these is a code fragment or filler that would match
/// almost any message, not a claim about the task.
static STOPWORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are", "it", "yes", "no",
        "none", "n/a", "na", "ok", "okay", "done", "true", "false", "if", "else", "for", "while",
        "return", "report", "retry", "answer", "final", "result", "block", "task", "not",
    ]
    .into_iter()
    .collect()
});

/// Words that turn a nearby candidate into a rejected one rather than an answer.
static NEGATIONS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "not", "no", "never", "isnt", "wasnt", "without", "excluding", "except", "rather",
        "instead", "neither", "nor", "wrong", "incorrect",
    ]
    .into_iter()
    .collect()
});

/// How many tokens before a candidate are checked for negation.
const NEGATION_WINDOW: usize = 3;

/// A terse answer that is entirely a quantity, optionally signed or unit-bearing.
static NUMBERISH: LazyLock<Regex> =
    LazyLock::new(|| regex(r"^[\s$€£+-]*[\d][\d,\s]*(?:\.\d+)?\s*%?$"));

/// A normalized token: a run of lowercase alphanumerics, keeping a decimal tail.
static TOKEN: LazyLock<Regex> = LazyLock::new(|| regex(r"[a-z0-9]+(?:\.[0-9]+)?"));

/// A currency-prefixed quantity embedded in prose.
static CURRENCY_QUANTITY: LazyLock<Regex> =
    LazyLock::new(|| regex(r"([$€£])\s*(\d[\d,]*(?:\.\d+)?)"));

/// A percent-suffixed quantity embedded in prose.
static PERCENT_QUANTITY: LazyLock<Regex> = LazyLock::new(|| regex(r"(\d[\d,]*(?:\.\d+)?)\s*%"));

/// An explicit statement of the answer, capturing what follows the marker.
static ANSWER_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    regex(
        r"(?i)(?:final\s+answer|answer|conclusion|result|correction|revised\s+answer|actually)\s*(?:is|:|=|-)\s*(.+)",
    )
});

/// A marker after which the operative answer restarts, used to split a span.
static CORRECTION_SPLIT: LazyLock<Regex> = LazyLock::new(|| {
    regex(r"(?i)\b(?:actually|instead|correction|revised(?:\s+answer)?|rather|i\s+meant)\b[,:\s]*")
});

/// Compiles a pattern that is a module constant.
///
/// The patterns here are fixed literals that are known to compile; a failure
/// would be a bug in this file rather than a runtime condition, so it degrades
/// to a regex that never matches instead of panicking.
fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).unwrap_or_else(|_| Regex::new(r"$.^").unwrap_or_else(|_| unreachable!()))
}

/// The unit a quantity carries. Part of the value, never discarded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Unit {
    /// A bare number.
    None,
    /// A currency marker, kept as the character itself so currencies differ.
    Currency(char),
    /// A percentage.
    Percent,
}

/// A parsed quantity: a magnitude and the unit that types it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Quantity {
    value: f64,
    unit: Unit,
}

impl Quantity {
    /// Whether two quantities are the same answer.
    ///
    /// Units must be identical, so `$5` never equals `5` or `€5`. Magnitudes
    /// compare with a relative tolerance rather than exactly, so `5.0` equals
    /// `5` without `100` ever equalling `1`.
    fn agrees_with(self, other: Quantity) -> bool {
        self.unit == other.unit
            && (self.value - other.value).abs()
                <= 1e-9 * self.value.abs().max(other.value.abs()).max(1.0)
    }
}

/// Parses a terse answer that is entirely a quantity.
///
/// Returns `None` for anything with surrounding prose. A string carrying two
/// different currency markers, or both a currency and a percent, is not a
/// coherent quantity and is rejected.
fn parse_quantity(s: &str) -> Option<Quantity> {
    let t = s.trim();
    if t.is_empty() || !NUMBERISH.is_match(t) {
        return None;
    }
    // Which currency symbols appear at all, not how many times: a repeated
    // marker is one currency, whereas two different markers are incoherent.
    let mut present = ['$', '€', '£'].into_iter().filter(|c| t.contains(*c));
    let currency = present.next();
    if present.next().is_some() {
        return None;
    }
    let unit = match (currency, t.contains('%')) {
        (Some(_), true) => return None,
        (Some(c), false) => Unit::Currency(c),
        (None, true) => Unit::Percent,
        (None, false) => Unit::None,
    };
    let digits: String = t
        .chars()
        .filter(|c| !matches!(c, ',' | '$' | '€' | '£' | '%' | '+') && !c.is_whitespace())
        .collect();
    digits.parse::<f64>().ok().map(|value| Quantity { value, unit })
}

/// The magnitude of a terse numeric answer, ignoring its unit.
///
/// Unit-blind, so it is used for deciding whether something *is* a number, never
/// for deciding whether two numbers agree.
fn parse_number(s: &str) -> Option<f64> {
    parse_quantity(s).map(|q| q.value)
}

/// Fuses unit markers onto their numbers before tokenization.
///
/// Without this, `$5 million` and `€5 million` tokenize identically and would
/// agree. Fusing makes the unit part of the token, so unit-bearing and bare
/// quantities can never match each other.
fn fuse_units(s: &str) -> String {
    let currency_fused = CURRENCY_QUANTITY.replace_all(s, |caps: &regex::Captures| {
        let marker = caps[1].chars().next().map_or(0, u32::from);
        format!(" cur{}u{} ", marker, caps[2].replace(',', ""))
    });
    PERCENT_QUANTITY
        .replace_all(&currency_fused, |caps: &regex::Captures| {
            format!(" pct{} ", caps[1].replace(',', ""))
        })
        .into_owned()
}

/// Normalizes text to the token sequence that matching compares.
fn norm_tokens(s: &str) -> Vec<String> {
    let fused = fuse_units(s).to_lowercase();
    TOKEN
        .find_iter(&fused)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Whether a candidate could plausibly identify an answer at all.
///
/// Guards against fragments that would match almost any message — a stopword, a
/// scrap of punctuation-heavy code, a two-character sliver. Standalone numbers
/// are always candidates; everything else must carry at least one substantial,
/// non-stopword token and be mostly alphanumeric.
fn informative(s: &str) -> bool {
    let s = s.trim();
    if parse_number(s).is_some() {
        return true;
    }
    let length = s.chars().count();
    if length < 3 {
        return false;
    }
    let alphanumeric = s.chars().filter(|c| c.is_alphanumeric()).count();
    if (alphanumeric as f64) / (length.max(1) as f64) < 0.6 {
        return false;
    }
    norm_tokens(s)
        .iter()
        .any(|t| t.chars().count() >= 3 && !STOPWORDS.contains(t.as_str()))
}

/// Whether two terse answers agree.
///
/// Typed quantity equality when both sides are quantities, exact normalized
/// token-sequence equality otherwise. A quantity never agrees with prose, and
/// `Paris` never agrees with `Paris, Texas`.
pub fn match_answers(a: &str, b: &str) -> bool {
    match (parse_quantity(a), parse_quantity(b)) {
        (Some(qa), Some(qb)) => qa.agrees_with(qb),
        // One side is a bare quantity and the other is not: different kinds of
        // answer, so they cannot agree.
        (Some(_), None) | (None, Some(_)) => false,
        (None, None) => {
            let ta = norm_tokens(a);
            !ta.is_empty() && ta == norm_tokens(b)
        }
    }
}

/// The span a longform message actually concludes with.
///
/// The last explicit answer marker wins; failing that, the last non-empty line.
/// A mention anywhere else in the text is reasoning, not a conclusion.
fn extract_final_answer(message: &str) -> String {
    let mut marked: Option<String> = None;
    for line in message.lines() {
        if let Some(caps) = ANSWER_MARKER.captures(line)
            && let Some(tail) = caps.get(1)
            && !tail.as_str().trim().is_empty()
        {
            marked = Some(tail.as_str().trim().to_string());
        }
    }
    if let Some(marked) = marked {
        return marked;
    }
    message
        .lines()
        .map(str::trim)
        .rfind(|line| !line.is_empty())
        .unwrap_or("")
        .to_string()
}

/// Whether any of the few tokens before position `index` negates what follows.
fn negated_at(tokens: &[String], index: usize) -> bool {
    tokens[index.saturating_sub(NEGATION_WINDOW)..index]
        .iter()
        .any(|t| NEGATIONS.contains(t.as_str()))
}

/// Reduces a span to its operative segment, reporting whether it stays ambiguous.
///
/// A same-line correction is structural, so `42, actually 17` resolves to `17`.
/// A segment that still carries negation or contradiction language cannot be
/// resolved mechanically at all — `X is not correct; Canada is` names two
/// countries and endorses neither in a form this can read — so it is ambiguous,
/// and ambiguity never becomes agreement.
fn resolve_span(span: &str) -> (String, bool) {
    let parts: Vec<&str> = CORRECTION_SPLIT.split(span).collect();
    let operative = match parts.last() {
        Some(last) if !last.trim().is_empty() => last.trim().to_string(),
        _ => span.trim().to_string(),
    };
    let ambiguous = norm_tokens(&operative)
        .iter()
        .any(|t| NEGATIONS.contains(t.as_str()));
    (operative, ambiguous)
}

/// A quantity found inside a span, with the byte offset it starts at.
struct SpanQuantity {
    value: f64,
    unit: Unit,
    start: usize,
}

/// Whether a character blocks a quantity from starting right after it.
fn blocks_before(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '.' | '€' | '£' | '$')
}

/// Whether a character blocks a quantity from ending right before it.
fn blocks_after(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '.' | '%')
}

/// Finds the quantities in a span that stand as values in their own right.
///
/// A quantity must be delimited: it may not run into a surrounding word,
/// decimal, or unit marker. Without that, `42` would match inside `420`.
///
/// The reference expresses the delimiters as regex lookaround, which the `regex`
/// crate does not support. Filtering whole regex matches after the fact is not
/// equivalent, because a backtracking engine retries a *shorter* extent when the
/// trailing delimiter fails — in `=5,,million`, `5,,` is rejected but `5,` is a
/// valid quantity, and only backtracking finds it. This scans the span directly
/// and reproduces that retry order: currency marker, then digit run, then
/// decimal tail, then percent, each tried longest-first.
fn span_quantities(span: &str) -> Vec<SpanQuantity> {
    let chars: Vec<char> = span.chars().collect();
    let at = |i: usize| chars.get(i).copied();
    let mut found = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        if at(start.wrapping_sub(1)).is_some_and(|c| start > 0 && blocks_before(c)) {
            start += 1;
            continue;
        }
        match scan_quantity(&chars, start) {
            Some((end, quantity)) => {
                found.push(SpanQuantity {
                    value: quantity.value,
                    unit: quantity.unit,
                    // Byte offset, for slicing the prefix the caller tokenizes.
                    start: chars[..start].iter().map(|c| c.len_utf8()).sum(),
                });
                start = end.max(start + 1);
            }
            None => start += 1,
        }
    }
    found
}

/// Scans one quantity at `start`, trying each quantifier longest-first.
///
/// Returns the end index and the parsed quantity for the first extent whose
/// trailing delimiter holds, mirroring how a backtracking engine resolves the
/// reference's pattern.
fn scan_quantity(chars: &[char], start: usize) -> Option<(usize, Quantity)> {
    let at = |i: usize| chars.get(i).copied();
    let digit = |i: usize| at(i).is_some_and(|c| c.is_ascii_digit());

    // `[$€£]?` is greedy: prefer consuming the marker, then try without it.
    let has_marker = at(start).is_some_and(|c| matches!(c, '$' | '€' | '£'));
    for marker_len in if has_marker { [1, 0].as_slice() } else { [0].as_slice() } {
        let number_start = start + marker_len;
        if !digit(number_start) {
            continue;
        }
        // `[\d,]*` after the leading digit, longest-first.
        let mut run = 0;
        while at(number_start + 1 + run).is_some_and(|c| c.is_ascii_digit() || c == ',') {
            run += 1;
        }
        for run_len in (0..=run).rev() {
            let after_run = number_start + 1 + run_len;
            // `(?:\.\d+)?` is greedy, and its `\d+` backtracks too.
            let mut tail = 0;
            if at(after_run) == Some('.') {
                while digit(after_run + 1 + tail) {
                    tail += 1;
                }
            }
            let ends = (1..=tail)
                .rev()
                .map(|t| after_run + 1 + t)
                .chain(std::iter::once(after_run));
            for number_end in ends {
                // `(%?)` is greedy, then the trailing delimiter must hold.
                let with_percent = [(number_end + 1, true), (number_end, false)];
                let without = [(number_end, false)];
                let percent_options: &[(usize, bool)] = if at(number_end) == Some('%') {
                    &with_percent
                } else {
                    &without
                };
                for (end, is_percent) in percent_options {
                    if at(*end).is_some_and(blocks_after) {
                        continue;
                    }
                    let text: String = chars[number_start..number_end].iter().collect();
                    let Some(value) = parse_number(&text) else {
                        continue;
                    };
                    let unit = match (*marker_len, *is_percent) {
                        (1, _) => Unit::Currency(at(start)?),
                        (_, true) => Unit::Percent,
                        _ => Unit::None,
                    };
                    return Some((*end, Quantity { value, unit }));
                }
            }
        }
    }
    None
}

/// Whether the answer appears in the span as the span's own claim.
///
/// A quantity answer matches a typed quantity of equal value and unit; a text
/// answer matches the whole span, or a contiguous run of tokens within it. A
/// single short token is not enough to match a longer span, since a common word
/// would land anywhere. Any match preceded by negation is a rejected candidate,
/// not an answer.
fn match_in_span(answer: &str, span: &str) -> bool {
    let answer_tokens = norm_tokens(answer);

    if let Some(target) = parse_quantity(answer) {
        return span_quantities(span).into_iter().any(|found| {
            let candidate = Quantity {
                value: found.value,
                unit: found.unit,
            };
            if !candidate.agrees_with(target) {
                return false;
            }
            let prefix = norm_tokens(&span[..found.start]);
            let length = prefix.len();
            !negated_at(&prefix, length)
        });
    }

    let span_tokens = norm_tokens(span);
    if span_tokens == answer_tokens {
        return true;
    }
    // A lone short token would match incidentally, so a run has to be either
    // several tokens long or one substantial one.
    let distinctive = answer_tokens.len() >= 2
        || answer_tokens
            .first()
            .is_some_and(|t| t.chars().count() >= 5);
    if !distinctive {
        return false;
    }
    // A run longer than the span cannot occur in it, and `windows` requires a
    // non-zero width; both are guarded here rather than relied on from above.
    let k = answer_tokens.len();
    if k == 0 || k > span_tokens.len() {
        return false;
    }
    span_tokens
        .windows(k)
        .enumerate()
        .any(|(i, run)| run == answer_tokens && !negated_at(&span_tokens, i))
}

/// Whether a witness answer agrees with an attempt's final message.
///
/// [`Tri::Unknown`] means the message's concluding span carries unresolved
/// negation or contradiction: the answer branch treats that as no agreement and
/// falls through to its verifier rungs, rather than guessing.
pub fn match_answer_verdict(answer: &str, message: &str) -> Tri {
    let answer = answer.trim();
    if !informative(answer) || message.trim().is_empty() {
        return Tri::No;
    }
    let (operative, ambiguous) = resolve_span(&extract_final_answer(message));
    if ambiguous {
        return Tri::Unknown;
    }
    if match_in_span(answer, &operative) {
        Tri::Yes
    } else {
        Tri::No
    }
}
