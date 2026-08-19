// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict comparison-verdict parsing.

/// Parses exactly one `Final verdict: Solution N` line in range.
pub fn parse_verdict(text: &str, candidate_count: usize) -> Option<usize> {
    let mut matches = text.lines().filter_map(parse_line);
    let position = matches.next()?;
    if matches.next().is_some() || position == 0 || position > candidate_count {
        return None;
    }
    Some(position)
}

fn parse_line(line: &str) -> Option<usize> {
    // Tolerate Markdown emphasis models add around the verdict (**bold**, *italic*, `code`,
    // _underline_). Trailing prose ("Solution 1 later") and a missing space ("Solution1") stay
    // rejected: only emphasis punctuation is stripped, never letters, digits, or interior words.
    let emphasis = |c: char| matches!(c, '*' | '_' | '`');
    let (label, choice) = line.trim().split_once(':')?;
    if !label
        .trim()
        .trim_matches(emphasis)
        .trim()
        .eq_ignore_ascii_case("final verdict")
    {
        return None;
    }
    let choice = choice.trim().trim_matches(emphasis).trim();
    let prefix = choice.get(..8)?;
    if !prefix.eq_ignore_ascii_case("solution") {
        return None;
    }
    let suffix = choice.get(8..)?;
    if !suffix.starts_with(char::is_whitespace) {
        return None;
    }
    let number = suffix.trim().trim_matches(emphasis).trim();
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    number.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_one_verdict_in_range() {
        assert_eq!(
            parse_verdict("Evidence.\nFinal verdict: Solution 2", 2),
            Some(2)
        );
        assert_eq!(parse_verdict(" final VERDICT: solution 1 ", 2), Some(1));
    }

    #[test]
    fn accepts_markdown_wrapped_verdict() {
        // Real judges (e.g. Claude Haiku) end with a bold verdict; strip the emphasis, keep the vote.
        assert_eq!(
            parse_verdict("Analysis.\n**Final verdict: Solution 1**", 2),
            Some(1)
        );
        assert_eq!(parse_verdict("Final verdict: **Solution 2**", 2), Some(2));
        assert_eq!(parse_verdict("`Final verdict: Solution 1`", 2), Some(1));
        // Emphasis stripping must not start accepting trailing prose or a missing space.
        assert_eq!(
            parse_verdict("**Final verdict: Solution 1 later**", 2),
            None
        );
        assert_eq!(parse_verdict("**Final verdict: Solution1**", 2), None);
    }

    #[test]
    fn rejects_missing_duplicate_and_out_of_range_verdicts() {
        assert_eq!(parse_verdict("Solution 1", 2), None);
        assert_eq!(
            parse_verdict("Final verdict: Solution 1\nFinal verdict: Solution 2", 2),
            None
        );
        assert_eq!(parse_verdict("Final verdict: Solution 3", 2), None);
        assert_eq!(parse_verdict("Final verdict: Solution 1 later", 2), None);
        assert_eq!(parse_verdict("Final verdict: Solution1", 2), None);
    }
}
