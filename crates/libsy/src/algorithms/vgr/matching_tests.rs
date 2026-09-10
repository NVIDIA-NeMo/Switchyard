// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral tests for answer agreement.
//!
//! Each test guards one class of false agreement that a looser matcher admits.
//! Every case here is drawn from the reference implementation's own suite, where
//! they exist because a real matcher shipped without them and committed wrong
//! answers: numeric prefix collisions, substring containment, code fragments
//! scored as answers, and mentions read as conclusions.
//!
//! Exhaustive equivalence with the reference is checked out of tree by a
//! differential harness; it is not duplicated here.

use super::matching::{match_answer_verdict, match_answers};
use super::rules::Tri;

/// Whether the answer is found to agree, discarding the ambiguity distinction.
fn agrees(answer: &str, message: &str) -> bool {
    match_answer_verdict(answer, message) == Tri::Yes
}

/// From `test_numeric_values_not_prefixes`.
#[test]
fn numbers_compare_by_value_never_by_prefix() {
    // A normalization that stripped trailing zeros once made "100" equal "1".
    assert!(!match_answers("100", "1"));
    assert!(!match_answers("10", "1"));
    assert!(!match_answers("5", "50"));
    assert!(match_answers("100", "100"));
    // Formatting is not meaning: separators and trailing zeros are noise.
    assert!(match_answers("1,234", "1234"));
    assert!(match_answers("5.0", "5"));
    // A number never agrees with prose.
    assert!(!match_answers("penguin", "8"));
}

/// Decimal comparison stays exact regardless of magnitude.
#[test]
fn decimals_compare_canonically_without_floating_point_rounding() {
    assert!(!match_answers("1,000,000,000", "1,000,000,001"));
    assert!(!match_answers("9007199254740992", "9007199254740993"));
    assert!(match_answers(
        "9007199254740993.000",
        "9,007,199,254,740,993"
    ));
    assert!(!agrees("1000000000", "Final answer: 1000000001 events"));
}

/// From `test_numeric_values_not_prefixes`.
#[test]
fn units_are_part_of_the_value() {
    // Same magnitude, different claim.
    assert!(!match_answers("$5.00", "5"));
    assert!(!match_answers("$5", "€5"));
    assert!(!match_answers("12%", "12"));
    assert!(match_answers("$5.00", "$5"));
    assert!(match_answers("12%", "12 %"));

    // Units survive into composite answers, where the quantity is embedded in
    // surrounding words rather than standing alone.
    assert!(!match_answers("$5 million", "€5 million"));
    assert!(!match_answers("12% increase", "12 increase"));
    assert!(!match_answers("$5 USD", "€5 USD"));
    assert!(match_answers("$5 million", "$5 million"));

    // And they survive anchoring into a longform message.
    assert!(!agrees("$5 million", "Final answer: €5 million"));
    assert!(agrees("€5 million", "Final answer: €5 million"));
    assert!(!agrees("$5", "Final answer: €5"));
}

/// From `test_text_equality_not_containment`.
#[test]
fn text_agreement_is_equality_not_containment() {
    // Containment made "no" agree with "not possible".
    assert!(!match_answers("no", "not possible"));
    assert!(!match_answers("Paris", "Paris, Texas"));
    // Case and trailing punctuation are noise; the token sequence is not.
    assert!(match_answers("Paris", "paris"));
    assert!(match_answers("Jerome Wiesner", "jerome  wiesner."));
}

/// Unicode letters remain significant and apostrophe variants normalize alike.
#[test]
fn unicode_tokens_preserve_letters_and_close_negation_bypasses() {
    assert!(!match_answers("café", "cafè"));
    assert!(match_answers("CAFÉ", "café"));
    assert!(match_answers("O'Brien", "O’Brien"));
    for message in ["Final answer: isn't 17", "Final answer: isn’t 17"] {
        assert_eq!(match_answer_verdict("17", message), Tri::Unknown);
    }
}

/// From `test_fragments_never_match_messages`.
#[test]
fn code_fragments_and_stopwords_never_match() {
    // Measured on a real benchmark: nine of ten "strict agreement" commits were
    // fragments like these matching incidentally inside a longer message.
    let message = "## Report\nif retry: report += '| Block'\n\
                   The z_ score threshold was 3.\nFinal answer: penguin\n";
    for fragment in [
        "if",
        "| Block",
        "retry",
        "z_",
        "report += \"",
        "## 1",
        "within =",
    ] {
        assert!(!agrees(fragment, message), "matched fragment {fragment:?}");
    }
}

/// From `test_mentions_are_not_answers`.
#[test]
fn matching_anchors_to_the_span_the_message_concludes_with() {
    // A candidate raised and then discarded during reasoning is not the answer.
    let considered_then_rejected = "We considered 42 as a candidate early on.\n\
                                    After checking the sources, 42 is wrong.\nFinal answer: 17";
    assert!(!agrees("42", considered_then_rejected));
    assert!(agrees("17", considered_then_rejected));

    // With no explicit marker, the last non-empty line is the conclusion.
    let compared = "The United States was a contender in this comparison.\n\
                    The country with the longest coastline is Canada.";
    assert!(!agrees("United States", compared));
    assert!(agrees("Canada", compared));
}

/// Marker extraction starts after the final marker, even on the same line.
#[test]
fn final_answer_marker_supersedes_earlier_same_line_markers() {
    let message = "Answer: 42; Final answer: 17";
    assert_eq!(match_answer_verdict("42", message), Tri::No);
    assert_eq!(match_answer_verdict("17", message), Tri::Yes);
}

/// From `test_corrections_and_negations`.
#[test]
fn corrections_supersede_and_contradictions_resolve_to_unknown() {
    // A later correction is the operative answer, across lines and within one.
    let across_lines = "Answer: 42\nSome checking...\nCorrection: 17";
    assert!(!agrees("42", across_lines));
    assert!(agrees("17", across_lines));
    let same_line = "Final answer: 42, actually 17";
    assert!(!agrees("42", same_line));
    assert!(agrees("17", same_line));
    assert!(!agrees("42", "Final answer: not 42, it is 17"));

    // A span that names two candidates and endorses neither readably cannot be
    // resolved mechanically. It is indeterminate for *every* candidate, which
    // the answer branch treats as no agreement rather than as a guess.
    for contradictory in [
        "Final answer: United States is not correct; Canada is",
        "Final answer: not United States; it is Canada",
    ] {
        for candidate in ["United States", "Canada"] {
            assert_eq!(
                match_answer_verdict(candidate, contradictory),
                Tri::Unknown,
                "{candidate:?} in {contradictory:?}"
            );
        }
    }
}

/// Infix contrasts remain ambiguous while postfix corrections restart the answer.
#[test]
fn infix_contrasts_do_not_masquerade_as_postfix_corrections() {
    for contrast in [
        "Final answer: 17 instead of 42",
        "Final answer: 17, instead of 42",
    ] {
        assert_eq!(match_answer_verdict("17", contrast), Tri::Unknown);
        assert_eq!(match_answer_verdict("42", contrast), Tri::Unknown);
    }

    let correction = "Final answer: 42; instead, 17";
    assert_eq!(match_answer_verdict("42", correction), Tri::No);
    assert_eq!(match_answer_verdict("17", correction), Tri::Yes);
}

/// Multiple viable values never collapse to a matching subset.
#[test]
fn multiple_typed_final_values_resolve_to_unknown() {
    let quantities = "Final answer: 17 and 42";
    assert_eq!(match_answer_verdict("17", quantities), Tri::Unknown);
    assert_eq!(match_answer_verdict("42", quantities), Tri::Unknown);

    let location = "Final answer: Paris, Texas";
    assert_eq!(match_answer_verdict("Paris", location), Tri::Unknown);
    assert_eq!(match_answer_verdict("Paris, Texas", location), Tri::Yes);
}

/// From `test_real_answers_match_messages`.
#[test]
fn genuine_answers_still_match_their_messages() {
    // The strictness above is worthless if it rejects real agreement.
    let message = "I checked the video.\nThe species shown is a penguin.\nFinal answer: penguin";
    assert!(agrees("penguin", message));
    assert!(agrees(
        "Jerome Wiesner",
        "It was Jerome Wiesner who said this."
    ));
    assert!(agrees("42", "The count came to 42 events total."));

    // But a number embedded in a longer number is a different number, and a
    // short token absent from the span does not match it.
    assert!(!agrees("42", "The count came to 420 events total."));
    assert!(!agrees("cat", message));
}
