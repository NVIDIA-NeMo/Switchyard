#!/bin/bash
#
# New activity on the repo over a configurable overnight period.

usage() {
 printf 'Usage: %s [hours]\n' "${0##*/}"
}

if (( $# > 1 )); then
 usage >&2
 exit 2
fi

if [[ ${1:-} == --help ]]; then
 usage
 exit 0
fi

hours="${1:-16}"
if [[ ! $hours =~ ^[0-9]+([.][0-9]+)?$ ]]; then
 printf 'Error: hours must be a non-negative number.\n' >&2
 usage >&2
 exit 2
fi

repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
since="$(date -u -d "$hours hours ago" +%Y-%m-%dT%H:%M:%SZ)"

printf 'Overnight activity on %s. Last %s hours.\n\n' "$repo" "$hours"

{
 printf 'WHEN\tKIND\tACTOR\tURL\tTEXT\n'

 # Newly created Issues and PRs
 gh api --paginate -X GET "repos/$repo/issues" \
 -f state=all -f since="$since" -f sort=updated -f direction=asc -f per_page=100 |
 jq -r --arg since "$since" '
 .[]
 | select(.created_at >= $since)
 | select(.user.login | endswith("[bot]") | not)
 | [.created_at,
 (if .pull_request then "NEW PR" else "NEW ISSUE" end),
 .user.login, .html_url, .title]
 | @tsv '

 # New regular comments on Issues and PR conversations
 gh api --paginate -X GET "repos/$repo/issues/comments" \
 -f since="$since" -f per_page=100 |
 jq -r --arg since "$since" '
 .[]
 | select(.created_at >= $since)
 | select(.user.login | endswith("[bot]") | not)
 | [.created_at, "ISSUE/PR COMMENT", .user.login, .html_url,
 (.body | split("\n")[0] | gsub("\t"; " "))]
 | @tsv '

 # New inline PR review comments
 gh api --paginate -X GET "repos/$repo/pulls/comments" \
 -f since="$since" -f per_page=100 |
 jq -r --arg since "$since" '
 .[]
 | select(.created_at >= $since)
 | select(.user.login | endswith("[bot]") | not)
 | [.created_at, "PR REVIEW COMMENT", .user.login, .html_url,
 (.body | split("\n")[0] | gsub("\t"; " "))]
 | @tsv '
} | column -ts $'\t'
