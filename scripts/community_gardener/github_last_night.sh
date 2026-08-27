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

# Keep a snapshot of currently open issues and non-draft PRs so activity on
# items that have since been closed or merged, and activity on drafts, is not
# reported.
open_items="$(mktemp)"
trap 'rm -f "$open_items"' EXIT
gh api --paginate -X GET "repos/$repo/issues" \
 -f state=open -f sort=updated -f direction=asc -f per_page=100 |
 jq -s --slurpfile prs <(
  gh api --paginate -X GET "repos/$repo/pulls" \
   -f state=open -f per_page=100 |
  jq -s '[add[] | select(.draft | not) | .issue_url]'
 ) '
 add
 | map(select(.pull_request == null or (.url as $url | $prs[0] | index($url))))
 ' > "$open_items"

report_comments() {
 local endpoint="$1" kind="$2" url_field="$3"
 gh api --paginate -X GET "repos/$repo/$endpoint" \
  -f since="$since" -f per_page=100 |
 jq -r --arg since "$since" --arg kind "$kind" --arg url_field "$url_field" \
  --slurpfile open "$open_items" '
 .[]
 | . as $comment
 | select(any($open[0][];
     .url == $comment[$url_field] or .pull_request.url == $comment[$url_field]))
 | select(.created_at >= $since)
 | select(.user.login | endswith("[bot]") | not)
 | [.created_at, $kind, .user.login, .html_url,
    (.body | split("\n")[0] | gsub("\t"; " "))]
 | @tsv '
}

printf 'Overnight activity on %s. Last %s hours.\n\n' "$repo" "$hours"

{
 printf 'WHEN\tKIND\tACTOR\tURL\tTEXT\n'

 # Newly created Issues and PRs
 jq -r --arg since "$since" '
 .[]
 | select(.created_at >= $since)
 | select(.user.login | endswith("[bot]") | not)
 | [.created_at,
 (if .pull_request then "NEW PR" else "NEW ISSUE" end),
 .user.login, .html_url, .title]
 | @tsv ' "$open_items"

 # New regular comments on Issues and PR conversations
 report_comments issues/comments 'ISSUE/PR COMMENT' issue_url

 # New inline PR review comments
 report_comments pulls/comments 'PR REVIEW COMMENT' pull_request_url
} | column -ts $'\t'
