#!/bin/bash
# shellcheck disable=SC2016 # Dollar-prefixed names in jq programs are jq variables.
#
# End-of-week GitHub activity report for the current repository.

set -euo pipefail

usage() {
 printf 'Usage: %s [github-username]\n' "${0##*/}"
 printf '\n'
 printf 'Report repository activity from the last six days.\n'
 printf 'If github-username is supplied, include a section for that user.\n'
}

if (( $# > 1 )); then
 usage >&2
 exit 2
fi

if [[ ${1:-} == --help ]]; then
 usage
 exit 0
fi

if [[ ${1:-} == -* ]]; then
 printf 'Error: unknown option: %s\n' "$1" >&2
 usage >&2
 exit 2
fi

readonly gardener="${1:-}"

required_commands=(gh jq)
if [[ -n $gardener ]]; then
 required_commands+=(xargs)
fi
for command in "${required_commands[@]}"; do
 if ! command -v "$command" >/dev/null 2>&1; then
  printf 'Error: %s is required.\n' "$command" >&2
  exit 1
 fi
done

repo="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
since="$(date -u -d '6 days ago' +%Y-%m-%dT%H:%M:%SZ)"
now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

report_tmp="$(mktemp -d)"
trap 'rm -rf -- "$report_tmp"' EXIT
readonly pulls_file="$report_tmp/pulls.json"
readonly issues_file="$report_tmp/issues.json"
readonly issue_comments_file="$report_tmp/issue-comments.json"
readonly review_comments_file="$report_tmp/review-comments.json"
readonly commits_file="$report_tmp/commits.json"

# Pulls and issues are downloaded once so every total uses the same snapshot.
gh api --paginate -X GET "repos/$repo/pulls" \
 -f state=all -f sort=created -f direction=asc -f per_page=100 |
 jq -s 'add // []' > "$pulls_file"

gh api --paginate -X GET "repos/$repo/issues" \
 -f state=all -f sort=created -f direction=asc -f per_page=100 |
 jq -s '[(add // [])[] | select(.pull_request == null)]' > "$issues_file"

gh api --paginate -X GET "repos/$repo/issues/comments" \
 -f since="$since" -f per_page=100 |
 jq -s 'add // []' > "$issue_comments_file"

gh api --paginate -X GET "repos/$repo/pulls/comments" \
 -f since="$since" -f per_page=100 |
 jq -s 'add // []' > "$review_comments_file"

count_items() {
 local file="$1" filter="$2"
 jq --arg since "$since" --arg user "$gardener" "[$filter] | length" "$file"
}

pulls_open_at_start="$(count_items "$pulls_file" \
 '.[] | select(.created_at < $since and (.closed_at == null or .closed_at >= $since))')"
pulls_open_draft="$(count_items "$pulls_file" \
 '.[] | select(.state == "open" and .draft)')"
pulls_open_ready="$(count_items "$pulls_file" \
 '.[] | select(.state == "open" and (.draft | not))')"
pulls_open_now=$((pulls_open_draft + pulls_open_ready))
pulls_opened="$(count_items "$pulls_file" '.[] | select(.created_at >= $since)')"
pulls_closed="$(count_items "$pulls_file" \
 '.[] | select(.closed_at != null and .closed_at >= $since)')"
pulls_merged="$(count_items "$pulls_file" \
 '.[] | select(.merged_at != null and .merged_at >= $since)')"

issues_open_at_start="$(count_items "$issues_file" \
 '.[] | select(.created_at < $since and (.closed_at == null or .closed_at >= $since))')"
issues_open_now="$(count_items "$issues_file" '.[] | select(.state == "open")')"
issues_opened="$(count_items "$issues_file" '.[] | select(.created_at >= $since)')"
issues_closed="$(count_items "$issues_file" \
 '.[] | select(.closed_at != null and .closed_at >= $since)')"

issue_comments="$(count_items "$issue_comments_file" \
 '.[] | select(.created_at >= $since and ((.user.login // "") | endswith("[bot]") | not))')"
review_comments="$(count_items "$review_comments_file" \
 '.[] | select(.created_at >= $since and ((.user.login // "") | endswith("[bot]") | not))')"
comments_total=$((issue_comments + review_comments))

if [[ -n $gardener ]]; then
 gh api --paginate -X GET "repos/$repo/commits" \
  -f author="$gardener" -f since="$since" -f until="$now" -f per_page=100 |
  jq -s 'add // []' > "$commits_file"

 my_pulls_opened="$(count_items "$pulls_file" \
  '.[] | select(.created_at >= $since and .user.login == $user)')"
 my_pulls_merged="$(count_items "$pulls_file" \
  '.[] | select(.merged_at != null and .merged_at >= $since and .user.login == $user)')"
 my_merges="$(count_items "$pulls_file" \
  '.[] | select(.merged_at != null and .merged_at >= $since and .merged_by.login == $user)')"
 my_issues_opened="$(count_items "$issues_file" \
  '.[] | select(.created_at >= $since and .user.login == $user)')"
 my_issue_comments="$(count_items "$issue_comments_file" \
  '.[] | select(.created_at >= $since and .user.login == $user)')"
 my_review_comments="$(count_items "$review_comments_file" \
  '.[] | select(.created_at >= $since and .user.login == $user)')"
 my_comments_total=$((my_issue_comments + my_review_comments))
 # A review updates its PR, so only recently updated PRs need to be checked. The
 # independent API requests are parallelized to keep the report quick to run.
 my_reviews="$(
  jq -r --arg since "$since" '.[] | select(.updated_at >= $since) | .number' "$pulls_file" |
   xargs -r -P 8 -I '{}' gh api --paginate -X GET "repos/$repo/pulls/{}/reviews" \
    -f per_page=100 \
    --jq ".[] | select(.submitted_at >= \"$since\" and .user.login == \"$gardener\") | .id" |
   wc -l | tr -d ' '
 )"
 my_commits="$(jq 'length' "$commits_file")"
fi

print_stat() {
 printf '  %-40s %8s\n' "$1" "$2"
}

printf 'Community Gardener weekly report\n'
printf 'Repository: %s\n' "$repo"
printf 'Window:     %s through %s (UTC)\n' "$since" "$now"

printf '\nPULL REQUESTS\n'
print_stat 'Open at start of window' "$pulls_open_at_start"
print_stat 'Open now - total' "$pulls_open_now"
print_stat 'Open now - draft' "$pulls_open_draft"
print_stat 'Open now - ready for review' "$pulls_open_ready"
print_stat 'Opened during window' "$pulls_opened"
print_stat 'Closed during window (includes merged)' "$pulls_closed"
print_stat 'Merged during window' "$pulls_merged"

printf '\nISSUES\n'
print_stat 'Open at start of window' "$issues_open_at_start"
print_stat 'Open now' "$issues_open_now"
print_stat 'Opened during window' "$issues_opened"
print_stat 'Closed during window' "$issues_closed"

printf '\nCOMMENTS (BOT ACCOUNTS EXCLUDED)\n'
print_stat 'Issue and PR conversation comments' "$issue_comments"
print_stat 'Inline PR review comments' "$review_comments"
print_stat 'Total comments' "$comments_total"

if [[ -n $gardener ]]; then
 printf '\nACTIVITY BY %s\n' "$gardener"
 print_stat 'PRs opened' "$my_pulls_opened"
 print_stat 'Authored PRs merged' "$my_pulls_merged"
 print_stat 'PRs merged as merger' "$my_merges"
 print_stat 'Issues opened' "$my_issues_opened"
 print_stat 'Issue and PR conversation comments' "$my_issue_comments"
 print_stat 'Inline PR review comments' "$my_review_comments"
 print_stat 'Total comments written' "$my_comments_total"
 print_stat 'PR reviews submitted' "$my_reviews"
 print_stat 'Commits on the default branch' "$my_commits"
fi
