---
name: switchyard-docs-novice-review
description: Review Switchyard documentation as a technically capable user who is new to the project. Use to find missing prerequisites, unexplained project terminology, incomplete setup steps, unclear success signals, and onboarding dead ends.
---

# Review Switchyard Documentation as a Newcomer

Adapted from NVIDIA Tech Docs Skill Library's `doc-persona-novice` workflow (Apache-2.0).

Read `AGENTS.md` and `docs/AGENTS.md` completely. Review without editing unless the user also asks
for fixes.

Assume solid general software knowledge but no prior Switchyard knowledge. Understand APIs, CLIs,
LLMs, YAML, Git, and terminals; do not manufacture confusion about standard concepts.

## Review Lenses

1. **First contact:** Can the reader explain what the documented feature does, why it exists, and
   when to use it?
2. **Prerequisites:** Are required packages, credentials, services, configuration, and platform
   constraints stated before the first step?
3. **Sequence:** Can the reader move from one step to the next without an implied action?
4. **Success signals:** Does each important operation explain how to tell that it worked?
5. **Recovery:** Are likely failures actionable or linked to the right troubleshooting material?
6. **Project terminology:** Are Switchyard-specific terms defined at first use or linked clearly?
7. **Navigation:** Can the reader find the next relevant task without understanding the repository
   layout?

## Findings

For each material issue, quote the smallest relevant passage, state where a capable newcomer gets
stuck, and propose the smallest correction. Distinguish project-specific jargon from normal
engineering terminology. End with an onboarding-readiness assessment and say clearly when no major
gaps are present.
