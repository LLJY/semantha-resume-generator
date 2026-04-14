---
project_id: sample_project
motivation_summary: Built because the existing local auth stack was slow, cold-start heavy, and did not support the target GPU platform.
problem_trigger: Upstream tooling lacked AMD GPU support and had poor auth latency.
personal_connection: I wanted a Linux-native solution I would actually use daily.
why_now: I had the target hardware on hand and a real need for low-latency local auth.
constraints_or_stakes: Needed to fail safely and integrate with existing PAM-based login flows.
preferred_role_family_targets: [linux-systems, platform-engineering, security-engineering]
context_keywords: [painkiller project, self-use, low-latency, platform gap]
---

## Why This Existed

The code only shows a daemon and PAM module. It does not show that the real motivation was fixing a daily usability gap in the Linux stack.

## Personal Context

This was built for a machine and workflow I actually used, not just as a demo.

## Resume Intent

Frame this as systems engineering motivated by a real platform gap, not as a toy ML project.

## Things Not Obvious From The Code

The strongest story is replacing a frustrating existing setup with a practical one that respected Linux deployment constraints.
