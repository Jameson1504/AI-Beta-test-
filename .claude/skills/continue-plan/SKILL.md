---
name: continue-plan
description: Continue the Mythos roadmap - read PLAN.md's phase checklist, pick the next open item, and implement it. Use when starting a session without a specific task, instead of beginning a new parallel effort.
---

# Continue the Mythos roadmap

1. Read `PLAN.md` and find the roadmap phases marked 🟡 (partially done) or ⏭
   (not started). Read `README.md`'s "Scaling up" section for the same list.
2. Pick the lowest-numbered open item unless the user says otherwise. As of
   the last update the open items were:
   - Phase 2 remainder: a real GPU-scale run config + docs (can't execute on
     CPU-only web containers — prepare configs/scripts and say so honestly).
   - Phase 3 remainder: preference tuning (DPO) and a larger instruction
     dataset.
   - Phase 4: eval harness (perplexity + tasks), inference server,
     quantization.
3. Before writing code, check what already exists — past sessions duplicated
   effort because they didn't look. `git log --oneline --all` and a quick read
   of `mythos/` are mandatory.
4. Implement on the current session branch, run `/verify-mythos`, then update
   the phase status in BOTH `PLAN.md` and `README.md` in the same commit.
