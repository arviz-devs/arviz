## Proposal: Split migration guide into two user-focused documents

**Context:** I think the current [migration guide](https://github.com/arviz-devs/arviz/blob/main/docs/source/user_guide/migration_guide.md) is too long and reads more like developer-facing documentation. For the 1.0 release, I’d like to propose splitting it into two documents with different purposes.

**What I’m proposing:**

1. **Migration guide (short, “need to know”)**  
   - **Audience:** Anyone upgrading from pre-1.0 who needs to fix breakage.  
   - **Content:** Breaking changes only; minimal before/after code; “what must I change?”  
   - **Length:** Short enough to read in a few minutes, with links to the second doc for detail.

2. **“What’s new in ArviZ 1.0?” (informative)**  
   - **Audience:** Users who want to understand what changed and how to use new features.  
   - **Content:** Full narrative (DataTree, rcParams, stats, plots, etc.) with context, rationale, and examples—still user-focused, not internal/developer detail.  
   - **Length:** Can be longer; this is the “full story” for users.

**Why:** One doc stays actionable for people in a hurry; the other serves everyone who wants to read about 1.0 without wading through implementation details.

**What I’d like to do:** I’m prepared to make these changes and open a that would add “What’s new in 1.0?”, trim the migration guide to need-to-know content, cross-link the two, and update the docs index. I’d welcome feedback on the approach before or during the PR.
