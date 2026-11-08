# History rewrite, 9 November 2026

Before the repository was published, its whole history (every commit of `main`, `cleanup` and
the tag `thesis-results-v1`) was rewritten with git-filter-repo to remove personal details. No
file was added or removed by the rewrite and no result changed; only the strings below.

## What was replaced

| what | before | after |
|---|---|---|
| author and committer of every commit | two identities with a personal e-mail address | `chappiethebot <42741718+chappiethebot@users.noreply.github.com>` |
| the Windows user name in paths, in every file version | `C:\Users\<name>` in every spelling (single, doubled and quadrupled backslashes, forward slashes, any case) | the same spelling with `USER`, for example `C:\Users\USER\llm-6dof-arm` |
| six lines naming the author by first name | `thesis2/RUN_ALL_TESTS.md` (2), `thesis3/bench/RUNBOOK.md`, `thesis3/bench/run_bench.py`, `docs/DEMO_PORTABLE.md`, `audit/PREREGISTRATION_v4.md` | "the author" ("the thesis author" in the preregistration's author line) |

The author's full name appears only in `README.md` (citation, contact) and `LICENSE`.

## Consequences

- **Every commit hash changed.** Hashes cited in the thesis or in documents written before the
  rewrite (for example `3000198`, the frozen scorer) refer to the old history. Look them up in
  [`COMMIT_MAP.md`](COMMIT_MAP.md): old `3000198` is `ea3d5984c2c2` now, and the tag
  `thesis-results-v1` points at `63bfe8c0e342` (old `dfe8b83`).
- **Paths read `C:\Users\USER\...`.** Logs, result files and scripts recorded absolute paths of the
  measurement machine; they now name the user `USER`. Scripts that hard-code such a path
  (listed under Known limitations in the README) need it changed to the local one anyway.
- **Nothing measured changed.** The two test sets, every result CSV and every checksum recorded
  in a tracked file are unchanged apart from the replaced strings; the README checks (smoke
  test, Tier 0, Tier 1) give the same output as before the rewrite.

The original history is kept privately by the author.
