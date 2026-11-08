# Isaac Sim demo videos

Screen recordings of the Isaac Sim 6.0 demo (Franka Panda) executing spoken or typed
commands, one video per command parser. The `.mp4` files are not in git (they are
gitignored); they are attached to GitHub release **v1.0**.

| file | parser shown |
|---|---|
| `distilbert.mp4` | DistilBERT classifier (`classifier:distilbert`) |
| `gpt4omini_cot.mp4` | gpt-4o-mini, chain-of-thought prompt (`cot`) |
| `gpt4omini_fewshot.mp4` | gpt-4o-mini, few-shot prompt (`few_shot`) |
| `gpt4omini_rag.mp4` | gpt-4o-mini, retrieval-augmented prompt (`rag`) |
| `gpt4omini_rolebased.mp4` | gpt-4o-mini, role-based prompt (`role_based`) |
| `qwen25_3b_zeroshot.mp4` | Qwen2.5-3B, zero-shot prompt (`zero_shot`) |

These are demonstrations, not measurements: no number in the thesis comes from them
(see the demo/measurement firewall in `docs/REPO_MAP.md`). Original file names are
listed in `docs/MOVED_PATHS.md`.
