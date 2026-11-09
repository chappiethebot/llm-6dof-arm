# Commit map

Old commit hash (before the history rewrite of 9 November 2026) -> new hash. Hashes cited in
the thesis or in older documents in this repository refer to the old history; look them up
here. See `docs/HISTORY_REWRITE.md`. 105 commits.

| date | old | new | subject |
|---|---|---|---|
| 2026-06-28 | `258b86ee9fd3106ab99c1896b571d30973ec7f32` | `3693bc2b6f7554d5310930ff602b000046072879` | Phase 0: project structure + smoke test |
| 2026-06-29 | `9b7585c9598b1569e62d0c12cb418f3aec84e4b7` | `abbc5ee5a7541ff878e608ad594cf2c79656c3dc` | Phase 0 complete: smoke test passing on Mac (mujoco, whisper, ollama, cloud) |
| 2026-07-05 | `43f45af86e6106b137a650b3ec6af1a40ec5738c` | `a24629a709424817cd35419bb1d7d482c3219952` | Phase 0: add RL + vision deps to requirements |
| 2026-07-05 | `ce74d7cbdf7674762db765bad22216eeee7e6f4c` | `9e9a0daf0f0b326d07c8ad035e44bad58a95a4a3` | Phase 0 delta: RL+vision deps, folder structure, updated smoke test |
| 2026-07-08 | `9861b5f4c1e7cf6765fdeea133c44f1f34b4cb9e` | `31624f12ce67a9c58e5a0d9150b08cf7df3266ab` | Phase 1 baseline: schema, validator, seen + sealed held-out sets |
| 2026-07-09 | `946ed1f901865f9e8154e1826ce43e93ce6c5043` | `fabb59c0f5633800cc86b72a7d771fda1893d6ae` | Phase 2: SimBackend with 7 verbs, obstacle avoidance, coordinate map |
| 2026-07-10 | `f812fdc66b529dc404d10a45f43f53d47344990d` | `28534034d7caaf73ab8ce5407542aff59c9ba07a` | Phase 2: reliable single-object real-grasp backend (pick_place/fetch/hold/inspect) |
| 2026-07-16 | `11d42cfc3a9ba20b2f4587dddb64bf2765a96942` | `f6b1edb43ac2e8b8c990e7c41ca2915d46242475` | requirements: add python-dotenv (fixes dotenv ImportError), add sounddevice for Phase 4, g |
| 2026-07-16 | `1fdf49d3c54da3f2df3f9c7f20aadbaae65942ee` | `cee542251f3af25b8ccec092bccbb21c95d0fad4` | Phase 3 step 3: dual-policy scorer (strict/normalized keys, ignore empty modifiers, failur |
| 2026-07-16 | `20f68cea80a525737f95d14776bcbb7b77f87070` | `caef4d2a94b4780e364b8c798c08979846ecde62` | Phase 3: end-to-end pipeline + package __init__ files |
| 2026-07-16 | `26e2dbc1d9525267f865de63a5b7d629ceabb1a5` | `3ff4de133a12bc35b93adbda583007ae0a894433` | Phase 3 step 3: per-object schema validator (shared legality rules, machine-tallyable reas |
| 2026-07-16 | `3fb08ffe3f160e76c2629931f6b3e38239734942` | `0cd0b47eef43f8ebca05a1f501ef5a2d6b1da63e` | Rough LLM->sim demo (cloud zero-shot -> Phase 2 backend), schema path fix |
| 2026-07-16 | `5476c047eb56d1cc047cb3330a9d19c4423451f0` | `b8c53a10af88307591f34decefe98e556828a28b` | Phase 3 hardening: type-guard validator + crash-proof run_once (nested model output no lon |
| 2026-07-16 | `8875d105a5d1029857e80db41499f872b1cf0a10` | `deae1cfb732cdd902db972bb7d6616616f1ba5b5` | Phase 3 step 1: LLM client interface (cloud + 3 local models), raw-output smoke test |
| 2026-07-16 | `bcfd9500610a8a5df2638ef226bbefb22362cefd` | `92992c543ea8bc438c62ae132743dbb3f46fa9e3` | Phase 3: manual REPL for typing commands into the pipeline (convenience, not measurement) |
| 2026-07-16 | `de1d2e42db8bd2f3ec7b65cc69a308581d4812f1` | `acc0d7ccb3038faa03c12ccc81c0e3f62bb8cba6` | Phase 3 step 2: JSON extractor (strip fences, find object in prose, reject non-JSON) |
| 2026-07-16 | `e8b48cca27137a44fe91b658fd3c74253bd98171` | `67c5dce1367f4dd81846720b0cbb6649a8bd0b1d` | Phase 3: 4-strategy prompt builder (injected few-shot; CoT reasoning-then-JSON, no contrad |
| 2026-07-18 | `1b94f22eab3d683d13e349a7678a1ab1677b399a` | `22cf45997ca858174d3f0cb7379d372d9b564f23` | Demo: terminal legend + block-alias labels; floor markers attempt (skips safely; mjpython  |
| 2026-07-18 | `2c5f1224b7cebe72642f79805f0fe2002933c23b` | `b44d70e9b7714a26c82d08e8519df01e7959e994` | Add deepseek-r1:7b as 5th model (reasoning) â€” grid now 5x4=20 cells |
| 2026-07-19 | `3380530c833b198b7a87c7231a93c55862ced06a` | `1e61a118fb0703ca43d89b7711ba1a9869c82bfb` | Phase 4: Mac runs under plain python not mjpython (fixes NSWindow crash); backend built be |
| 2026-07-19 | `5f1f9f45dab3e5f9bfb40dfd1ae90cd0ede56ecf` | `625169c7d61220f51578edf230089f2f00264364` | Phase 5: single-cell measurement runner (bench/run_cell.py); one row per run, T=0, fixed c |
| 2026-07-19 | `75950d7288eb38c605fabc338f158fe54df4d7f1` | `cd6727dc44c0ec7cc2e1a5945cd1b88da088c3cc` | Phase 4: live voice input (mic -> Whisper -> run_once) + keyboard/voice demo launcher |
| 2026-07-20 | `29161d4fb325ab57e476e4f5c7532b4be720cc98` | `e352782e85676844a118f4751c1d6bbcf34e0970` | Phase 5: grid wrapper (bench/run_grid.py) with append-only resume; add is_warmup column; f |
| 2026-07-20 | `30001987d8d114c6c7fc5da384052411fe96f0dc` | `ea3d5984c2c20f7d15e7a7d851949e0688a0945c` | Phase 5: grid wrapper with resume + is_warmup; corrected normalized scorer (representation |
| 2026-07-20 | `cae9ab4e4f7287b0af403f5be4fd18d1c6a24bc5` | `d55ed598b3630e936ec498ec353d5aa5624db04d` | Phase 5: wire few-shot (k=3 exemplars, excluded from test set); fix extract_json to take l |
| 2026-07-20 | `cb6e0b5747122eb0e733966e75317a783f21f6e6` | `fa0a5989e526e1a3cd7ceb2a5b4dd56dcd2e6bd6` | Phase 5: enable held-out grid pass (few-shot uses seen exemplars; seal flag required in wr |
| 2026-07-21 | `091378d5b31733e0606071fcbeb7a7f0eb80f625` | `f3a3d263756fd1817966f53551c26b3ce7d28fb4` | Phase 5: analysis script â€” accuracy heatmap, latency tails, generalization gap, cot stri |
| 2026-07-23 | `554936482b844f00cc68afb9d3cd84b129df15f2` | `4802e85ba6fbf1ac8b3ebb036c0ff113941a5e8c` | Phase 5: token/cost backfill; summary adds reject-rate, throughput, cloud cost per command |
| 2026-07-25 | `154818d52e605159b12991dda006a65cd5219d79` | `582c28f911d6a8b68a25b422d9bbb33901dd3104` | V2 datasets, phrase source, reachability measurement |
| 2026-07-25 | `30dd2d15ce8ec93030ed91a4d432759a21f533bf` | `f14fac4868a7cf1bfff5d5f72aa260c99cb58f7e` | V2: 3x3 grid schema, reachability sweep, seen_v2 + held_out_v2 datasets |
| 2026-07-25 | `31e3b6ebd405c54f67b26e820c9ffee4379fef1b` | `2e3b23c1efa9db130f91176f45ca2ab37ce4aed2` | Diagnostic A: cold-start explains all local T=0 variation; accuracy unaffected |
| 2026-07-25 | `3e0411279e74bf56c45396a347c8cb0c84aa8ff9` | `7a34aabe72f3ea92a9d2f9b0fd42220cabb1e814` | Capture Ollama reasoning traces; add thinking columns to results schema |
| 2026-08-07 | `b2a22b3e87cfb86523883608427a0a02359fd2fd` | `94b8697c8b145d072fb78f00a15d82e94b23c5ea` | V2 wiring: grid legend in prompts, dataset table in run_cell, legend ablation |
| 2026-08-07 | `e4744d277934be00d3fb5607fd5d35c2f5bb6b62` | `29205955859b1b61a574e025a5cd2600b537f1f5` | V2 grid wrapper: dataset-driven schema, class-balanced exemplars, run conditions logged |
| 2026-08-20 | `13d34990c9052454632a7578a60c5acf4ee76a24` | `e8d57664aeacfb88c1fb24e78c5c00b4fbc9b1bc` | V2 analysis: per-cell diagnostics, phrase-legend collision finding, thesis figures |
| 2026-08-20 | `73a4f310eb14e355d53c5e25fa0f697f9d4605b3` | `5e51830164d9de1ce333f07e2ddad21e393a1e83` | V2 demo: workspace symbol resolution, grid-aware backend and demos, off-table z check |
| 2026-08-26 | `3a4a43605394f3f32136469b8117db36445fe2b8` | `02d79d85a3ca2d53fed0314155eebf197ee8f096` | track sealed datasets (train_v2, dev_v2, held_out_v3=217), classifier label maps, LoRA dev |
| 2026-08-26 | `aea20caf4d591713c2ba30bced29a4ddad0e5302` | `9837ab02a1ee2360ba8da8b2d7fe39f7aeb93de8` | rung 3+4 clients, DistilBERT classifier, data gate: held_out_v3 181->217 container rows. M |
| 2026-08-27 | `0ff113eb6ccf76cf96aaf74963388386bed6e5e3` | `8cb59333dbaa7fa53a9b150596ffc7ced423e1fe` | figures: plain-language labels, per-figure captions, corrected accuracy wording |
| 2026-08-27 | `4d8266b5ed3b15fb524c9ad57eb2ea1d1cf4b5f1` | `75668a5da4f1d57bebb56186f4704ca18d614ef4` | audit: location/verb/target confusion matrices and v3 result figures |
| 2026-08-27 | `55dba3a82e84e103c03b703fae51b3162a03a41c` | `18d9dbd0adeb626a84993fa3bc738d2c94ef5805` | demo: roster from all_clients(), wire few-shot and rag examples, defer sim imports |
| 2026-08-27 | `7b71bcd0da90657db8c8b829590d27e331f28daa` | `9382ad9ae124f423a4e450e7f7080e8783246283` | audit: demonstrate the LoRA row collapse is an out-of-vocabulary failure |
| 2026-08-27 | `b8f13859a84c04b96a6073dc01f43051352ff831` | `f88f00320d8c5d0075888e90d45276cd207d82cd` | audit: implementation and validity audit of the v3 measurement |
| 2026-08-27 | `dba5ae9548655ece5e8f7cd804dd1a79011c6287` | `b130b97c33d121c4caa27d3ff1a5183014025547` | chore: pin dependencies and lock the audit-time environment |
| 2026-08-27 | `f9e2c39fae8d2b4010bd6a4c8211ed5f5ec22ed3` | `2c6f9ebcfc504d82a3244c28cb78ba6a50642694` | audit: explain the LoRA row-B collapse, surface parse failures, fix reproduction test |
| 2026-08-28 | `01aaa49ba9c531618237f5fb386d7e4086457f72` | `ab6ae21de0f478a4e5d435e9465cabf298c5eeab` | run_grid: record the legend version in the run manifest |
| 2026-08-28 | `1334f83e634e9c94fe5e35703b05916722a137b9` | `a9c1ffefd3a4d7a26d513aed68b037f1daa89c18` | prereg amendment: V4a-lite condition, written before running it |
| 2026-08-28 | `1b4729f72a42ef1e83b60b5f1e78c82b0903bde5` | `60d5128595e85d106c24a9e7532a5c99cd4a3b5a` | discard contaminated V4a local run; two concurrent writers on one output file |
| 2026-08-28 | `2f103ae4a623e0c21517361a967333608b131cd3` | `a305f6e22d973d0dd1c58abd45eaea0119a69116` | audit: exhaustive vocabulary-conflict scan of the command datasets |
| 2026-08-28 | `371426db391e607d58805fbc8ee13ba4a8373c6a` | `bce04de5038b36147d75d4ea03192fad42e800fc` | untrack the 9.3MB discarded CSV; keep its README and manifest |
| 2026-08-28 | `48b838a8c419702af4459624d255aa5b1a28760e` | `0f642004c63a195833c0332b5975272b1d31611f` | prereg: V4 legend and vocabulary conditions, written before any V4 measurement |
| 2026-08-28 | `d421e5d6f7d8edd04acc86ff907ae85c8dd030b8` | `e20a17b59071977827cdf47edad592c6b88da23f` | track the V4 condition datasets (v4b, v4c) |
| 2026-08-29 | `44b82abbefdbec7940bb7befcc8e7d3d599e6421` | `d594906b919523fb2e2ecada78370a4e67daff52` | V4 complete: all conditions run, rescored, analysed, figures |
| 2026-08-29 | `adb728515d2e82d96b3172dabdf8852dec416181` | `47f8276340296a5abaca5746943c5361b1dcaf23` | Add headless Isaac Sim 6.0 grid scene with front camera |
| 2026-08-29 | `af721e20c86297acb639a4c474ffcbcfde3859fb` | `a1081f2fc8b1e64eb5a1227e4b8d52e3ac402600` | Ignore out/; remove crashing windowed Isaac scene |
| 2026-08-29 | `b47404010f279ab068efeb426e8bfe5a1ec1c1f3` | `c090c9fad1d789fb13e9495a8af9823907447b38` | audit: DEFENCE_NUMBERS.md - one page of quotable numbers |
| 2026-08-29 | `bc57fb89423e8c157a1012dc6677ad1201e0ec02` | `5bda23b5863cb867586f2494ad320705163fd069` | demo: V5 legend + constrained decoding, wired as the default demo config |
| 2026-08-29 | `f27795d27c73702ed48356cd295f0f249d5e3a49` | `05df86bf91160ae1edfe0a5a9f7cff25a1ccd087` | step 9: classifier margins + offline escalation simulation |
| 2026-08-30 | `36e5002fe9fc8871f8e01f7f51472d8cb95ea0f0` | `95672436310536bd17d6cd8ec7f26e364a7d8f77` | Fix Franka grasp: open gripper after world.reset(); add instrumentation |
| 2026-08-30 | `f6eee750f212b8367506420761099d868d1b1928` | `d43966d8ca6fb488fe5dad07c5a8b408eeb75ad4` | Add Isaac backend: 7 verbs on Franka with live stage pose resolution |
| 2026-08-31 | `4c2d6a399d376960abb31adba3146000191e2a14` | `3472a845e7146cf4b8354870d22f5aa811c8cf35` | Isaac Sim demo: 7-verb Franka backend, LLM bridge, model/strategy selection, voice input,  |
| 2026-09-01 | `dc876a6c288fb6dd4646553f1a0b3e10f41b8298` | `33b8ec5b4f78174e3b736527c15d5c93bf8792fe` | Add MuJoCo bridge; backend-aware vocabulary in client |
| 2026-09-02 | `f8a8064c117df91116cd3b3444ccc4a2e9316b7f` | `712030c07bc32c274d9ee1770330e8bdb9bd213f` | cleanup: fix run_demo default crash, add entrypoint gate, merge client roster/grammar/voic |
| 2026-09-07 | `1330e06c73d225eae171f9bb44f30709934537bc` | `1fa68667ec9e64e83773a641b9a501a04402e4e7` | audit: V3 vs V4 comparison as four priced prompt-design bugs |
| 2026-09-07 | `deb89f283ac9449b12ad89a1c2a478aaa1405852` | `3298cb183f6ac2d4d8c525f8208149a4c8834521` | demo: Isaac parity with MuJoCo at the LLM layer |
| 2026-09-07 | `e5ac03d2745f7bb0a8cde1aa9aa4eeb632688d73` | `685ac781a1e22a9028361c5764f8d171a819f0f2` | audit: drop duplicate V4 figures; fig5/6/7 supersede them |
| 2026-09-10 | `093c1ee4e4f9388a63adcd0214c0288a684966cd` | `1b03ee24daa9e5665bd0a63245dc542b04576d7d` | V5 final measurement: backfill script, manifests, figures, docs (36 cells, 12152 rows, leg |
| 2026-09-14 | `151d1ded6be5f07583078f044c77962408e628d5` | `f94747e8988094fdab963a2dd38d9080f02b800d` | Demo hardening: strategy allowlist, scene reset, error taxonomy, preflight, offline mode,  |
| 2026-09-14 | `207d66a4e144c9359e802738d3cd08e255dc988c` | `d3ba20de1236346f5cb2eae26f0c10a4e4a4af38` | demo: split execution_failed into distinguishable outcomes |
| 2026-09-14 | `3de0dc8e0bf76f51f69401ca375a3b146a238462` | `de8c896a8ea72a944ce742f9a7b466a76310a6e3` | demo: descended is control_budget, not unreachable; record the evidence |
| 2026-09-14 | `3ef5e25a3d30790e159d1d4d988207017ba4b199` | `d8780a09642fb8caa3bd028ddaaebe7757f43984` | demo: preflight, offline assets, bounded cloud + bridge, startup checks |
| 2026-09-14 | `4074fbc5a2d5fef4069a0e7d8f291ab1111ca02b` | `0205b478504913cf3da1f9b47fc837639fdfd260` | demo: add a scene reset over a control channel, not a verb |
| 2026-09-14 | `4633a224d28cf207abf33d48ce4ebeeb4f214db0` | `8658ff5dbff1d95fd74c9adbaa7e6522281fdf2b` | demo: validator-feedback retry, one re-ask, both attempts printed |
| 2026-09-14 | `6a90d4cbcf33a78c6c305c4740cf1a217860ddb2` | `23fe7e6df1b5bfbcef90c5d76a1ff28d9ae306cc` | demo: the 24-command Task 12 set, with each expectation's basis recorded |
| 2026-09-14 | `9bfec8b9d4b9bc653f98fb0f84a2bbdac31a2089` | `76fb5e027767a9ed147fbdd9a2139d167904a1fd` | demo: unlock all five strategies for the LoRA adapters |
| 2026-09-14 | `e029a3175cfe7ade9ccc7b3994fd2786983a041d` | `0c56849c329ca1691d923b73ebad5d78b7fad419` | demo: retry only on content failures, never on transport |
| 2026-09-22 | `3fd5e516b7e8df9674e43e6917dc42b57e2cef4c` | `5d69ed35b584909353946a234e9a8870d27e14a1` | Add V5 result CSVs and thesis2 analysis (A1-A10, Part B review fixes) |
| 2026-09-22 | `5accc70e4ca1959dbda179fdb4c4c50721cee4df` | `b534363c0b5a98f3fb0475b1cc97ce0bc184d9a3` | Add A11, geometry-gate and legend-check scripts and outputs; S-C/D/E cleanup |
| 2026-09-22 | `fc3959308617df0ecec6f9272c1ed379cb85d56e` | `52543b72a2126d80426a210103dd950642007f0f` | Holm-Bonferroni correction across the declared family; conditioned headline |
| 2026-09-25 | `bb23a2fc76021662b302f9036ac119e56219f882` | `efeec38d9172340ba0000e276853cdef321aeeeb` | Final-run scripts: SALSA D..Q relabel, full v4c and energy lists, split-aware A8/A10/A12,  |
| 2026-09-25 | `e01da1591fcbb2e7ed164e3d715a2fef93d02fb6` | `ec16158a964758217e2c80a3df39b3636ce39d46` | Stage 0-6 scripts, legacy retrain, GRPO (thesis2rl), demo parsers |
| 2026-09-26 | `371699a2f1c4f44a0b6f915cfc86127da2a4e5ae` | `2ace8e534229e7cfcd4583c74a08d46cb0aafc01` | Speculative pred files, RUN_ORDER path fix, 12 V5 cells re-run: 31/31 cells 216/216 vs V5 |
| 2026-09-26 | `8eacc88d38c898c33a99f565d39cdfe29e2519a9` | `9807fd833739481f4282c9958e7532f8af866a2d` | Fix V5-cells loader/retriever, strengthen gates, repro join key, completeness census; A8/A |
| 2026-09-26 | `a9ab2e1cc53aff2c7bbf025a9caf2a66fd4734da` | `adc38a4d0c982a5c96c57fc2cbaf11828042fd6a` | Stage 0 complete: new arms trained, legacy arms reproduced 217/217, 14 arms frozen |
| 2026-09-26 | `bababbd323d127f88325c19fe5be44e140009540` | `790bd9f38e1e9b75aef50753a96c55cdd6e40ccc` | Cloud gating by accuracy, gpt-4o-mini retries with backoff, --retry-errors; v3 cloud cells |
| 2026-09-27 | `52013eff342a304444c2f5722ea0f5ca0f981f80` | `3be4cc3d6a60bb31f09b196f34c8c98187830e42` | Checker: V5 comparisons v3-only; every-row-failed and model_error FAIL on all cells; v4c l |
| 2026-09-27 | `a0f90a749cc8fd658604ed345bfea7f410f2c258` | `d2ab33b1babbb3117cc7fe5fcd9b53e06f29d63b` | Phase A complete: held_out_v3 done, 22 arms reproduced |
| 2026-09-27 | `f11b0fbd47430160fcab8b61c1ba485bfee02ea0` | `b8f350db26699c72085677ce6df3a0dea6d50abd` | Ensemble from V5 cells file, outcome-gated reproducibility (0.9630 on 216, reproduced) |
| 2026-09-28 | `9d60087e45abd76f68386243aaf04620d2936e50` | `3b7c760d3a3cb3907d38ad524a0fbc3eed561805` | Speculative arms: energy through s3_energy with the shared protocol |
| 2026-09-28 | `b104ffa3dfec4471f2ba2b51e61c25b04e1c83e0` | `5817500bd2423c2f364f82b496833dd8a4d7c96b` | Phase B complete: v4c, energy, significance (6/8 primary claimable), figures, report |
| 2026-09-28 | `bd60c64ecac2f752675f22e9b0f72cd7eb57ec49` | `94726328b750d44b7fe60f5564b3aaf4adb4bd7e` | Final run complete: speculative energy, final analysis, 6/8 primary claimable |
| 2026-09-28 | `c864508148238eef517447f3bfad8a9052214518` | `7b408e2ed68cc361622284fd2c4d608077497be1` | Final fix pass: energy frontier GPU-only, fig5 Stage-3 source, VRAM floor, v4c rejection f |
| 2026-09-29 | `3c844ece8b681a091903214300bb94215b9961bc` | `7f0bf3b4aa054ea1894da5bd477c125fd94af7bc` | Interpretation corrected from per-word confusion: classifier C-default, phrasing confound, |
| 2026-10-06 | `2926292a936eddd20509f085169730c7cadb0108` | `4b433931e18f043fa0345c158c5e8c07154df847` | thesis3 unified bench: harness, run scripts, board-power energy |
| 2026-10-08 | `ba9e86dfdc8353fe2ceb6377a6be3754e29848e2` | `1279a7e0ef4844c2557f9d715311df0f20fa2c1d` | verify: gap rule max(5 s, 1% of load window) |
| 2026-11-08 | `1b175cc149516d03b39870d6e8e740c0ffb84501` | `c8d9916ce0e0296e10eef0640953a450439d0604` | README: map thesis figures and tables to their data files |
| 2026-11-08 | `1bfc6a93ed6424aa2d6b8329cdd56eef40892fd6` | `82d1acedfd008cf59deb22ab95c95cabb4432430` | thesis3: update corrupt_test c7 to the current gap rule |
| 2026-11-08 | `23107ee1ba4e723f89d9a428d042b1d2fb4d08f0` | `62b07f91d5061696674ff14be46bc1c1ffbb1b9e` | Repository cleanup: scripts/, archive/, docs/, media/ (see docs/MOVED_PATHS.md) |
| 2026-11-08 | `9d1d5636894793918bf180541befcc541a3e8db0` | `fa521dcdebafc49b527d8d7bb0a8bde0f857c193` | Add MIT licence; README: licences, citation, contact, conda ToS note |
| 2026-11-08 | `dfe8b839213d3d8cb69a84ef67afd4999c0c5d7f` | `4e3df8343654d0c764517a9af95067412abc8469` | Add repo audit of 29 Sep 2026 |
| 2026-11-08 | `e59408773a8c9093acbb52171132fbd49c02bb15` | `8f6db07af08c2fa955b2db23d1aaf925914db98f` | media: name the LoRA model in the Qwen video |
| 2026-11-08 | `f0349c9e851af19f527ed34010bb6158fea8a2b7` | `fe26881fa331b6af386fb9f91107ff169441dc6c` | Add README and environment files |
| 2026-11-09 | `6afa73f894f431a157555a02970a3dd35f7fd098` | `eec17c064c8b935857ff112a378d1a9f84721dce` | Merge cleanup: layout, README, environment files |
| 2026-11-09 | `b170ec8021c230b3392f4e6f974be62bf960a445` | `fefbb885912b4f1ad7d10bdfffb13e6720f45ec3` | README: Table D.1 sources (frozen_checkpoints.csv and release assets) |
| 2026-11-09 | `bb0bc87e885ebc128a4a35eb6fc0884447c5e10d` | `9685c38b58557af8a02b76593c1d4003abcffe7b` | README: release v1.0 asset names, sizes and unpack steps; thesis2/models not released |
