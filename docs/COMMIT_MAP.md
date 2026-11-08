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
| 2026-08-27 | `0ff113eb6ccf76cf96aaf74963388386bed6e5e3` | `7893135a24f59a8bd13485e3142fc62e92cf75a8` | figures: plain-language labels, per-figure captions, corrected accuracy wording |
| 2026-08-27 | `4d8266b5ed3b15fb524c9ad57eb2ea1d1cf4b5f1` | `54ead9c67d74dc0c85afe7f3f1b3bbc26db8f0a4` | audit: location/verb/target confusion matrices and v3 result figures |
| 2026-08-27 | `55dba3a82e84e103c03b703fae51b3162a03a41c` | `1e0645c23cf93c39b8f8b9739b54660e8aad9f87` | demo: roster from all_clients(), wire few-shot and rag examples, defer sim imports |
| 2026-08-27 | `7b71bcd0da90657db8c8b829590d27e331f28daa` | `6f5a29200a2da042f2b273fc35f1efc83de7c3ce` | audit: demonstrate the LoRA row collapse is an out-of-vocabulary failure |
| 2026-08-27 | `b8f13859a84c04b96a6073dc01f43051352ff831` | `200161ebbe41ea15489627c510ebd6a7760eb3a5` | audit: implementation and validity audit of the v3 measurement |
| 2026-08-27 | `dba5ae9548655ece5e8f7cd804dd1a79011c6287` | `e5842dadc0765ebbcdc915140e2ca9e3d9f5f98e` | chore: pin dependencies and lock the audit-time environment |
| 2026-08-27 | `f9e2c39fae8d2b4010bd6a4c8211ed5f5ec22ed3` | `80d0d51a0f5633663e11df8be38716552f997bbc` | audit: explain the LoRA row-B collapse, surface parse failures, fix reproduction test |
| 2026-08-28 | `01aaa49ba9c531618237f5fb386d7e4086457f72` | `dfb607de39e907ece80933efadc0fa955ee2adf4` | run_grid: record the legend version in the run manifest |
| 2026-08-28 | `1334f83e634e9c94fe5e35703b05916722a137b9` | `54d438329f9ecf3403ee03913a9addc9252b0cfd` | prereg amendment: V4a-lite condition, written before running it |
| 2026-08-28 | `1b4729f72a42ef1e83b60b5f1e78c82b0903bde5` | `12b5a6ffca3d35b5066250fbc826a9c861e8c97c` | discard contaminated V4a local run; two concurrent writers on one output file |
| 2026-08-28 | `2f103ae4a623e0c21517361a967333608b131cd3` | `ae37dccc4a8a61cfb56996563fcb37b3fb8d13a0` | audit: exhaustive vocabulary-conflict scan of the command datasets |
| 2026-08-28 | `371426db391e607d58805fbc8ee13ba4a8373c6a` | `27466bcd5d3ceb959d0970983095a7d6a8157a8c` | untrack the 9.3MB discarded CSV; keep its README and manifest |
| 2026-08-28 | `48b838a8c419702af4459624d255aa5b1a28760e` | `2c8c0cecbe8947099b279b6ec1241ea08044f508` | prereg: V4 legend and vocabulary conditions, written before any V4 measurement |
| 2026-08-28 | `d421e5d6f7d8edd04acc86ff907ae85c8dd030b8` | `25903df045a1cb6cb25b6489bd097829a7e89fae` | track the V4 condition datasets (v4b, v4c) |
| 2026-08-29 | `44b82abbefdbec7940bb7befcc8e7d3d599e6421` | `91723adaca8737bc345d710c177cab9a0f1cfcec` | V4 complete: all conditions run, rescored, analysed, figures |
| 2026-08-29 | `adb728515d2e82d96b3172dabdf8852dec416181` | `5b67d74365ddd3cebf891c47573b01a11ffd68a7` | Add headless Isaac Sim 6.0 grid scene with front camera |
| 2026-08-29 | `af721e20c86297acb639a4c474ffcbcfde3859fb` | `127a32d4dec1a93341f91f95908abd0782fd4665` | Ignore out/; remove crashing windowed Isaac scene |
| 2026-08-29 | `b47404010f279ab068efeb426e8bfe5a1ec1c1f3` | `3468631f16c73292f220291b73b66b440cbff6ee` | audit: DEFENCE_NUMBERS.md - one page of quotable numbers |
| 2026-08-29 | `bc57fb89423e8c157a1012dc6677ad1201e0ec02` | `482214ad9aeebb368b321897f583f2ff67c2b81f` | demo: V5 legend + constrained decoding, wired as the default demo config |
| 2026-08-29 | `f27795d27c73702ed48356cd295f0f249d5e3a49` | `2edf3af258666c232120cc7f25a0c38e55345f29` | step 9: classifier margins + offline escalation simulation |
| 2026-08-30 | `36e5002fe9fc8871f8e01f7f51472d8cb95ea0f0` | `509ef98f73121bfbff2e7a06fac246e313396bc1` | Fix Franka grasp: open gripper after world.reset(); add instrumentation |
| 2026-08-30 | `f6eee750f212b8367506420761099d868d1b1928` | `ad5845995271a1d5c0452055eaf96407e5d7ad5a` | Add Isaac backend: 7 verbs on Franka with live stage pose resolution |
| 2026-08-31 | `4c2d6a399d376960abb31adba3146000191e2a14` | `8b63eea0fe876a102eb001a99f39db82d9e8a630` | Isaac Sim demo: 7-verb Franka backend, LLM bridge, model/strategy selection, voice input,  |
| 2026-09-01 | `dc876a6c288fb6dd4646553f1a0b3e10f41b8298` | `b718c59110664c363c6a4b51d1106dcbcc02f6c1` | Add MuJoCo bridge; backend-aware vocabulary in client |
| 2026-09-02 | `f8a8064c117df91116cd3b3444ccc4a2e9316b7f` | `485bca47a06205de1ca442f2fea3613691dcdefe` | cleanup: fix run_demo default crash, add entrypoint gate, merge client roster/grammar/voic |
| 2026-09-07 | `1330e06c73d225eae171f9bb44f30709934537bc` | `b36180562484fb10af6d1e46a8c3f9f8a98c5c68` | audit: V3 vs V4 comparison as four priced prompt-design bugs |
| 2026-09-07 | `deb89f283ac9449b12ad89a1c2a478aaa1405852` | `7a42b93b2b328ea42c92a557396ed279e9e3cd68` | demo: Isaac parity with MuJoCo at the LLM layer |
| 2026-09-07 | `e5ac03d2745f7bb0a8cde1aa9aa4eeb632688d73` | `004edb6303e6b8e4515ca203e5c98ae824383f7f` | audit: drop duplicate V4 figures; fig5/6/7 supersede them |
| 2026-09-10 | `093c1ee4e4f9388a63adcd0214c0288a684966cd` | `2b62c3fd7de1fc780160d8b7d3758e805de70fe1` | V5 final measurement: backfill script, manifests, figures, docs (36 cells, 12152 rows, leg |
| 2026-09-14 | `151d1ded6be5f07583078f044c77962408e628d5` | `3dc2545051a5ee65de99519c1df5b1e8ca348e18` | Demo hardening: strategy allowlist, scene reset, error taxonomy, preflight, offline mode,  |
| 2026-09-14 | `207d66a4e144c9359e802738d3cd08e255dc988c` | `849364e1fb4150f67f3d72d90ee75919125c7e2f` | demo: split execution_failed into distinguishable outcomes |
| 2026-09-14 | `3de0dc8e0bf76f51f69401ca375a3b146a238462` | `d948bd61e20352e59e52e1aaef9c64b1c80f49b0` | demo: descended is control_budget, not unreachable; record the evidence |
| 2026-09-14 | `3ef5e25a3d30790e159d1d4d988207017ba4b199` | `160b491249f3d3a9fce15e0e73adcc77e26f00e4` | demo: preflight, offline assets, bounded cloud + bridge, startup checks |
| 2026-09-14 | `4074fbc5a2d5fef4069a0e7d8f291ab1111ca02b` | `3da636568cc1536920b8520fe7554da5f0f7704e` | demo: add a scene reset over a control channel, not a verb |
| 2026-09-14 | `4633a224d28cf207abf33d48ce4ebeeb4f214db0` | `fa4ddfe6116d07aab5d92cf24a99b900e1286199` | demo: validator-feedback retry, one re-ask, both attempts printed |
| 2026-09-14 | `6a90d4cbcf33a78c6c305c4740cf1a217860ddb2` | `87e8bd8c1fbd0f033a4a65ec4cfef7afaa1857c6` | demo: the 24-command Task 12 set, with each expectation's basis recorded |
| 2026-09-14 | `9bfec8b9d4b9bc653f98fb0f84a2bbdac31a2089` | `fc2e1eaeb9c5e07c6a86aa7578bf5df5d6f83dbd` | demo: unlock all five strategies for the LoRA adapters |
| 2026-09-14 | `e029a3175cfe7ade9ccc7b3994fd2786983a041d` | `263563131f02d72080d88084d4dbec505ab7f86b` | demo: retry only on content failures, never on transport |
| 2026-09-22 | `3fd5e516b7e8df9674e43e6917dc42b57e2cef4c` | `7e809c6142c0952d2a7d3346d286bfa9b2dee1c5` | Add V5 result CSVs and thesis2 analysis (A1-A10, Part B review fixes) |
| 2026-09-22 | `5accc70e4ca1959dbda179fdb4c4c50721cee4df` | `c7b41ee94421ea815158f2d4b49226be70256764` | Add A11, geometry-gate and legend-check scripts and outputs; S-C/D/E cleanup |
| 2026-09-22 | `fc3959308617df0ecec6f9272c1ed379cb85d56e` | `044b364ce6f4057cdc0344de8b94952618774ff9` | Holm-Bonferroni correction across the declared family; conditioned headline |
| 2026-09-25 | `bb23a2fc76021662b302f9036ac119e56219f882` | `8293a1151b251bba1a396299ad44175f6f18e28a` | Final-run scripts: SALSA D..Q relabel, full v4c and energy lists, split-aware A8/A10/A12,  |
| 2026-09-25 | `e01da1591fcbb2e7ed164e3d715a2fef93d02fb6` | `abe8f0afbc2f20b816788f60be86543932ca30d8` | Stage 0-6 scripts, legacy retrain, GRPO (thesis2rl), demo parsers |
| 2026-09-26 | `371699a2f1c4f44a0b6f915cfc86127da2a4e5ae` | `cd868417a099fd9a6d2a9c3228c988fe8239de47` | Speculative pred files, RUN_ORDER path fix, 12 V5 cells re-run: 31/31 cells 216/216 vs V5 |
| 2026-09-26 | `8eacc88d38c898c33a99f565d39cdfe29e2519a9` | `e200a8eed258d29c951024193af9db9d24c5f3c3` | Fix V5-cells loader/retriever, strengthen gates, repro join key, completeness census; A8/A |
| 2026-09-26 | `a9ab2e1cc53aff2c7bbf025a9caf2a66fd4734da` | `ae985653fccc8097ef78afc6c590014c203c5356` | Stage 0 complete: new arms trained, legacy arms reproduced 217/217, 14 arms frozen |
| 2026-09-26 | `bababbd323d127f88325c19fe5be44e140009540` | `cec743fc468f298fab9ff45be9bb18b3f5b900ce` | Cloud gating by accuracy, gpt-4o-mini retries with backoff, --retry-errors; v3 cloud cells |
| 2026-09-27 | `52013eff342a304444c2f5722ea0f5ca0f981f80` | `74c138bfbd8171bc796972a5a7d65dd9191aed45` | Checker: V5 comparisons v3-only; every-row-failed and model_error FAIL on all cells; v4c l |
| 2026-09-27 | `a0f90a749cc8fd658604ed345bfea7f410f2c258` | `49bf2a1b2940b47365c6d4883db8469d4f0a760a` | Phase A complete: held_out_v3 done, 22 arms reproduced |
| 2026-09-27 | `f11b0fbd47430160fcab8b61c1ba485bfee02ea0` | `f988986a8e32a6071d324253b3bd7e60f2a17673` | Ensemble from V5 cells file, outcome-gated reproducibility (0.9630 on 216, reproduced) |
| 2026-09-28 | `9d60087e45abd76f68386243aaf04620d2936e50` | `7afb747b72f71e65e9f94aa2b1fe70511732b843` | Speculative arms: energy through s3_energy with the shared protocol |
| 2026-09-28 | `b104ffa3dfec4471f2ba2b51e61c25b04e1c83e0` | `bdabb48eb4102d92350e9b35859d462559a7b228` | Phase B complete: v4c, energy, significance (6/8 primary claimable), figures, report |
| 2026-09-28 | `bd60c64ecac2f752675f22e9b0f72cd7eb57ec49` | `5ec640caf5faa6f664d4d5d1b88b953b6cf72090` | Final run complete: speculative energy, final analysis, 6/8 primary claimable |
| 2026-09-28 | `c864508148238eef517447f3bfad8a9052214518` | `7d8f5bee997d0601072df8b44c5ca9eed859f074` | Final fix pass: energy frontier GPU-only, fig5 Stage-3 source, VRAM floor, v4c rejection f |
| 2026-09-29 | `3c844ece8b681a091903214300bb94215b9961bc` | `89ec7c774f975530adb98600d6df5b90d78a5a55` | Interpretation corrected from per-word confusion: classifier C-default, phrasing confound, |
| 2026-10-06 | `2926292a936eddd20509f085169730c7cadb0108` | `7f5de54cf04defeb85d55e424de8569b701d8e14` | thesis3 unified bench: harness, run scripts, board-power energy |
| 2026-10-08 | `ba9e86dfdc8353fe2ceb6377a6be3754e29848e2` | `7cc3932561364341f514267bc9ad1b21cff1d944` | verify: gap rule max(5 s, 1% of load window) |
| 2026-11-08 | `1b175cc149516d03b39870d6e8e740c0ffb84501` | `274f209a6396ff54f2c23d905ce29854ee84999f` | README: map thesis figures and tables to their data files |
| 2026-11-08 | `1bfc6a93ed6424aa2d6b8329cdd56eef40892fd6` | `3c34ed4bdd2df061d27529e9bffe9ee0154bcdd4` | thesis3: update corrupt_test c7 to the current gap rule |
| 2026-11-08 | `23107ee1ba4e723f89d9a428d042b1d2fb4d08f0` | `3d2a389fc20f807e77fab24292b5fded99f173e3` | Repository cleanup: scripts/, archive/, docs/, media/ (see docs/MOVED_PATHS.md) |
| 2026-11-08 | `9d1d5636894793918bf180541befcc541a3e8db0` | `8960a08ab88c167e91e64f86c147cd0fcfd4c936` | Add MIT licence; README: licences, citation, contact, conda ToS note |
| 2026-11-08 | `dfe8b839213d3d8cb69a84ef67afd4999c0c5d7f` | `63bfe8c0e3420ecac8320de4c5906c50fda62469` | Add repo audit of 29 Sep 2026 |
| 2026-11-08 | `e59408773a8c9093acbb52171132fbd49c02bb15` | `9d943a75d00ee07c4bedc8c2a2ce17a573d2b026` | media: name the LoRA model in the Qwen video |
| 2026-11-08 | `f0349c9e851af19f527ed34010bb6158fea8a2b7` | `a72496642119bf19c48ba0185c379a76bc984ffa` | Add README and environment files |
| 2026-11-09 | `6afa73f894f431a157555a02970a3dd35f7fd098` | `28c8b53c5428c3d988ed890ae5122ed7a43a242a` | Merge cleanup: layout, README, environment files |
| 2026-11-09 | `b170ec8021c230b3392f4e6f974be62bf960a445` | `c9a0d09aba61b2b6e8ebbce4fee89c47ed8ca53f` | README: Table D.1 sources (frozen_checkpoints.csv and release assets) |
| 2026-11-09 | `bb0bc87e885ebc128a4a35eb6fc0884447c5e10d` | `f54a3227d142b017b18fafcafab6a91b094a8130` | README: release v1.0 asset names, sizes and unpack steps; thesis2/models not released |
