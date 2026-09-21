"""thesis2/scripts/a1_download_probe.py -- actually attempt the two downloads.

A1.2 (TinyBERT) and A1.5 (Flan-T5) need weights that are not in the local cache.
The previous session did NOT attempt them. This does, and reports the exact
error if it fails.

CERTIFICATE VERIFICATION IS NOT DISABLED. No verify=False, no REQUESTS_CA_BUNDLE
override, no truststore shim. If the chain is intercepted, that is reported as a
finding about the network, not worked around.
"""
import os
import sys
import traceback

# make sure any offline flag from a parent shell cannot mask a real network test
for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    if os.environ.pop(k, None) is not None:
        print("  (cleared %s so this is a genuine network attempt)" % k)

TARGETS = [
    ("A1.2 TinyBERT-4", "huawei-noah/TinyBERT_General_4L_312D", "auto"),
    ("A1.5 Flan-T5-base", "google/flan-t5-base", "seq2seq"),
]

print("=" * 78)
print("A1 DOWNLOAD PROBE -- certificate verification left ON")
print("=" * 78)

results = []
for label, repo, kind in TARGETS:
    print()
    print("--- %s : %s ---" % (label, repo))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(repo)
        print("  tokenizer OK (vocab %d)" % len(tok))
        if kind == "seq2seq":
            from transformers import AutoModelForSeq2SeqLM
            m = AutoModelForSeq2SeqLM.from_pretrained(repo)
        else:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(repo)
        n = sum(p.numel() for p in m.parameters())
        print("  weights OK: %.2f M parameters" % (n / 1e6))
        results.append((label, repo, "OK", "%.2f M params" % (n / 1e6)))
        del m
    except Exception as e:
        msg = "%s: %s" % (type(e).__name__, e)
        print("  FAILED")
        print("  " + msg[:900].replace("\n", "\n  "))
        results.append((label, repo, "FAILED", msg[:400]))

print()
print("=" * 78)
for label, repo, status, detail in results:
    print("  %-20s %-44s %s" % (label, repo, status))
print("=" * 78)
