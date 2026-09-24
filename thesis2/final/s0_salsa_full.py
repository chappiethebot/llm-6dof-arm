"""thesis2/final/s0_salsa_full.py -- STAGE 0: SALSA-full, every field by logits.

Single-pass logit classification of EVERY field, producing a complete command.
For each field in order (verb, then target, then location):

  prompt = the V5 prompt for the command
         + a fixed question listing that field's options as LETTERS in SCHEMA
           ORDER (verb A-G, target A-C, location A-N), ending with "Answer:"

  read the logits at the position after "Answer:", restrict to the letter tokens
  of the LEGAL options for that field, argmax, map back to the schema symbol.

Field masks are applied by construction, from the same schema the shared emit()
reads: a verb that requires no target is never asked for one; a verb that
requires no location is never asked for one; containers are offered only when
the predicted verb is pick_place. The object is built by the SHARED emit().

The three passes share the prompt-prefix KV cache: the V5 prompt is processed
once per command and each field question is appended to a crop of that cache.

VOCABULARY. The question text names only field symbols and the letters; the V5
legend is the existing one, unchanged. The startup guard refuses to run if any
held-out row word (back/front/upper/bottom/top/lower) appears in the question
text this script adds. The V5 legend itself is pre-existing and is NOT modified
-- it is excluded from the guard and quoted as-is.

NO DIGITS MAPPING. The digits mapping is withdrawn (FINAL_REPORT B-4) and is not
implemented here.

MODES
  --check          tokenizer + 5 train_v2 commands. Prints the token ids of
                   ' A'..' N' (each must be ONE token) and confirms the model's
                   greedy next token after "Answer:" is a letter token.
                   GATE FAIL if either fails.
  --dev --model M  dev_v2 action match for one model. Selection input only;
                   held-out is never opened by this script.

Run (thesis2gpu):
  python thesis2/final/s0_salsa_full.py --check --model qwen2.5-3b-lora
  python thesis2/final/s0_salsa_full.py --dev  --model qwen2.5-3b-lora
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

import torch                                                            # noqa: E402

LETTERS = [chr(65 + i) for i in range(14)]          # A .. N
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# The four models named in the brief. Keys are the --model values.
MODELS = {
    "qwen2.5-3b-lora": dict(base="Qwen/Qwen2.5-3B-Instruct",
                            adapter="ft/runs/qwen3b_lora/checkpoint-87"),
    "Qwen2.5-3B-Instruct": dict(base="Qwen/Qwen2.5-3B-Instruct", adapter=None),
    # checkpoint-87 is the selected checkpoint for BOTH LoRA models
    # (ft/merge_and_export.py:40 -- "epoch 3, earliest-plateau rule, BOTH models").
    "qwen2.5-0.5b-lora": dict(base="Qwen/Qwen2.5-0.5B-Instruct",
                              adapter="ft/runs/qwen0.5b_lora/checkpoint-87"),
    "Qwen2.5-0.5B-Instruct": dict(base="Qwen/Qwen2.5-0.5B-Instruct", adapter=None),
}

# ------------------------------------------------------------ the questions
# Field option lists come from the SCHEMA via common.SPACE, so they cannot drift
# from emit()'s field masks. Only the carrier sentences are written here.
Q_VERB = ("Which action does the command ask for?\n"
          "Answer with one letter only.\n%s\nAnswer:")
Q_TARGET = ("Which object does the command refer to?\n"
            "Answer with one letter only.\n%s\nAnswer:")
Q_LOCATION = ("Which named place does the command refer to?\n"
              "Answer with one letter only.\n%s\nAnswer:")


def options_block(symbols):
    return "\n".join("%s = %s" % (LETTERS[i], s) for i, s in enumerate(symbols))


def legal_locations(verb):
    """Field mask for the location question, from the schema.

    Containers are offered only for pick_place; grid cells and runtime symbols
    are always legal when the verb requires a location.
    """
    if "location" not in C.VERB_FIELDS[verb]:
        return []
    if verb == "pick_place":
        return list(C.LOCATIONS)
    return [l for l in C.LOCATIONS if not l.startswith("box_")]


class SalsaFull:
    """One model; three logit reads per command; one shared prefix cache."""

    def __init__(self, key):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        spec = MODELS[key]
        self.key = key
        self.tok = AutoTokenizer.from_pretrained(spec["base"])
        self.model = AutoModelForCausalLM.from_pretrained(
            spec["base"], dtype=torch.float16).to("cuda")
        if spec["adapter"]:
            from peft import PeftModel
            p = C.REPO / spec["adapter"]
            if not p.exists():
                raise SystemExit("GATE FAIL: adapter %s not found for model %r" % (p, key))
            self.model = PeftModel.from_pretrained(self.model, str(p)).merge_and_unload()
        self.model.eval()
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok.encode(" " + L, add_special_tokens=False)
            if len(ids) != 1:
                raise SystemExit("GATE FAIL: ' %s' is %d tokens for %s; the letter "
                                 "mapping requires one token per option." % (L, len(ids), key))
            self.letter_ids.append(ids[0])
        if len(set(self.letter_ids)) != len(self.letter_ids):
            raise SystemExit("GATE FAIL: letter token ids collide for %s" % key)
        from bench.run_cell import apply_legend
        from llm import prompts as P
        apply_legend("v5")
        self.P = P
        self.schema = P.load_schema(str(C.SCHEMA_FILE))

    def chat(self, user):
        return ("<|im_start|>system\n%s<|im_end|>\n"
                "<|im_start|>user\n%s<|im_end|>\n"
                "<|im_start|>assistant\n" % (SYSTEM, user))

    def prefix_text(self, command):
        """The V5 prompt for this command -- the shared prefix for all 3 passes."""
        return self.chat(self.P.build_prompt(command, "zero_shot", self.schema))

    @torch.no_grad()
    def _ask(self, cache, prefix_len, question, symbols):
        """Append one question to a crop of the prefix cache and read letters."""
        n = cache.get_seq_length() - prefix_len
        if n > 0:
            cache.crop(prefix_len)
        qids = self.tok(question, add_special_tokens=False)["input_ids"]
        out = self.model(input_ids=torch.tensor([qids], device="cuda"),
                         past_key_values=cache, use_cache=True)
        logits = out.logits[0, -1]
        ids = [self.letter_ids[i] for i in range(len(symbols))]
        sel = logits[ids].float()
        probs = torch.softmax(sel, dim=-1)
        k = int(probs.argmax())
        return symbols[k], float(probs[k]), [round(float(x), 6) for x in probs], logits

    @torch.no_grad()
    def predict(self, command, want_probs=False):
        """verb -> target -> location, masks applied, shared emit(). Returns
        (obj, info)."""
        t0 = time.perf_counter()
        pids = self.tok(self.prefix_text(command), add_special_tokens=False)["input_ids"]
        out = self.model(input_ids=torch.tensor([pids], device="cuda"), use_cache=True)
        cache = out.past_key_values
        plen = cache.get_seq_length()

        verb, p_verb, probs_verb, _ = self._ask(
            cache, plen, Q_VERB % options_block(C.VERBS), C.VERBS)
        need = C.VERB_FIELDS[verb]

        t_idx, p_target, probs_target = 0, "", []
        if "target" in need:
            tgt, p_target, probs_target, _ = self._ask(
                cache, plen, Q_TARGET % options_block(C.TARGETS), C.TARGETS)
            t_idx = C.TARGETS.index(tgt)

        l_idx, p_loc, probs_loc = 0, "", []
        legal = legal_locations(verb)
        if legal:
            loc, p_loc, probs_loc, _ = self._ask(
                cache, plen, Q_LOCATION % options_block(legal), legal)
            l_idx = C.LOCATIONS.index(loc)

        obj = C.emit(C.VERBS.index(verb), t_idx, l_idx, C.SPACE)
        ms = (time.perf_counter() - t0) * 1000
        info = dict(latency_ms=ms, p_verb=p_verb, p_target=p_target, p_location=p_loc,
                    n_location_options=len(legal))
        if want_probs:
            info.update(probs_verb=probs_verb, probs_target=probs_target,
                        probs_location=probs_loc, location_options=legal)
        return obj, info


def mode_check(key):
    """Tokenizer + 5 train_v2 commands. This is the only mode the author runs."""
    from transformers import AutoTokenizer
    print()
    print("1. LETTER TOKENS ' A' .. ' N' for %s" % MODELS[key]["base"])
    tok = AutoTokenizer.from_pretrained(MODELS[key]["base"])
    ok_tok = True
    ids = []
    for L in LETTERS:
        e = tok.encode(" " + L, add_special_tokens=False)
        ids.append(e)
        flag = "OK" if len(e) == 1 else "*** %d TOKENS ***" % len(e)
        print("     %-4r -> %-10s %s" % (" " + L, e, flag))
        ok_tok &= (len(e) == 1)
    flat = [e[0] for e in ids if len(e) == 1]
    ok_tok &= (len(set(flat)) == len(flat))
    print("   all one token: %s ; ids unique: %s" % (ok_tok, len(set(flat)) == len(flat)))

    print()
    print("2. GREEDY NEXT TOKEN AFTER 'Answer:' on 5 %s commands" % C.TRAIN)
    sf = SalsaFull(key)
    rows = C.load_split(C.TRAIN)[:5]
    ok_greedy = True
    for r in rows:
        pids = sf.tok(sf.prefix_text(r["command"]), add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = sf.model(input_ids=torch.tensor([pids], device="cuda"), use_cache=True)
            cache = out.past_key_values
            plen = cache.get_seq_length()
            qids = sf.tok(Q_VERB % options_block(C.VERBS), add_special_tokens=False)["input_ids"]
            o2 = sf.model(input_ids=torch.tensor([qids], device="cuda"),
                          past_key_values=cache, use_cache=True)
        nxt = int(o2.logits[0, -1].argmax())
        is_letter = nxt in sf.letter_ids
        ok_greedy &= is_letter
        print("     %-52s next=%-6s %-8r %s"
              % (r["command"][:52], nxt, sf.tok.decode([nxt]),
                 "LETTER" if is_letter else "*** NOT A LETTER ***"))
    C.gate(ok_tok and ok_greedy, [
        "letter tokens one-per-option and unique: %s" % ok_tok,
        "greedy next token after 'Answer:' is a letter on 5/5 train_v2 commands: %s" % ok_greedy,
        "model: %s%s" % (MODELS[key]["base"], " + " + MODELS[key]["adapter"]
                         if MODELS[key]["adapter"] else ""),
        "this gate uses train_v2 only; no held-out file was opened",
    ])


def mode_dev(key):
    sf = SalsaFull(key)
    rows = C.load_split(C.DEV)
    hits, recs = 0, []
    t0 = time.time()
    for i, r in enumerate(rows):
        obj, info = sf.predict(r["command"])
        ok = bool(C.score(obj, C.truth_of(r)).strict_match)
        hits += ok
        recs.append(dict(command_id=i, command=r["command"], pred=json.dumps(obj),
                         truth=json.dumps(C.truth_of(r)), correct=ok,
                         latency_ms=round(info["latency_ms"], 2)))
        if (i + 1) % 20 == 0:
            print("     %3d/%d  (%.0fs)" % (i + 1, len(rows), time.time() - t0))
    acc = hits / len(rows)
    p = C.OUT / ("selection_salsa_full_%s.json" % key)
    p.write_text(json.dumps(dict(arm="salsa_full_%s" % key, model=MODELS[key],
                                 dev_action_match=round(acc, 4), dev_hits=hits,
                                 dev_total=len(rows), per_command=recs), indent=2),
                 encoding="utf-8")
    C.gate(True, [
        "dev_v2 action match for salsa_full_%s: %d/%d = %.4f" % (key, hits, len(rows), acc),
        "selection record: %s" % p.relative_to(C.REPO),
        "no held-out file was opened",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="tokenizer + 5 train_v2 commands")
    g.add_argument("--dev", action="store_true", help="dev_v2 action match")
    ap.add_argument("--model", default="qwen2.5-3b-lora", choices=sorted(MODELS))
    a = ap.parse_args()
    C.start_log("s0_salsa_full_%s_%s" % ("check" if a.check else "dev", a.model))
    C.env_banner()
    print("  model: %s" % a.model)
    # Guard only the text THIS script adds; the V5 legend is pre-existing.
    C.assert_no_heldout_vocab(Q_VERB, Q_TARGET, Q_LOCATION,
                              options_block(C.VERBS), options_block(C.TARGETS),
                              options_block(C.LOCATIONS),
                              where="the SALSA-full question text")
    if a.check:
        mode_check(a.model)
    else:
        mode_dev(a.model)


if __name__ == "__main__":
    main()
