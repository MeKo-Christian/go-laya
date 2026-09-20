# Invariants the Go port must reproduce

> Derived from `original/laya/` (upstream Python `laya` 0.3.4). Every item cites the line that defines
> it. Referenced from `PLAN.md` §5. These are assertions, not guidance: each one is a test.

Numbered checklist. Each item cites the line that defines it.

**`build_sequence` — token budget and layout (common.py:49-86)**

1. The literal mask-token _string_ is replaced by a single space `" "` in all three text sources before tokenisation: instructions (L62), every option text (L68), and the serialized state (L83). Nothing else is stripped.
2. Head text is exactly `"<qtype> question: <instructions>"` — e.g. `"choice question: Which team?"` — tokenised with `add_special_tokens=False` (L63).
3. Option _i_'s ids are `[mask_token_id] + encode(" " + optionText)[:48]`, so at most **49** ids per option. The 48 cap applies to the option text only, not to the mask token, and the leading space is part of the tokenised text (L66-69).
4. Option order is `option_order` when supplied, else `0..n-1` (L61). The public API never supplies it (agent.py:261), but keep the parameter — the option-order-robustness benchmark uses it.
5. `opt_budget = head_max_len − Σ len(opt_ids)` (L70). Despite the name, this is the budget left for the **head**, and it can go negative.
6. If `opt_budget < 16`: `per = max(4, (head_max_len − 16) // max(1, nOptions))`, every option truncated to `per` ids **including its mask token**, then `opt_budget` recomputed (L71-74). This runs **at most once** and may leave `opt_budget` still below 16 (77 options at `head_max_len=192` → `per=4`, sum 308, `opt_budget = −116`).
7. `head_ids = head_ids[:max(8, opt_budget)]` — the head always keeps at least 8 ids; a negative budget clamps to 8 (L75).
8. Layout is exactly `[CLS] + head_ids + [SEP] + opt0 + opt1 + … + [SEP] + state_ids + [SEP]` (L76-85).
9. `markers[i]` is the index of option _i_'s **first** id (its mask token), recorded before the option is appended (L78-80), so `markers[0] == 1 + len(head_ids) + 1`.
10. `room = max(0, max_len − len(ids) − 1)` is computed after the options' `[SEP]` and reserves exactly one slot for the trailing `[SEP]` (L82).
11. State ids are truncated **right** by default (`st[:room]`) and **left** only when `truncate_left=True` (L84). The public API always uses right truncation (agent.py:261).
12. Return is `ids[:max_len]` and `[m for m in markers if m < max_len]` (L86) — the final `[SEP]` can be truncated away.
13. A question whose surviving marker count differs from its rendered option count is rejected with `ValueError("question %r options exceed head_max_len=%d")` (agent.py:262-263). This is the only user-visible validation error in the inference path.

**`serialize_state` / `render_options` / `render_criterion` (common.py:15-46)**

14. `choice`: an option renders as the bare key **iff** its value is `None` or `""` — exactly those two. `0`, `False`, `0.0` are legitimate values and render as `"zero: 0"` / `"no: false"` (L38, test_criteria.py:65-67). Otherwise `"<key>: <rendered>"`. Order is criteria definition order.
15. `score`: `"level %d: %s"` with `i` starting at 0 (L40).
16. `noul`: always exactly two options, index **0 = false**, index **1 = true** (L43-46). Defaults: `"false: no, the statement does not hold"` and `"true: yes, the statement holds"`. A `"true"`/`"false"` value of `None` or `""` falls back to the default; any other value is rendered.
17. `render_criterion`: a `str` passes through byte-identical; anything else is `json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)`. Golden cases from test_criteria.py:33-41: `{"desc": "phishing"}` → `'{"desc": "phishing"}'`, `["a","b"]` → `'["a", "b"]'`, `3` → `"3"`, `False` → `"false"`, `None` → `"null"`, `{"d":"münchen"}` → `'{"d": "münchen"}'` (non-ASCII literal, **not** `ü`). Unserialisable values must not raise — fall back to a string form.
18. `serialize_state`: a `str` passes through; anything else is `json.dumps(state, ensure_ascii=False)` with **default** separators, which are the same `", "` / `": "`. Key order is insertion order; no HTML escaping of `<`, `>`, `&`. **Go's `encoding/json` violates all three of: separators (it emits `{"a":1}`), key ordering (it sorts map keys), and HTML escaping (on by default). A custom encoder is mandatory, or the model sees different bytes than the Python version.**

**`Agent._to_internal` + `Agent.system_one` — numerics and formatting (agent.py:229-343)**

19. A `choice` question whose `criteria` is a list becomes `{c: None for c in list}` → bare labels in list order (L233-234).
20. Non-string `instructions` become `json.dumps(ins)` with **default `ensure_ascii=True`** — the one place laya escapes non-ASCII (L236-237).
21. `max_len` and `head_max_len` come from cfg with defaults **512** and **192** (L256-257), and are documented as runtime-mutable (README:368).
22. `temp_bucket(qtype, k) = "<name>:<size>"` where size is `"2"` for k≤2, `"3-5"` for k≤5, `"6-10"` for k≤10, else `"11+"` (common.py:209-211). Examples: `"noul:2"`, `"choice:3-5"`, `"score:3-5"`.
23. `t_scale = temperature_by_options[temp_bucket] if present else temperature[qtype]`, with defaults `temperature=[1.0,1.0,1.0]` and `temperature_by_options={}` (L194-195, 304). `laya-multilingual` ships with **no** fitted temperatures (README:355).
24. `z = logits[r, :k] / max(1e-3, t_scale)` then `p = exp(z − max(z)); p /= sum(p)` — a max-subtracted softmax over exactly the first _k_ logits (L305-307). The `1e-3` floor guards a zero/negative temperature.
    - **24a. Precision** _(added 2026-09-20)_: `logits` arrives as numpy **float32** (L294) and stays float32 through the division by a Python float (NumPy 2 weak-scalar promotion), `np.exp`, and the normalisation; `confidence_from_probs` (#25) is float32 arithmetic too. `score` (#28) is `float((np.arange(k) * p).sum())` — float64 over a float32 `p`; `noul` (#26/#27) is `float(p[1])` from float32. The act softmax (#32) is torch float32. Go must do the softmax and entropy in `float32` where numpy does, or the fourth decimal after `round` will disagree. `scripts/dump_python_parity.py:747-751` mirrors this.
25. `choice` and `score` confidence = `round(clip(1 − H(p)/ln k, 0, 1), 4)` with `H = −Σ p·ln(clip(p, 1e-12, 1))`; `k < 2` returns `1.0` (common.py:200-206, agent.py:309).
26. `noul` confidence = `round(max(p[1], 1 − p[1]), 4)` — **not** the entropy formula (L335). For k=2 the two happen to disagree, so this must be a separate code path.
27. `noul` value = `round(p[1], 4)`: **index 1 is P(true)** (L334).
28. `score` value = `round(Σ i·p[i], 4)` for i in 0..k−1 — an expectation over levels, not an argmax (L322).
29. Every emitted float is `round(x, 4)`. **Python's `round()` is round-half-to-even on the exact binary value of the double; Go's `math.Round` is half-away-from-zero.** Use `strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)`, which rounds the same way, and test the tie cases (e.g. `0.00125`, `2.5e-5`).
30. Key sets per answer type are exactly as in `docs/API.md`'s "Return-value shapes" — in particular a `noul` answer has **no** `probabilities` and no `legend`, and a `choice` answer has no `legend`. Every key of a shape is emitted even when its value is empty or `""` (no `omitempty`).
    - **30a. Tie-break** _(added 2026-09-20)_: `keys[int(p.argmax())]` (L315) — numpy returns the **first** maximum, so on an exact tie the earlier criterion wins.
31. `score`'s `legend` maps `str(i)` to the **raw** criterion value, not the rendered option text (L326). A dict criterion stays a dict in the legend.
32. `action.act_probability = round(softmax(act_logits, -1)[r, 0], 4)` — **column 0** of the action head (L295, 310).
33. Envelope: `model` is the constant `"laya-rl-agent"`; `usage.input_tokens` is `int(attention_mask.sum())` over the **whole batch**, not per question; `usage.output_tokens` is always `0` (L339-343).
34. `predict` is an alias of `system_one` (L345) and `RLAgent` an alias of `Agent` (L348).

**Model forward, if the head is reimplemented rather than exported (common.py:105-126)**

35. The type embedding is added to **every** position, not just `[CLS]`: `h = h + type_emb(qtype)[:, None, :]` (L109).
36. The head layers are run manually (`for layer in self.head.layers`, L112-113), which **bypasses** `nn.TransformerEncoder`'s optional final norm. Layers are `norm_first=True`, `nhead = max(1, d//64)`, `dim_feedforward = 4d` (L96-98).
37. Masked positions get `-1e4`, not `-inf` (L117).
38. `act_head` features are `[top1, top1−top2, normalised entropy, k/255.0]` where the entropy denominator is `ln(clamp(markerCount, min=2))` and the probabilities are **detached** (L119-123). Pooled input is `h[:, 0]` after the head layers (L124).

**`Router.route` (router.py:241-290)**

39. Precedence, strictly: explicit `model` > explicit `task` > detected workflow (**only** when `auto_task_detection`) > explicit `lang` > detected script/language > `default` (L254-290, tests 122-151).
40. `normalise_name`: `str(name).strip().lower()`, then the alias map, then membership in `DEFAULT_MODELS`; unknown raises (L100-106). `"ML"` → `"multilingual"`, `"English"` → `"english"` (test_router.py:104-108).
41. The `task` branch maps `task.lower().replace("-","_") == "typed_decisions"` to `"typed-decisions"`; any other task string goes through `normalise_name` (L260).
42. Workflow matching requires **exact set equality** of question ids against one of the four signatures — a superset or a partial overlap matches nothing (L109-119, tests 98-100).
43. The `lang` branch: `lang.lower().split("-")[0] in ("en","eng","english")` → `english`, everything else → `multilingual` (L271). So `"en-GB"` → english, `"de"` → multilingual.
44. Detection-branch reasons, verbatim (L276-288):

- `script == "unknown"` → `default`, `"no letters detected in state; using default (%s)"`
- `script != "latin"` → `multilingual`, `"non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"` with `100 * non_latin_fraction`
- latin but not English → `multilingual`, `"Latin script but language looks like %r, not English"` — `%r` yields Python repr quoting, i.e. `'de'` with single quotes
- otherwise → `english`, `"English Latin text"`

45. All five `RouteDecision` keys are always present. `detection` is the `analyse()` result **only** on the detection path and `None` on the model/task/workflow/lang paths; `workflow` is `None` on the model/task paths but carries the detected workflow name on the **lang and detection** paths even though it did not drive the decision (L256-290).
46. **Deviation to decide:** the auto-workflow branch sets `repo=self.models["typed-decisions"]` — the raw `(repo, subfolder)` tuple — while every other branch uses `_repo_str()` and yields a string (L266 vs L256/261/272/289). Through `json.dumps` this surfaces as a JSON array instead of a string. **Recommendation: always emit the string, and note the deviation in the Go docs.**
47. `_repo_str`: `"repo/sub"` when a subfolder is set, else `"repo"`; a plain string spec passes through unchanged, so a local-path override like `/tmp/ml` survives (L51-54, test_router.py:216-231).

**Router lifecycle (router.py:144-238)**

48. `max_loaded = max(1, int(max_loaded))` (L160).
49. `_order` is least-recently-used **first**. `load` appends on a miss and touches on a hit; `_evict` pops from the front while `len(_order) > max_loaded`, then reconciles `_agents` against `_order` (L169-195, tests 192-209).
50. `attach` registers the agent, touches it, and raises `max_loaded` to `len(self._agents)` so the LRU cannot immediately evict it (L197-208, tests 259-270). It accepts aliases.
51. `preload` raises `max_loaded` to `max(max_loaded, len(names), len(_agents))` before building, and skips names that are already resident (L218-222).
52. `unload()` with no argument clears everything; with a name it removes just that one (L225-234).
53. `Router.predict` returns the `system_one` payload with `result["routing"] = dict(decision)` added (L305-308); `system_one` is an alias of `predict` (L311).

**`lang` (lang.py)**

54. Latin is `cp < 0x0250` **or** `0x1E00 ≤ cp ≤ 0x1EFF`; only characters where `str.isalpha()` is true are counted at all (L96-101, 118-121). Go: `unicode.IsLetter`.
55. `_SCRIPT_RANGES` order is load-bearing: first match wins, and hangul precedes kana precedes han (L17-42).
56. **`detect_script` tie-break:** `counts` is built in encounter order and `counts["latin"]` is assigned **after** the loop (L106), so on an exact tie the first non-Latin script _encountered in the text_ beats latin, and `max()` keeps the first maximum in iteration order. Go maps iterate randomly — this needs an explicit ordered counter or the result becomes nondeterministic.
57. `detect_script` returns `"unknown"` when there are no letters at all (L109). Golden cases, test_router.py:29-46: 14 inputs covering latin, devanagari, kana, han, hangul, arabic, tamil, cyrillic, thai, greek, hebrew, `""` and `"12345 6789"`.
58. _(order corrected 2026-09-20)_ `script_profile`'s dict is seeded with `{"latin": 0}` (L116), so **`latin` is first** and the other scripts follow in text-encounter order — the opposite of `detect_script`'s order in #56, where `latin` is appended after the loop. Both are observable and they are not the same order. `script_profile` returns fractions over alphabetic characters only, **omitting zero counts** (so `latin` disappears from a pure-Hindi profile), and `{}` when there are no letters (L113-130).
59. `non_latin_fraction = round(1.0 − profile.get("latin", 0.0), 4)`, or `0.0` when the profile is empty; the `"unknown"` branch hard-codes `0.0` (L168-171).
60. `state_text` flattens **string leaves only** of str/dict/list/tuple, depth-first in value order, with dict **keys ignored**, a depth cap of 6, joined with a single space, truncated to 4000 chars (L67-88). Asserted by test_router.py:80-86: English keys around Hindi content must still be detected as non-English.
61. `guess_latin_language`: fewer than 4 **words** → `None` (L139-140) — `_WORD.findall` counts every letter run in the text, not stop-word hits; the per-language matching happens afterwards at L141 _(wording corrected 2026-09-20)_. Words come from `[^\W\d_]+` — effectively Unicode letter runs — lowercased. Go's `\w` is ASCII-only in RE2, so use `unicode.IsLetter` splitting rather than a direct regex translation.
62. `diac_rate = (count of chars in _NON_EN_DIACRITICS) / max(1, len(lowered))`, computed over the **whole lowered text including spaces and punctuation**, not just letters (L143-145).
63. The non-English candidate is `max` over `{fr, de, es, pt, it, nl}` by score, ties broken by `_STOP`'s insertion order (fr, de, es, pt, it, nl) (L147-148). Go must keep an explicit ordered slice.
64. Decision ladder, in order (L149-156): (a) `best == 0 and diac_rate < 0.02` → `"en"` if `en > 0` else `None`; (b) `best_lg and best >= max(2, en + 2)` → `best_lg`; (c) `diac_rate >= 0.04 and best_lg and best >= en` → `best_lg`; (d) `"en"` if `en > 0` else `None`. Golden cases in test_router.py:65-76, including the guard that a long English sentence stays `"en"`.
65. `analyse`: on the latin path `is_english = lang in (None, "en")`, so an undecided Latin text counts as English; `"unknown"` returns `is_english=True`; any non-latin script returns `is_english=False` and `language=None` (L169-177).

**`email` (email.py)**

66. Normalisation order is `"\r\n"→"\n"`, then `"\r"→"\n"`, then the **two-character literal** `"\\n"` → `"\n"` (L25). The third replacement handles JSON-escaped bodies.
67. A quote-header break only fires when at least one line has already been kept (`and lines`, L28) — a mail that _opens_ with `From: …` is not emptied.
68. Lines whose `lstrip()` starts with `">"` are dropped individually; they do not terminate the scan (L30).
69. Kept lines are `rstrip()`ped (L32).
70. The signature scan starts at index `max(1, min(int(len(lines) * 0.6), len(lines) - 8))` and stops at the **first** line that is both ≤40 chars when stripped **and** matches a signature marker (L34-37). When `len(lines) - 8` is negative the `min` picks it and `max(1, …)` clamps the start to 1.
71. Paragraphs split on `r"\n\s*\n"`; any paragraph where the disclaimer regex **searches** (anywhere, not anchored) is dropped entirely (L39).
72. Survivors are `strip()`ped, empties dropped, joined with `"\n\n"`, then runs of spaces and tabs collapse to one space — newlines are preserved (L40). Finally truncated to `max_chars` (default 3000, L41). _(2026-09-20)_ The truncation counts **code points**, not bytes, and a negative `max_chars` is a Python slice from the end rather than an error — `clean_email_body("hello", -2) == "hel"`, and `0` gives `""`.
73. `email_state` builds keys in the order `subject` (stripped), `body`, then `from` **only when `sender` is truthy**, then non-`None` extras (L44-53). Order matters because it becomes the serialized prompt. _(2026-09-20)_ `state.update()` **replaces in place**, and `from` is the only key an extra can collide with (`subject`/`body`/`sender`/`clean` are named parameters, so Python raises `TypeError` instead). With a sender present, an extra `from` overrides the value and **keeps the sender's position**; without one it is appended at the end. A Go port that appends unconditionally emits a duplicate key and the wrong order.
74. Go regex notes: all these patterns are RE2-compatible (`{0,300}` is supported), but Go's `\w` and `\s` are ASCII-only while Python's are Unicode-aware for `str`. `^`/`$` need `(?m)` semantics only if you stop matching line-by-line — the Python code matches per line with `re.match`, so anchor with `\A`-style per-line matching in Go too.

---
