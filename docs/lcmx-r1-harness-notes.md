# LCM-X R1 harness notes (OmniMemEval fork, branch r1-measure)

Notes from the R1 recall measurement. They concern the harness, not LCM-X.

1. **LoCoMo image captions are never ingested for most libraries.** The default ingest path builds
   `speaker: text` only (`scripts/locomo/locomo_ingestion.py`). The zep and graphiti paths read the key
   `blip_captions`, but the dataset key is `blip_caption`, so they never attach a caption either. Some gold
   answers come from an image (for example, a sampled single-hop item whose gold is only in the caption).
   Not changed here.
2. **The BEAM abstention judge is lenient.** In a v0.28.0 board run, 15 of 40 abstention items scored 1.0
   although the judge's own reason says the answer gave details instead of abstaining. BEAM abstention scores
   are likely inflated for every system. Not changed here.
3. **LoCoMo dual-speaker duplication.** The default path stores the whole conversation under each speaker
   and pastes both searches; the shared-conversation paths (cognee, everos, supermemory) paste the same
   top-k into both slots. For a lossless store, about half of the 40 context lines are duplicates.
   `LCMX_LOCOMO_STORE=single` (opt-in, R1 protocol only; set `LCMX_TOPK_CAP>=2*TOPK`) keeps one store per
   conversation and fills the two slots with 2*top_k distinct hits. That is an effective top-k change:
   any public-board use needs disclosure and the board's acceptance.
4. **Thinking models.** `ANSWER_EXTRA_BODY` / `EVAL_EXTRA_BODY` (JSON objects) are merged into every request
   body, e.g. `{"thinking": {"type": "disabled"}}` for GLM at z.ai's OpenAI-compatible endpoint.
