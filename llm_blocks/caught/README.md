# Blocks an LLM signed off that were wrong

Kept so the checks that now catch them keep catching them.

- `b_rmsnorm_run5.v`: Sonnet's RMSNorm from the fifth end-to-end run
  (tiny_qwen3, d_model 64). It never gives index 0 and gives index 63
  twice, so the count of outputs is right and every value it gives is
  too: it passed all 268 checks of the testbench as it was, and the
  decode step built from it came out wrong. The testbench now checks
  each index exactly once, and the formal contract (`contracts.py`)
  fails it on "when busy falls, every index has been given exactly
  once", with index 0 never given and 63 given on cycles 151 and 152.
