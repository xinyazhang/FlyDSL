* `_KERNEL_BLOCK_M = 128` at interface.py is dead code
* Move `from dataclasses import dataclass, fields, replace` to the beginning
* There are three knobs for fp math? `fp_mode`/`unsafe_fp_math`/`fast_fp_math`
* You probably misread my plan: I'm leaning let caller construct
  FmhaInputMetadata object and pass it to plan()
  + The overall principle: always lean to pass arguments as packed
    dataclass object if there are too many arguments.
      - Actually I discussed with FlyDSL developer and they also agree we should
        do this when passing things to @flyc.jit/kernel but this feature is not
        implemented yet.
