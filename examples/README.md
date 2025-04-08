# Examples

This example showcases how to bind library dependencies and execute the `Aesop`
tactic in Lean. First build the example project:
``` sh
pushd Example
lake build
popd
```
This would generate compiled `.olean` files. Then run one of the examples from the
project root:
``` sh
poetry run examples/aesop.py
poetry run examples/sketch.py
```

Warning: If you make modifications to any Lean files, you must re-run `lake
build`! Moreover, the version of the Lean used in the example folder (including
dependencies in `lakefile.lean` and `lean-toolchain`) **must match exactly**
with the version in `src/`!

* `aesop.py`: Example of how to use the `aesop` tactic
* `sketch.py`: Example of loading a sketch

