# PyPantograph

A Machine-to-Machine Interaction System for Lean 4.

## Installation

1. Install `uv`
2. Clone this repository with submodules:
```sh
git clone --recurse-submodules <repo-path>
```
3. Install `elan` and `lake`: See [Lean Manual](https://docs.lean-lang.org/lean4/doc/setup.html)
4. Execute
```sh
cd <repo-path>
uv sync
```

## Documentation

Build the documentations by
```sh
uvx --with notebook jupyter-book build docs
```
Then serve
```sh
cd docs/_build/html
python3 -m http.server -d .
```

## Examples

For API interaction examples, see `examples/README.md`.

## Referencing

[Paper Link](https://arxiv.org/abs/2410.16429)

```bib
@misc{pantograph,
      title={Pantograph: A Machine-to-Machine Interaction Interface for Advanced Theorem Proving, High Level Reasoning, and Data Extraction in Lean 4},
      author={Leni Aniva and Chuyue Sun and Brando Miranda and Clark Barrett and Sanmi Koyejo},
      year={2024},
      eprint={2410.16429},
      archivePrefix={arXiv},
      primaryClass={cs.LO},
      url={https://arxiv.org/abs/2410.16429},
}
```
