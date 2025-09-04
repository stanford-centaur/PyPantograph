# PyPantograph

A Machine-to-Machine Interaction System for Lean 4.

## Installation

1. Install `uv`
2. Install `elan`: See [Lean Manual](https://docs.lean-lang.org/lean4/doc/setup.html)

### Install as a project dependency

3. Add the package to your project:
```sh
uv add git+https://github.com/stanford-centaur/PyPantograph
uv sync
```

### Build wheels from source

3. Clone this repository with submodules:
```sh
git clone --recurse-submodules <repo-path>
```
4. Execute
```sh
cd <repo-path>
uv build
```
5. Built wheels can be found at `dist/*.whl`

## Documentation

Build the documentations by
```sh
uv run --group dev jupyter-book build doc
```
Then serve
```sh
python3 -m http.server -d doc/_build/html
```

### Examples

For API interaction examples, see `examples/README.md`.

### Contributing

Execute unit tests with

```sh
uv run pytest
```

## Reference

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
