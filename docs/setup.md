# Setup

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

`uv build` builds a wheel of Pantograph in `dist` which can then be installed. For
example, a downstream project could have this line in its `pyproject.toml`

```toml
pantograph = { file = "path/to/wheel/dist/pantograph-0.3.0-cp312-cp312-manylinux_2_40_x86_64.whl" }
```

All interactions with Lean pass through the `Server` class. Create an instance of Pantograph using
```python
from pantograph import Server
server = Server()
```

## Lean Dependencies

The server created from `Server()` is sufficient for basic theorem proving tasks
reliant on Lean's `Init` library. Some users may find this insufficient and want
to use non-builtin libraries such as Aesop or Mathlib4.

To use external Lean dependencies such as
[Mathlib4](https://github.com/leanprover-community/mathlib4), Pantograph relies
on an existing Lean repository. Instructions for creating this repository can be
found [here](https://docs.lean-lang.org/lean4/doc/setup.html#lake).

After creating this initial Lean repository, execute in the repository
```sh
lake build
```

to build all files from the repository. This step is necessary after any file in
the repository is modified.

Then, feed the repository's path to the server
```python
server = Server(project_path="./path-to-lean-repo/")
```

For a complete example, see `examples/`.

## Server Parameters

The server has some additional options.

- `core_options`: These options are passed to Lean's kernel. For example
  `set_option pp.all true` in Lean corresponds to passing `pp.all=true` to
  `core_options`.
- `options`: These options are given to Pantograph itself. See below.
- `timeout`: This timeout controls the maximum wait time for the server
  instance. If the server instance does not respond within this timeout limit,
  it gets terminated. In some cases it is necessary to increase this if loading
  a Lean project takes too long.

A special note about running in Jupyter: Use the asynchronous version of each
function.

```python
server = await Server.create()
unit, = await server.load_sorry_async(sketch)
print(unit.goal_state)
```

### Options

- `automaticMode`: Set to false to disable automatic goal continuation.
- `timeout`: Set to a positive integer to set tactic execution timeout.
- `printDependentMVars`: Set to true to explicitly store goal inter-dependencies
