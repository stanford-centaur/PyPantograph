# Introduction

This is Pantograph, an machine-to-machine interaction interface for Lean 4.
Its main purpose is to train and evaluate theorem proving agents. The main
features are:
1. Interfacing via the Python library, REPL, or C Library
2. A mixture of expression-based and tactic-based proofs
3. Pantograph has been designed with facilitating search in mind. A theorem
   proving agent can simultaneously explore multiple branches of the proof.
4. Handling of metavariable coupling
5. Reading/Adding symbols from the environment
6. Extraction of tactic training data
7. Support for drafting

## Name

The name Pantograph is a pun. It means two things
- A pantograph is an instrument for copying down writing. As an agent explores
  the vast proof search space, Pantograph records the current state to ensure
  the proof is sound.
- A pantograph is also an equipment for an electric train. It supplies power to
  a locomotive. In comparison the (relatively) simple Pantograph software powers
  theorem proving projects.

## Design Rationale

The Lean 4 interface is not conducive to search. Readers familiar with Coq may
know that the Coq Serapi was superseded by CoqLSP. In the opinion of the
authors, this is a mistake. An interface conducive for human operators to write
proofs is often not an interface conductive to search.

Almost all of Pantograph's business logic is written in Lean, and Pantograph
achieves tighter coupling between the data extraction and proof search
components.

## Caveats and Limitations

Pantograph does not exactly mimic Lean LSP's behaviour. That would not grant the
flexibility it offers.  To support tree search means Pantograph has to act
differently from Lean in some times, but never at the sacrifice of soundness.

- When Lean LSP says "don't know how to synthesize placeholder", this indicates
  the human operator needs to manually move the cursor to the placeholder and
  type in the correct expression. This error therefore should not halt the proof
  process, and the placeholder should be turned into a goal.
- When Lean LSP says "unresolved goals", that means a proof cannot finish where
  it is supposed to finish at the end of a `by` block. Pantograph will raise the
  error in this case, since it indicates the termination of a proof search branch.
- `pick_goal` or `swap` will not work since they run contrary to tree search
  paradigms. However, if there are tactics which perform non-trivial operations
  to multiple goals at the same time, this constrain could potentially be
  relaxed at a cost of great bookkeeping overhead to the user.

Pantograph cannot perform things that are inherently constrained by Lean. These
include:

- If a tactic loses track of metavariables, it will not be caught until the end
  of the proof search. This is a bug in the tactic itself.
- Timeouts for executing tactics is not available. Maybe this will change in the
  future.
- Interceptions of parsing errors generally cannot be turned into goals (e.g.
  `def mystery : Nat := :=`) due to Lean's parsing system.

Each Pantograph version is anchored to a Lean version specified in
`src/lean-toolchain`. Features can be backported to older Lean versions upon
request.

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
