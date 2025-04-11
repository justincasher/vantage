# src/lean_automator/lean/prompts.py

"""Contains prompt templates for Lean code generation using an LLM.

These constants define the structured prompts sent to the language model
for generating Lean statement signatures and proof tactics.
"""

LEAN_STATEMENT_GENERATOR_PROMPT = """
You are an expert Lean 4 programmer translating mathematical statements into
formal Lean code. **You are working in a restricted environment with ONLY the
Lean 4 prelude and explicitly provided dependencies.**

**Goal:** Generate the Lean 4 statement signature (including necessary standard
imports if absolutely needed, usually none) for the item named `{unique_name}`
({item_type_name}), based on its LaTeX statement.

**LaTeX Statement:**
--- BEGIN LATEX STATEMENT ---
{latex_statement}
--- END LATEX STATEMENT ---

**Available Proven Dependencies (For Context - Names and Types):**
{dependency_context_for_statement}

**Refinement Feedback (If Applicable):**
{statement_error_feedback}

**Instructions:**
1.  Translate the LaTeX Statement into a formal Lean 4 theorem/definition
    signature (e.g., `theorem MyTheorem (n : Nat) : ...`). **IMPORTANT: Choose
    a Lean formulation that uses ONLY features available in the standard Lean 4
    prelude (like basic types `Nat`, `List`, `Prop`, `Type`, logical connectives
    `∀`, `∃`, `∧`, `∨`, `¬`, basic arithmetic `+`, `*`, `>`, `=`, induction
    principles) and the provided dependencies.** For example, expressing
    'infinitely many primes' as `∀ n, ∃ p > n, Nat.IsPrime p` is preferred over
    using `Set.Infinite` which requires extra libraries.
2.  Include necessary minimal standard imports ONLY if required beyond the
    prelude (e.g., often no imports are needed). **DO NOT generate imports for
    the dependencies listed above; they will be handled automatically.**
3.  **CRITICAL: DO NOT use or import ANYTHING from `mathlib` or `Std` unless
    explicitly provided in the dependencies.** Code relying on concepts like
    `Set`, `Finset`, `Data.`, `Mathlib.` etc., is INCORRECT for this task.
4.  Append ` := sorry` to the end of the statement signature.
5.  Output **only** the Lean code containing any necessary standard imports and
    the complete signature ending in `sorry`, marked between `--- BEGIN LEAN HEADER ---`
    and `--- END LEAN HEADER ---`. Here is an example output:

--- BEGIN LEAN HEADER ---
theorem MyTheorem (n : Nat) : Exists (m : Nat), m > n := sorry
--- END LEAN HEADER ---
"""

LEAN_PROOF_GENERATOR_PROMPT = """
You are an expert Lean 4 programmer completing a formal proof. **You are
working in a restricted environment with ONLY the Lean 4 prelude and explicitly
provided dependencies.**

**Goal:** Complete the Lean proof below by replacing `sorry`.

**Lean Statement Shell (Target):**
--- BEGIN LEAN HEADER ---
{lean_statement_shell}
--- END LEAN HEADER ---

**Informal LaTeX Proof (Use as Guidance):**
(This informal proof might contain errors, but use it as a guide for the formal
Lean proof structure and steps.)
--- BEGIN LATEX PROOF ---
{latex_proof}
--- END LATEX PROOF ---

**Available Proven Dependencies (Lean Code):**
(You MUST use these definitions and theorems. **Assume they are correctly
imported automatically.**)
{dependency_context_for_proof}

**Previous Attempt Error (If Applicable):**
(The previous attempt to compile the generated Lean code failed with the
following error. Please fix the proof tactics.)
--- BEGIN LEAN ERROR ---
{lean_error_log}
--- END LEAN ERROR ---

**Instructions:**
1.  Write the Lean 4 proof tactics to replace the `sorry` in the provided Lean
    Statement Shell.
2.  Ensure the proof strictly follows Lean 4 syntax and logic.
3.  You have access ONLY to Lean 4 prelude features (basic types, logic,
    induction, basic tactics like `rw`, `simp`, `intro`, `apply`, `exact`,
    `have`, `let`, `by_contra`, `cases`, `induction`, `rfl`) and the 'Available
    Proven Dependencies' provided above. **Use `simp` frequently to simplify
    goals and unfold definitions (like the definition of `List.append` when
    applied to `::`).** Use `rw [axiom_name]` or `rw [← axiom_name]` for
    intermediate steps. **Do NOT try to `rw` using function names (like
    `List.append`) or constructor names (like `List.cons`).**
4.  **CRITICAL: DO NOT use or import ANYTHING from `mathlib` or `Std` unless
    explicitly provided in the dependencies.** Code using `Set`, `Finset`,
    advanced tactics (like `linarith`, `ring`), or library functions beyond the
    prelude or provided dependencies is INCORRECT.
5.  **CRITICAL: DO NOT generate any `import` statements. Assume necessary
    dependency imports are already present.**
6.  Use the Informal LaTeX Proof as a *guide* but prioritize formal correctness
    using ONLY the allowed features.
7.  Before significant tactics (`rw`, `simp` variants, `apply`, `induction`
    steps, `cases`, `have`, `let`), add **two** comment lines in the following
    format:
    * `-- Goal: [Brief summary or key part of the current proof goal]`
    * `-- Action: [Explain the planned tactic, which rule/hypothesis (like 'ih')
       is used, and why (note rw ← if needed)]`
* Do **not** add these comments for simple tactics like `rfl`, `exact ...`,
  `done`, or simple structural syntax.
* **Example:**
    ```lean
    -- Goal: List.reverse l' = List.append (List.reverse l') []
    -- Action: Apply List.append_nil_ax to simplify the RHS
    rw [List.append_nil_ax]
    ```
* **Example:**
    ```lean
    -- Goal: List.append (List.append A B) [x] = List.append A (List.reverse (x :: xs))
    -- Action: Apply List.reverse_cons_ax to the RHS
    rw [List.reverse_cons_ax]
    ```
* **Example:**
    ```lean
    -- Goal: List.reverse (xs ++ l') ++ [x] = ...
    -- Action: Apply the induction hypothesis 'ih' to the LHS
    rw [ih]
    ```
8.  **TACTIC NOTE (Reflexivity):** After applying tactics like `rw [...]` or
    `simp [...]`, check if the resulting goal is of the form `X = X`. If it is,
    the goal is solved by reflexivity. **DO NOT add `rfl` in this case.** Simply
    proceed or end the branch. Avoid redundant tactics.
9.  **TACTIC NOTE (Finishing with Axioms/Hypotheses):** If the goal is an
    equality `LHS = RHS` and the *final* step is to apply a single axiom or
    hypothesis `h : LHS = RHS` (or `h : RHS = LHS`), **prefer using `exact h`
    (or `exact h.symm`)** instead of using `rw [h]` or `rw [← h]` as the very
    last tactic for that goal branch. Use `rw` for intermediate steps.
10. **Error Handling:** If fixing an error based on 'Previous Attempt Error',
    carefully analyze the error message and modify the proof tactics
    accordingly. **Do NOT change the theorem signature provided.**
    * **Specifically for "no goals to be solved" errors:** If the error log
      contains `error: ...:N:M: no goals to be solved` pointing to a line `N`
      containing a tactic (like `rfl`), it almost always means the goal was
      already solved implicitly by the tactic on the line *before* `N`. You
      should **remove the superfluous tactic on line `N`** in your corrected
      proof attempt.
11. Ensure the proof block is correctly terminated (e.g., no stray `end`).
12. Output **only** the complete Lean code block, including the *unchanged*
    statement signature, and the full proof replacing sorry (with comments),
    marked between `--- BEGIN LEAN CODE ---` and `--- END LEAN CODE ---`. Here is
    an example output:

--- BEGIN LEAN CODE ---
theorem MyTheorem (n : Nat) : Exists (m : Nat), m > n := by
  -- Use existence introduction
  apply Exists.intro (n + 1)
  -- Apply the definition of successor and less than
  simp [Nat.succ_eq_add_one, Nat.lt_succ_self]
--- END LEAN CODE ---
"""
