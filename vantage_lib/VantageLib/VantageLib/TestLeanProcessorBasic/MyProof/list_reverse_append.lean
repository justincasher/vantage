-- Auto-generated imports based on plan_dependencies --
import VantageLib.TestLeanProcessorBasic.List.append_assoc_ax
import VantageLib.TestLeanProcessorBasic.List.append_nil_ax
import VantageLib.TestLeanProcessorBasic.List.nil_append_ax
import VantageLib.TestLeanProcessorBasic.List.reverse_cons_ax
import VantageLib.TestLeanProcessorBasic.List.reverse_nil_ax

theorem VantageLib.TestLeanProcessorBasic.MyProof.list_reverse_append {α : Type} (l l' : List α) : List.reverse (l ++ l') = List.reverse l' ++ List.reverse l := by
  induction l with
  | nil =>
    -- Goal: List.reverse ([] ++ l') = List.reverse l' ++ List.reverse []
    -- Action: Use nil_append_ax to simplify the left-hand side.
    rw [VantageLib.TestLeanProcessorBasic.List.nil_append_ax]
    -- Goal: List.reverse l' = List.reverse l' ++ List.reverse []
    -- Action: Use reverse_nil_ax to simplify the right-hand side.
    rw [VantageLib.TestLeanProcessorBasic.List.reverse_nil_ax]
    -- Goal: List.reverse l' = List.reverse l' ++ []
    -- Action: Use append_nil_ax to simplify the right-hand side.
    rw [VantageLib.TestLeanProcessorBasic.List.append_nil_ax]
  | cons x xs ih =>
    -- Goal: List.reverse ((x :: xs) ++ l') = List.reverse l' ++ List.reverse (x :: xs)
    -- Action: By definition of append, (x :: xs) ++ l' = x :: (xs ++ l').
    rw [List.cons_append] -- Rewriting with this non-axiom is acceptable in intermediate steps to unfold the definition of `++` when applied to cons
    -- Goal: List.reverse (x :: (xs ++ l')) = List.reverse l' ++ List.reverse (x :: xs)
    -- Action: Use reverse_cons_ax to transform the left-hand side.
    rw [VantageLib.TestLeanProcessorBasic.List.reverse_cons_ax]
    -- Goal: List.reverse (xs ++ l') ++ [x] = List.reverse l' ++ List.reverse (x :: xs)
    -- Action: Use reverse_cons_ax to transform the right-hand side.
    rw [VantageLib.TestLeanProcessorBasic.List.reverse_cons_ax]
    -- Goal: List.reverse (xs ++ l') ++ [x] = List.reverse l' ++ (List.reverse xs ++ [x])
    -- Action: Apply the induction hypothesis 'ih' to the left-hand side
    rw [ih]
    -- Goal: List.reverse l' ++ List.reverse xs ++ [x] = List.reverse l' ++ (List.reverse xs ++ [x])
    -- Action: Use append_assoc_ax to rearrange the right-hand side.
    exact VantageLib.TestLeanProcessorBasic.List.append_assoc_ax (List.reverse l') (List.reverse xs) [x]