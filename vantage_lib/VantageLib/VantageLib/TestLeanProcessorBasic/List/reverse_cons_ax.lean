universe u
axiom VantageLib.TestLeanProcessorBasic.List.reverse_cons_ax {α : Type u} (x : α) (xs : List α) : List.reverse (x :: xs) = List.append (List.reverse xs) [x]