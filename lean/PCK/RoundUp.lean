/-
Upward rounding. The checker evaluates the threshold's sums (a . |x| and sum |y|) with every
operation rounded up (CUDA `__fadd_ru`, `__fmul_ru`, `__dadd_ru`), so the computed value is
never below the exact one. Only `a + b ≤ fl(a + b)` is needed, whatever the tree.
-/
import PCK.Rounding

namespace PCK

/-- An addition that never rounds down. -/
structure UpAddModel where
  add : ℝ → ℝ → ℝ
  ge : ∀ a b, a + b ≤ add a b

namespace STree

/-- The tree's sum computed with round-up additions. -/
def evalUp (up : UpAddModel) (v : ℕ → ℝ) : STree → ℝ
  | leaf i => v i
  | node l r => up.add (l.evalUp up v) (r.evalUp up v)

/-- Rounding up from leaves that are upper bounds gives an upper bound on the exact sum. -/
theorem evalExact_le_evalUp (up : UpAddModel) {v w : ℕ → ℝ} (hvw : ∀ k, w k ≤ v k)
    (t : STree) : t.evalExact w ≤ t.evalUp up v := by
  induction t with
  | leaf i => exact hvw i
  | node l r ihl ihr =>
    simp only [evalExact, evalUp]
    exact (add_le_add ihl ihr).trans (up.ge _ _)

end STree

end PCK
