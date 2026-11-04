/-
Summation trees. A GPU reduction adds its terms in some binary tree; the rounding error
of the result is controlled by the tree's height, not by the number of terms
(Higham, Section 4.2). The GEMV kernels in `src/pck/cuda` use trees whose height is
`pck.fp.tree_height(n, threads, vec)`.
-/
import Mathlib.Tactic

namespace PCK

/-- A binary summation tree over term indices. -/
inductive STree where
  | leaf (i : ℕ)
  | node (l r : STree)
  deriving Repr, DecidableEq

namespace STree

/-- Height: the longest chain of additions from a leaf to the root. -/
def height : STree → ℕ
  | leaf _ => 0
  | node l r => max l.height r.height + 1

/-- The exact (real) sum of the terms at the leaves. -/
noncomputable def evalExact (x : ℕ → ℝ) : STree → ℝ
  | leaf i => x i
  | node l r => l.evalExact x + r.evalExact x

/-- The sum of absolute values of the terms at the leaves. -/
noncomputable def absSum (x : ℕ → ℝ) : STree → ℝ
  | leaf i => |x i|
  | node l r => l.absSum x + r.absSum x

theorem absSum_nonneg (x : ℕ → ℝ) (t : STree) : 0 ≤ t.absSum x := by
  induction t with
  | leaf i => exact abs_nonneg _
  | node l r ihl ihr => simp only [absSum]; linarith

theorem abs_evalExact_le_absSum (x : ℕ → ℝ) (t : STree) : |t.evalExact x| ≤ t.absSum x := by
  induction t with
  | leaf i => simp [evalExact, absSum]
  | node l r ihl ihr =>
    simp only [evalExact, absSum]
    exact (abs_add_le _ _).trans (add_le_add ihl ihr)

theorem height_left_lt (l r : STree) : l.height < (node l r).height := by
  simp only [height]; omega

theorem height_right_lt (l r : STree) : r.height < (node l r).height := by
  simp only [height]; omega

end STree

end PCK
