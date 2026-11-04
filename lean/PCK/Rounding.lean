/-
The standard rounding model for addition (Higham, Section 2.2), and the step inequality
for `gamma` that drives every error bound built from it.
-/
import PCK.Gamma
import PCK.STree

namespace PCK

/-- A rounded addition with unit roundoff `u`: `|fl(a + b) - (a + b)| ≤ u |a + b|`.
Round-to-nearest FP32 or FP64 addition satisfies this whenever the result does not
overflow (gradual underflow makes subnormal sums exact). -/
structure AddModel (u : ℝ) where
  add : ℝ → ℝ → ℝ
  rel_err : ∀ a b, |add a b - (a + b)| ≤ u * |a + b|

namespace STree

/-- The tree's sum computed with rounded additions. -/
def evalFl {u : ℝ} (fl : AddModel u) (x : ℕ → ℝ) : STree → ℝ
  | leaf i => x i
  | node l r => fl.add (l.evalFl fl x) (r.evalFl fl x)

end STree

/-- One more rounding on top of `k`: `(1 + u)(1 + gamma_k) ≤ 1 + gamma_{k+1}`. -/
theorem gamma_step {k : ℕ} {u : ℝ} (hu : 0 ≤ u) (hk : ((k : ℝ) + 1) * u < 1) :
    u + gamma k u + u * gamma k u ≤ gamma (k + 1) u := by
  have hku : (k : ℝ) * u < 1 := by nlinarith
  have hne : (1 : ℝ) - k * u ≠ 0 := by linarith
  have lhs : u + gamma k u + u * gamma k u = ((k : ℝ) + 1) * u / (1 - k * u) := by
    rw [eq_div_iff hne]
    unfold gamma
    have : k * u / (1 - k * u) * (1 - k * u) = (k : ℝ) * u := div_mul_cancel₀ _ hne
    linear_combination (1 + u) * this
  rw [lhs]
  unfold gamma
  push_cast
  apply div_le_div_of_nonneg_left
  · positivity
  · linarith
  · nlinarith

end PCK
