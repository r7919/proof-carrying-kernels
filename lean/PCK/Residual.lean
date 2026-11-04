/-
The checker's own rounding of the residual. The checker sums its residual terms in an FP64
tree: the per-thread partials of c . x (each from a compensated FP32 chain, so slightly off
the chain's exact sum) and the terms -r_i y_i (exact). Two errors compound: the leaves'
perturbation and the tree's rounding.
-/
import PCK.TreeError

namespace PCK

open STree

namespace STree

theorem evalExact_sub (v w : ℕ → ℝ) (t : STree) :
    t.evalExact (fun k => v k - w k) = t.evalExact v - t.evalExact w := by
  induction t with
  | leaf i => rfl
  | node l r ihl ihr => simp only [evalExact, ihl, ihr]; ring

theorem absSum_le_add (v w : ℕ → ℝ) (t : STree) :
    t.absSum v ≤ t.absSum w + t.absSum (fun k => v k - w k) := by
  induction t with
  | leaf i =>
    simp only [absSum]
    calc |v i| = |w i + (v i - w i)| := by ring_nf
      _ ≤ |w i| + |v i - w i| := abs_add_le _ _
  | node l r ihl ihr => simp only [absSum]; linarith

end STree

/-- Rounded tree over perturbed leaves: if the leaves `v` are within a total of `D` of the
exact leaves `w`, the computed sum is within `gamma_H * (sum |w| + D) + D` of the exact sum. -/
theorem perturbed_tree_error {u : ℝ} (fl : AddModel u) (hu : 0 ≤ u) (t : STree)
    (ht : (t.height : ℝ) * u < 1) (v w : ℕ → ℝ) {D : ℝ}
    (hD : t.absSum (fun k => v k - w k) ≤ D) :
    |t.evalFl fl v - t.evalExact w| ≤ gamma t.height u * (t.absSum w + D) + D := by
  have hg := gamma_nonneg hu ht
  have h1 := tree_sum_error fl v hu t ht
  have h2 : |t.evalExact v - t.evalExact w| ≤ D := by
    rw [← evalExact_sub]
    exact (abs_evalExact_le_absSum _ t).trans hD
  have h3 : t.absSum v ≤ t.absSum w + D := (absSum_le_add v w t).trans (by linarith)
  calc |t.evalFl fl v - t.evalExact w|
      = |(t.evalFl fl v - t.evalExact v) + (t.evalExact v - t.evalExact w)| := by ring_nf
    _ ≤ |t.evalFl fl v - t.evalExact v| + |t.evalExact v - t.evalExact w| := abs_add_le _ _
    _ ≤ gamma t.height u * t.absSum v + D := add_le_add h1 h2
    _ ≤ gamma t.height u * (t.absSum w + D) + D := by gcongr

/-- The constant the checker uses: with `D ≤ g2 * A + eta` (Dot2's guarantee per chain,
summed) and `sum |w| ≤ A + Y`, the error is at most `(gH + g2 + gH * g2) * (A + Y) +
(1 + gH) * eta`. -/
theorem compound_error {gH g2 A Y D eta wSum : ℝ} (hgH : 0 ≤ gH) (hg2 : 0 ≤ g2)
    (hY : 0 ≤ Y) (hw : wSum ≤ A + Y) (hD : D ≤ g2 * A + eta) :
    gH * (wSum + D) + D ≤ (gH + g2 + gH * g2) * (A + Y) + (1 + gH) * eta := by
  have : gH * (wSum + D) ≤ gH * (A + Y + (g2 * A + eta)) :=
    mul_le_mul_of_nonneg_left (by linarith) hgH
  nlinarith [mul_nonneg hgH hg2, mul_nonneg (mul_nonneg hgH hg2) hY, mul_nonneg hg2 hY]

end PCK
