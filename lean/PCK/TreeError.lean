/-
The core lemma behind T1 (Higham, Section 4.2): summing with any binary tree of height `h`
in a rounding model with unit roundoff `u` has error at most `gamma_h * sum |x_i|`. The bound
depends on the tree's height, not on the number of terms, which is what makes the checker's
threshold tight enough to catch low-precision shortcuts.

Property-tested in `tests/test_gemv.py::test_any_summation_tree_obeys_height_bound`.
-/
import PCK.Rounding

namespace PCK

open STree

theorem tree_sum_error {u : ℝ} (fl : AddModel u) (x : ℕ → ℝ) (hu : 0 ≤ u) (t : STree)
    (ht : (t.height : ℝ) * u < 1) :
    |t.evalFl fl x - t.evalExact x| ≤ gamma t.height u * t.absSum x := by
  induction t with
  | leaf i => simp [evalFl, evalExact, height, gamma]
  | node l r ihl ihr =>
    have hh : (node l r).height = max l.height r.height + 1 := rfl
    set k := max l.height r.height
    rw [hh] at ht ⊢
    push_cast at ht
    have hku : (k : ℝ) * u < 1 := by nlinarith
    have hl : l.height ≤ k := le_max_left _ _
    have hr : r.height ≤ k := le_max_right _ _
    have hlu : (l.height : ℝ) * u < 1 :=
      lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hl) hu) hku
    have hru : (r.height : ℝ) * u < 1 :=
      lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast hr) hu) hku
    have hA := absSum_nonneg x l
    have hB := absSum_nonneg x r
    have el : |l.evalFl fl x - l.evalExact x| ≤ gamma k u * l.absSum x :=
      (ihl hlu).trans (mul_le_mul_of_nonneg_right (gamma_mono hu hl hku) hA)
    have er : |r.evalFl fl x - r.evalExact x| ≤ gamma k u * r.absSum x :=
      (ihr hru).trans (mul_le_mul_of_nonneg_right (gamma_mono hu hr hku) hB)
    have sl := abs_evalExact_le_absSum x l
    have sr := abs_evalExact_le_absSum x r
    have hg : 0 ≤ gamma k u := gamma_nonneg hu hku
    have hstep := gamma_step hu ht
    simp only [evalFl, evalExact, absSum]
    generalize l.evalFl fl x = a at el ⊢
    generalize r.evalFl fl x = b at er ⊢
    generalize l.evalExact x = sa at el sl ⊢
    generalize r.evalExact x = sb at er sr ⊢
    generalize l.absSum x = A at el sl hA ⊢
    generalize r.absSum x = B at er sr hB ⊢
    have hround := fl.rel_err a b
    have hab : |a + b| ≤ (1 + gamma k u) * (A + B) := by
      have : |a + b| ≤ |a - sa| + |b - sb| + |sa| + |sb| := by
        calc |a + b| = |(a - sa) + (b - sb) + sa + sb| := by ring_nf
          _ ≤ |(a - sa) + (b - sb) + sa| + |sb| := abs_add_le _ _
          _ ≤ |(a - sa) + (b - sb)| + |sa| + |sb| := by gcongr; exact abs_add_le _ _
          _ ≤ |a - sa| + |b - sb| + |sa| + |sb| := by gcongr; exact abs_add_le _ _
      nlinarith
    calc |fl.add a b - (sa + sb)|
        = |(fl.add a b - (a + b)) + ((a - sa) + (b - sb))| := by ring_nf
      _ ≤ |fl.add a b - (a + b)| + (|a - sa| + |b - sb|) :=
          (abs_add_le _ _).trans (by gcongr; exact abs_add_le _ _)
      _ ≤ u * ((1 + gamma k u) * (A + B)) + (gamma k u * A + gamma k u * B) := by
          gcongr
          exact hround.trans (mul_le_mul_of_nonneg_left hab hu)
      _ = (u + gamma k u + u * gamma k u) * (A + B) := by ring
      _ ≤ gamma (k + 1) u * (A + B) := mul_le_mul_of_nonneg_right hstep (by linarith)

end PCK
