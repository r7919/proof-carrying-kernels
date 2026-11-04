/-
The error constant gamma_k = k u / (1 - k u) of Higham, *Accuracy and Stability of
Numerical Algorithms*, Section 3.1. It bounds the relative error of k chained roundings
with unit roundoff u, provided k u < 1.
-/
import Mathlib.Tactic

namespace PCK

/-- Higham's `gamma_k = k u / (1 - k u)`. -/
noncomputable def gamma (k : ℕ) (u : ℝ) : ℝ := k * u / (1 - k * u)

theorem gamma_zero (u : ℝ) : gamma 0 u = 0 := by simp [gamma]

theorem gamma_nonneg {k : ℕ} {u : ℝ} (hu : 0 ≤ u) (hk : k * u < 1) : 0 ≤ gamma k u := by
  unfold gamma
  apply div_nonneg
  · positivity
  · linarith

/-- `gamma` grows with `k`: more roundings, a larger bound. -/
theorem gamma_mono {j k : ℕ} {u : ℝ} (hu : 0 ≤ u) (hjk : j ≤ k) (hk : k * u < 1) :
    gamma j u ≤ gamma k u := by
  have hj : (j : ℝ) * u ≤ k * u := by
    have : (j : ℝ) ≤ k := by exact_mod_cast hjk
    nlinarith
  unfold gamma
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- `k u ≤ gamma_k`, so `gamma` is never smaller than the first-order term. -/
theorem le_gamma {k : ℕ} {u : ℝ} (hu : 0 ≤ u) (hk : k * u < 1) : k * u ≤ gamma k u := by
  unfold gamma
  rw [le_div_iff₀ (by linarith)]
  have : 0 ≤ (k : ℝ) * u := by positivity
  nlinarith

end PCK
