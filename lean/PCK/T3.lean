/-
T3 (the checker's own arithmetic is safe), and T1/T2 for the values the checker computes.

The checker never sees the exact residual rho or the exact threshold. It computes
  rho_hat  in FP64 (round to nearest) from compensated FP32 partials, and
  tau_hat  = coef * s + geps * ((1 + u) * s + yabs) + floor, every operation rounded up,
where s >= a . |x| and yabs >= sum |y| are round-up sums (`evalExact_le_evalUp`) and the
error of rho_hat is bounded by `perturbed_tree_error` / `compound_error`. These theorems
take those facts as hypotheses and conclude what the kernel decides:

  * `t3_no_false_alarm`: if T1's bound holds for the exact residual, |rho_hat| <= tau_hat.
  * `t3_detects`: if the residual is off by more than tau_hat + E, |rho_hat| > tau_hat.
  * `t3_single_fault_detected`: one output off by more than 2 tau_hat always fails.

Assumptions not proven here (stated as hypotheses where used):
  * every rounded operation satisfies the standard model (`AddModel`, round-up `ge`),
    i.e. no overflow; products of an FP32 and a BF16 value are exact in FP64;
  * the per-thread compensated FP32 dot product (Dot2, Ogita, Rump and Oishi 2005) is within
    gamma_{2L}(u32)^2 * sum |c_j x_j| (+ an underflow floor) of its chain's exact sum.
-/
import PCK.T2
import PCK.Residual
import PCK.RoundUp

namespace PCK

/-- The computed check passes whenever the exact residual is within T1's bound. -/
theorem t3_no_false_alarm {u gh coef geps fl S A Y s yabs rho rhoHat tau : ℝ}
    (hu : 0 ≤ u) (hgh : 0 ≤ gh) (hgeps : 0 ≤ geps) (hS0 : 0 ≤ S)
    (hrho : |rho| ≤ (gh + u) * S)
    (herr : |rhoHat - rho| ≤ geps * (A + Y) + fl)
    (hA : A ≤ (1 + u) * S) (hS : S ≤ s) (hY : Y ≤ yabs) (hcoef : gh + u ≤ coef)
    (htau : coef * s + geps * ((1 + u) * s + yabs) + fl ≤ tau) :
    |rhoHat| ≤ tau := by
  have h1 : (gh + u) * S ≤ coef * s := mul_le_mul hcoef hS hS0 (by linarith)
  have h2 : geps * (A + Y) ≤ geps * ((1 + u) * s + yabs) := by
    apply mul_le_mul_of_nonneg_left _ hgeps
    have : (1 + u) * S ≤ (1 + u) * s := mul_le_mul_of_nonneg_left hS (by linarith)
    linarith
  calc |rhoHat| = |rho + (rhoHat - rho)| := by ring_nf
    _ ≤ |rho| + |rhoHat - rho| := abs_add_le _ _
    _ ≤ tau := by linarith

/-- The computed check fails whenever the exact residual exceeds `tau + E`, where `E`
bounds the checker's rounding of the residual. -/
theorem t3_detects {E rho rhoHat tau : ℝ} (herr : |rhoHat - rho| ≤ E) (hbig : tau + E < |rho|) :
    tau < |rhoHat| := by
  have : |rho| ≤ |rhoHat| + |rhoHat - rho| := by
    calc |rho| = |rhoHat - (rhoHat - rho)| := by ring_nf
      _ ≤ |rhoHat| + |rhoHat - rho| := abs_sub _ _
  linarith

open Finset in
/-- T2 for the computed check: if one output deviates by more than `2 * tau` (and the rest
are within spec), the block fails, for every sign vector. Needs `tau` to cover the exact
threshold plus the checker's rounding `E`, which `tau_hat` does by construction. -/
theorem t3_single_fault_detected {b n : ℕ} {u : ℝ} {h : ℕ} (hu : 0 ≤ u)
    (hh : (h : ℝ) * u < 1)
    (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ) (y : Fin b → ℝ) (i₀ : Fin b)
    (hy : ∀ i, i ≠ i₀ → |y i - ∑ j, W i j * x j| ≤ gamma h u * ∑ j, |W i j * x j|)
    (r : Fin b → ℝ) (hr : ∀ i, r i = 1 ∨ r i = -1)
    (c : Fin n → ℝ) (hc : ∀ j, |c j - ∑ i, r i * W i j| ≤ u * |∑ i, r i * W i j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j)
    {E rhoHat tau : ℝ}
    (herr : |rhoHat - (∑ j, c j * x j - ∑ i, r i * y i)| ≤ E)
    (hcover : (gamma h u + u) * ∑ j, a j * |x j| + E ≤ tau)
    (hdev : 2 * tau < |y i₀ - ∑ j, W i₀ j * x j|) :
    tau < |rhoHat| := by
  have hlow := t2_residual_lower hu hh W x y i₀ hy r hr c hc a ha
  exact t3_detects herr (by linarith)

end PCK
