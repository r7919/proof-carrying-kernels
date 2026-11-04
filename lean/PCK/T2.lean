/-
T2 (guaranteed detection), for one checksum block, before the checker's own rounding.

If every output in the block but one (`i₀`) is within the summation-tree bound, and `y i₀`
is off from its exact value by more than twice the threshold `τ = (gamma_h + u) * (a . |x|)`,
then the exact residual exceeds `τ`, for every sign pattern: the faulty output enters the
residual with weight `|r i₀| = 1`, and everything else contributes at most `τ`.

Property-tested in `tests/test_checker.py::test_large_single_error_always_caught` (CPU) and
`tests/gpu/test_checker_cuda.py::test_large_single_error_always_caught` (GPU).
-/
import PCK.T1

namespace PCK

open Finset

variable {b n : ℕ}

theorem t2_block {u : ℝ} {h : ℕ} (hu : 0 ≤ u) (hh : (h : ℝ) * u < 1)
    (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ) (y : Fin b → ℝ) (i₀ : Fin b)
    (hy : ∀ i, i ≠ i₀ → |y i - ∑ j, W i j * x j| ≤ gamma h u * ∑ j, |W i j * x j|)
    (r : Fin b → ℝ) (hr : ∀ i, r i = 1 ∨ r i = -1)
    (c : Fin n → ℝ) (hc : ∀ j, |c j - ∑ i, r i * W i j| ≤ u * |∑ i, r i * W i j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j)
    (hdev : 2 * ((gamma h u + u) * ∑ j, a j * |x j|) < |y i₀ - ∑ j, W i₀ j * x j|) :
    (gamma h u + u) * ∑ j, a j * |x j| < |∑ j, c j * x j - ∑ i, r i * y i| := by
  set S := ∑ j, a j * |x j|
  set D := fun i => y i - ∑ j, W i j * x j
  have hD0 : |r i₀ * D i₀| = |D i₀| := by rcases hr i₀ with h0 | h0 <;> simp [h0]
  -- The other rows contribute at most gamma_h * S.
  have hrest : |∑ i ∈ univ.erase i₀, r i * D i| ≤ gamma h u * S := by
    refine (abs_sum_le_sum_abs _ _).trans ?_
    have : ∑ i ∈ univ.erase i₀, |r i * D i| = ∑ i ∈ univ.erase i₀, |D i| :=
      sum_congr rfl fun i _ => by rcases hr i with h1 | h1 <;> simp [h1]
    rw [this]
    exact rows_term_le (gamma_nonneg hu hh) W x y _
      (fun i hi => hy i (ne_of_mem_erase hi)) a ha
  have hchk := checksum_term_le (x := x) hu W hr c hc a ha
  have hsplit : ∑ i, r i * D i = r i₀ * D i₀ + ∑ i ∈ univ.erase i₀, r i * D i :=
    (add_sum_erase _ _ (mem_univ i₀)).symm
  rw [residual_split W x y r c]
  change _ < |∑ j, (c j - ∑ i, r i * W i j) * x j - ∑ i, r i * D i|
  rw [hsplit]
  -- |A - (B + R)| ≥ |B| - |A| - |R|
  have key : |r i₀ * D i₀| - |∑ j, (c j - ∑ i, r i * W i j) * x j|
      - |∑ i ∈ univ.erase i₀, r i * D i|
      ≤ |∑ j, (c j - ∑ i, r i * W i j) * x j - (r i₀ * D i₀ + ∑ i ∈ univ.erase i₀, r i * D i)| := by
    have h1 := abs_sub_abs_le_abs_sub (r i₀ * D i₀)
      (∑ j, (c j - ∑ i, r i * W i j) * x j - ∑ i ∈ univ.erase i₀, r i * D i)
    have h2 := abs_sub (∑ j, (c j - ∑ i, r i * W i j) * x j)
      (∑ i ∈ univ.erase i₀, r i * D i)
    have e : r i₀ * D i₀ - (∑ j, (c j - ∑ i, r i * W i j) * x j -
        ∑ i ∈ univ.erase i₀, r i * D i) =
        -(∑ j, (c j - ∑ i, r i * W i j) * x j - (r i₀ * D i₀ + ∑ i ∈ univ.erase i₀, r i * D i)) := by
      ring
    rw [e, abs_neg] at h1
    linarith
  rw [hD0] at key
  have hD : 2 * ((gamma h u + u) * S) < |D i₀| := hdev
  nlinarith
end PCK
