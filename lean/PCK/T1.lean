/-
T1 (no false alarms), for one checksum block, before the checker's own rounding.

Setting: a block of `b` output rows of y = W x. Each computed output `y i` is within the
summation-tree bound of its exact row sum (what `tree_sum_error` gives for any tree of
height `h`). The checker holds secret signs `r i ∈ {±1}`, checksums `c j` that equal
`∑ i, r i * W i j` up to relative error `u` (FP32 storage), and bounds
`a j ≥ ∑ i, |W i j|` (stored rounded up). Then the residual is within the threshold:

  |∑ j, c j * x j - ∑ i, r i * y i| ≤ (gamma h u + u) * ∑ j, a j * |x j|

for every sign pattern and every input. The checker's own FP64 evaluation of both sides is
handled separately (T3).
-/
import PCK.TreeError

namespace PCK

open Finset

variable {b n : ℕ}

theorem abs_signed_sum_le {r : Fin b → ℝ} (hr : ∀ i, r i = 1 ∨ r i = -1) (v : Fin b → ℝ) :
    |∑ i, r i * v i| ≤ ∑ i, |v i| := by
  refine (abs_sum_le_sum_abs _ _).trans (le_of_eq (sum_congr rfl fun i _ => ?_))
  rcases hr i with h | h <;> simp [h]

/-- The residual splits into the checksum rounding and the kernel's rounding. -/
theorem residual_split (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ) (y r : Fin b → ℝ)
    (c : Fin n → ℝ) :
    ∑ j, c j * x j - ∑ i, r i * y i =
      ∑ j, (c j - ∑ i, r i * W i j) * x j - ∑ i, r i * (y i - ∑ j, W i j * x j) := by
  simp only [sub_mul, mul_sub, sum_sub_distrib, sum_mul, mul_sum]
  rw [sum_comm (f := fun j i => r i * W i j * x j)]
  simp only [mul_assoc]
  ring

/-- Checksum rounding contributes at most `u * (a . |x|)`. -/
theorem checksum_term_le {u : ℝ} (hu : 0 ≤ u) (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ)
    {r : Fin b → ℝ} (hr : ∀ i, r i = 1 ∨ r i = -1)
    (c : Fin n → ℝ) (hc : ∀ j, |c j - ∑ i, r i * W i j| ≤ u * |∑ i, r i * W i j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j) :
    |∑ j, (c j - ∑ i, r i * W i j) * x j| ≤ u * ∑ j, a j * |x j| := by
  refine (abs_sum_le_sum_abs _ _).trans ?_
  rw [mul_sum]
  refine sum_le_sum fun j _ => ?_
  rw [abs_mul, ← mul_assoc]
  refine mul_le_mul_of_nonneg_right ?_ (abs_nonneg _)
  exact (hc j).trans (mul_le_mul_of_nonneg_left
    ((abs_signed_sum_le hr fun i => W i j).trans (ha j)) hu)

/-- Rows within the tree bound contribute at most `gamma_h * (a . |x|)` in total, for any set
of rows `s`. -/
theorem rows_term_le {g : ℝ} (hg : 0 ≤ g) (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ)
    (y : Fin b → ℝ) (s : Finset (Fin b))
    (hy : ∀ i ∈ s, |y i - ∑ j, W i j * x j| ≤ g * ∑ j, |W i j * x j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j) :
    ∑ i ∈ s, |y i - ∑ j, W i j * x j| ≤ g * ∑ j, a j * |x j| := by
  calc ∑ i ∈ s, |y i - ∑ j, W i j * x j|
      ≤ ∑ i ∈ s, g * ∑ j, |W i j * x j| := sum_le_sum hy
    _ ≤ ∑ i, g * ∑ j, |W i j * x j| :=
        sum_le_sum_of_subset_of_nonneg (subset_univ s) fun i _ _ =>
          mul_nonneg hg (sum_nonneg fun j _ => abs_nonneg _)
    _ = g * ∑ j, (∑ i, |W i j|) * |x j| := by
        rw [← mul_sum, sum_comm]
        simp only [abs_mul, sum_mul]
    _ ≤ g * ∑ j, a j * |x j| := by
        refine mul_le_mul_of_nonneg_left (sum_le_sum fun j _ => ?_) hg
        exact mul_le_mul_of_nonneg_right (ha j) (abs_nonneg _)

/-- T1 for one block: the exact residual never exceeds `(gamma_h + u) * (a . |x|)`. -/
theorem t1_block {u : ℝ} {h : ℕ} (hu : 0 ≤ u) (hh : (h : ℝ) * u < 1)
    (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ) (y : Fin b → ℝ)
    (hy : ∀ i, |y i - ∑ j, W i j * x j| ≤ gamma h u * ∑ j, |W i j * x j|)
    (r : Fin b → ℝ) (hr : ∀ i, r i = 1 ∨ r i = -1)
    (c : Fin n → ℝ) (hc : ∀ j, |c j - ∑ i, r i * W i j| ≤ u * |∑ i, r i * W i j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j) :
    |∑ j, c j * x j - ∑ i, r i * y i| ≤ (gamma h u + u) * ∑ j, a j * |x j| := by
  have hkern : |∑ i, r i * (y i - ∑ j, W i j * x j)| ≤ gamma h u * ∑ j, a j * |x j| :=
    (abs_signed_sum_le hr _).trans
      (rows_term_le (gamma_nonneg hu hh) W x y univ (fun i _ => hy i) a ha)
  rw [residual_split W x y r c]
  calc _ ≤ |∑ j, (c j - ∑ i, r i * W i j) * x j| + |∑ i, r i * (y i - ∑ j, W i j * x j)| :=
        abs_sub _ _
    _ ≤ u * ∑ j, a j * |x j| + gamma h u * ∑ j, a j * |x j| :=
        add_le_add (checksum_term_le hu W x hr c hc a ha) hkern
    _ = (gamma h u + u) * ∑ j, a j * |x j| := by ring

end PCK
