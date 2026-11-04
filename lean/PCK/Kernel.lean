/-
From summation trees to kernels: a GEMV row computed by any tree whose leaves visit every
column exactly once (padding columns contribute +0) satisfies the per-row hypothesis of
`t1_block`. This is the shape of both CUDA GEMV kernels: `pck.gemv.emulate_gemv` and
`emulate_gemv_coalesced` differ only in which tree they use, and
`pck.fp.tree_height(n, threads, vec)` is its height.
-/
import PCK.T1

namespace PCK

open Finset

namespace STree

/-- The term indices at the leaves, left to right. -/
def leaves : STree → List ℕ
  | leaf i => [i]
  | node l r => l.leaves ++ r.leaves

theorem evalExact_eq (x : ℕ → ℝ) (t : STree) : t.evalExact x = (t.leaves.map x).sum := by
  induction t with
  | leaf i => simp [evalExact, leaves]
  | node l r ihl ihr => simp [evalExact, leaves, ihl, ihr]

theorem absSum_eq (x : ℕ → ℝ) (t : STree) :
    t.absSum x = (t.leaves.map fun i => |x i|).sum := by
  induction t with
  | leaf i => simp [absSum, leaves]
  | node l r ihl ihr => simp [absSum, leaves, ihl, ihr]

end STree

/-- A list that is a permutation of `0, ..., N-1` sums like `Finset.range N`. -/
theorem sum_of_perm_range {l : List ℕ} {N : ℕ} (hl : l.Perm (List.range N)) (f : ℕ → ℝ) :
    (l.map f).sum = ∑ j ∈ range N, f j := by
  rw [(hl.map f).sum_eq, Finset.sum_eq_multiset_sum, Finset.range_val, ← Multiset.coe_range,
    Multiset.map_coe, Multiset.sum_coe]

/-- The row's terms on `ℕ`, zero past column `n` (the kernels' padding). -/
noncomputable def rowTerms {n : ℕ} (w x : Fin n → ℝ) (j : ℕ) : ℝ :=
  if h : j < n then w ⟨j, h⟩ * x ⟨j, h⟩ else 0

theorem sum_rowTerms {n N : ℕ} (hN : n ≤ N) (w x : Fin n → ℝ) (g : ℝ → ℝ) (hg : g 0 = 0) :
    ∑ j ∈ range N, g (rowTerms w x j) = ∑ j, g (w j * x j) := by
  rw [← sum_range_add_sum_Ico _ hN]
  have hpad : ∑ j ∈ Ico n N, g (rowTerms w x j) = 0 :=
    sum_eq_zero fun j hj => by
      have hjn : ¬ j < n := by have := (mem_Ico.mp hj).1; omega
      rw [rowTerms, dite_eq_right hjn, hg]
  rw [hpad, add_zero, Finset.sum_range]
  refine sum_congr rfl fun j _ => ?_
  simp [rowTerms]

/-- A row summed by a tree that visits each of the `N ≥ n` (padded) columns once, with
height at most `h`, is within the bound that `t1_block` assumes. -/
theorem row_within_bound {u : ℝ} {h n N : ℕ} (fl : AddModel u) (hu : 0 ≤ u)
    (hh : (h : ℝ) * u < 1) (hN : n ≤ N) (t : STree) (ht : t.height ≤ h)
    (hl : t.leaves.Perm (List.range N)) (w x : Fin n → ℝ) :
    |t.evalFl fl (rowTerms w x) - ∑ j, w j * x j| ≤ gamma h u * ∑ j, |w j * x j| := by
  have htu : (t.height : ℝ) * u < 1 :=
    lt_of_le_of_lt (mul_le_mul_of_nonneg_right (by exact_mod_cast ht) hu) hh
  have hexact : t.evalExact (rowTerms w x) = ∑ j, w j * x j := by
    rw [STree.evalExact_eq, sum_of_perm_range hl]
    simpa using sum_rowTerms hN w x id rfl
  have habs : t.absSum (rowTerms w x) = ∑ j, |w j * x j| := by
    rw [STree.absSum_eq, sum_of_perm_range hl (fun i => |rowTerms w x i|)]
    exact sum_rowTerms hN w x (fun v => |v|) abs_zero
  have := tree_sum_error fl (rowTerms w x) hu t htu
  rw [hexact, habs] at this
  exact this.trans (mul_le_mul_of_nonneg_right (gamma_mono hu ht hh)
    (sum_nonneg fun j _ => abs_nonneg _))

end PCK
