/-
The top-k sampling check is exactly right.

The checker (`pck.sampling.check_topk`, `sampling_kernel.cu`) accepts a selection `S` of
indices into the logits `l` iff `S` has `k` elements and every selected logit is at least
every rejected one. `topK_check_iff_max_sum` shows this is the same as `S` being a set of
`k` indices with the largest possible total logit: an independent definition of "a top-k
set" that also allows any tie-breaking. `prefix_iff_min_ge_max` shows the kernel's
computation (min over selected >= max over rejected) decides the same thing.
-/
import Mathlib.Tactic

namespace PCK

open Finset

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- What the checker tests: every selected logit is >= every rejected logit. -/
def IsPrefix (l : V → ℝ) (S : Finset V) : Prop := ∀ i ∈ S, ∀ j ∉ S, l j ≤ l i

/-- The checker's top-k verdict. -/
def TopKCheck (l : V → ℝ) (k : ℕ) (S : Finset V) : Prop := S.card = k ∧ IsPrefix l S

omit [Fintype V] [DecidableEq V] in
/-- Two equal-size sets where every element of `A` is at most every element of `B`. -/
theorem sum_le_sum_of_forall_le {l : V → ℝ} {A B : Finset V} (hcard : A.card = B.card)
    (hle : ∀ a ∈ A, ∀ b ∈ B, l a ≤ l b) : ∑ a ∈ A, l a ≤ ∑ b ∈ B, l b := by
  rcases A.eq_empty_or_nonempty with rfl | hA
  · have : B = ∅ := card_eq_zero.mp (by simpa using hcard.symm)
    simp [this]
  · set c := A.sup' hA l
    have h1 : ∑ a ∈ A, l a ≤ A.card • c := sum_le_card_nsmul A l c fun a ha => le_sup' l ha
    have h2 : B.card • c ≤ ∑ b ∈ B, l b := card_nsmul_le_sum B l c fun b hb =>
      sup'_le hA l fun a ha => hle a ha b hb
    rw [hcard] at h1
    linarith

omit [Fintype V] in
theorem topK_check_iff_max_sum (l : V → ℝ) (k : ℕ) (S : Finset V) :
    TopKCheck l k S ↔ S.card = k ∧ ∀ T : Finset V, T.card = k → ∑ i ∈ T, l i ≤ ∑ i ∈ S, l i := by
  constructor
  · rintro ⟨hk, hpre⟩
    refine ⟨hk, fun T hT => ?_⟩
    -- Compare only where the sets differ; those parts have equal size.
    have hcard : (T \ S).card = (S \ T).card := by
      have h1 := card_sdiff_add_card_inter T S
      have h2 := card_sdiff_add_card_inter S T
      rw [inter_comm] at h2
      omega
    have hle : ∑ i ∈ T \ S, l i ≤ ∑ i ∈ S \ T, l i :=
      sum_le_sum_of_forall_le hcard fun a ha b hb =>
        hpre b (mem_sdiff.mp hb).1 a (mem_sdiff.mp ha).2
    have hT := sum_inter_add_sum_sdiff T S l
    have hS := sum_inter_add_sum_sdiff S T l
    rw [inter_comm] at hS
    linarith
  · rintro ⟨hk, hmax⟩
    refine ⟨hk, fun i hi j hj => ?_⟩
    by_contra hnot
    have hlt : l i < l j := lt_of_not_ge hnot
    -- Swapping i out and j in would give a k-set with a larger total.
    set T := insert j (S.erase i)
    have hjT : j ∉ S.erase i := fun h => hj (mem_of_mem_erase h)
    have hTcard : T.card = k := by
      rw [card_insert_of_notMem hjT, card_erase_of_mem hi, hk]
      have : 0 < k := hk ▸ card_pos.mpr ⟨i, hi⟩
      omega
    have hsum : ∑ x ∈ T, l x = ∑ x ∈ S, l x - l i + l j := by
      rw [sum_insert hjT, ← sum_erase_add S l hi]
      ring
    have := hmax T hTcard
    linarith

/-- The kernel's test: the minimum over selected logits is at least the maximum over
rejected ones (vacuous if nothing is rejected). -/
theorem prefix_iff_min_ge_max (l : V → ℝ) (S : Finset V) (hS : S.Nonempty)
    (hR : (univ \ S).Nonempty) :
    IsPrefix l S ↔ (univ \ S).sup' hR l ≤ S.inf' hS l := by
  constructor
  · intro h
    exact sup'_le hR l fun j hj => le_inf' hS l fun i hi => h i hi j (mem_sdiff.mp hj).2
  · intro h i hi j hj
    exact (le_sup' l (mem_sdiff.mpr ⟨mem_univ j, hj⟩)).trans (h.trans (inf'_le l hi))

end PCK
