/-
Membership checks for samplers that never build their selection set.

vLLM samples top-k / top-p with FlashInfer's rejection sampling: it returns a token, not a
set, so the set checks of `TopK.lean` and `TopP.lean` have nothing to check. What can be
checked is membership: is the sampled token `t` in a correct top-k set, and in the top-p
nucleus? (vLLM applies top-k first, then top-p over the survivors, renormalized.)

* `topK_member_iff`: `t` is in some correct top-k set (`TopKCheck`) iff fewer than `k`
  logits are strictly larger than its own. Both directions, so the count check is exact.
* `topP_member_sound` / `topP_member_detects`: with the checker's mass fraction `rhat`
  within `ε` of the true fraction `r` of kept mass strictly above `t`, and slack
  `σ ≥ tol + ε`, a sample meeting the sampler's own guarantee (`r < p + tol`) always
  passes, and a rejected sample has `r ≥ p + tol`: a violation beyond the tolerance.
-/
import PCK.TopP

namespace PCK

open Finset

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The tokens whose logit is strictly larger than `t`'s. -/
noncomputable def above (l : V → ℝ) (t : V) : Finset V := univ.filter (fun j => l t < l j)

/-- `t` with every token above it is a prefix of the logits. -/
theorem prefix_insert_above (l : V → ℝ) (t : V) : IsPrefix l (insert t (above l t)) := by
  intro i hi j hj
  simp only [above, mem_insert, mem_filter, mem_univ, true_and, not_or, not_lt] at hi hj
  rcases hi with rfl | hi
  · exact hj.2
  · linarith [hj.2]

/-- A prefix extends to a prefix of any larger size, adding a largest remaining logit
at a time. -/
theorem exists_prefix_superset (l : V → ℝ) :
    ∀ n : ℕ, ∀ A : Finset V, IsPrefix l A → A.card + n ≤ Fintype.card V →
      ∃ S, A ⊆ S ∧ S.card = A.card + n ∧ IsPrefix l S := by
  intro n
  induction n with
  | zero => intro A hA _; exact ⟨A, subset_refl A, by simp, hA⟩
  | succ n ih =>
    intro A hA hcard
    have hlt : A.card < (univ : Finset V).card := by rw [card_univ]; omega
    have hne : (univ \ A).Nonempty := by
      rw [← card_pos, card_sdiff_of_subset (subset_univ A)]
      omega
    obtain ⟨m, hm, hmax⟩ := exists_max_image (univ \ A) l hne
    have hmA : m ∉ A := (mem_sdiff.mp hm).2
    have hpre : IsPrefix l (insert m A) := by
      intro i hi j hj
      have hjA : j ∉ A := fun h => hj (mem_insert_of_mem h)
      rcases mem_insert.mp hi with rfl | hi
      · exact hmax j (mem_sdiff.mpr ⟨mem_univ j, hjA⟩)
      · exact hA i hi j hjA
    have hc : (insert m A).card = A.card + 1 := card_insert_of_notMem hmA
    obtain ⟨S, hsub, hS, hSpre⟩ := ih (insert m A) hpre (by omega)
    exact ⟨S, (subset_insert m A).trans hsub, by omega, hSpre⟩

/-- **Top-k membership is exactly a count**: `t` belongs to some correct top-k set iff
fewer than `k` logits are strictly larger than `t`'s (and `k` tokens exist). -/
theorem topK_member_iff (l : V → ℝ) (k : ℕ) (t : V) :
    (∃ S, TopKCheck l k S ∧ t ∈ S) ↔ (above l t).card < k ∧ k ≤ Fintype.card V := by
  constructor
  · rintro ⟨S, ⟨hk, hpre⟩, ht⟩
    have hsub : above l t ⊆ S.erase t := by
      intro j hj
      have hlt : l t < l j := (mem_filter.mp hj).2
      refine mem_erase.mpr ⟨fun h => by rw [h] at hlt; exact lt_irrefl _ hlt, ?_⟩
      by_contra hjS
      exact absurd (hpre t ht j hjS) (not_le.mpr hlt)
    have h1 := card_le_card hsub
    rw [card_erase_of_mem ht] at h1
    have hpos : 0 < S.card := card_pos.mpr ⟨t, ht⟩
    refine ⟨by omega, ?_⟩
    rw [← hk, ← card_univ]
    exact card_le_card (subset_univ S)
  · rintro ⟨hlt, hk⟩
    set A := insert t (above l t)
    have hA : A.card ≤ k := (card_insert_le t _).trans (by omega)
    obtain ⟨S, hsub, hS, hSpre⟩ :=
      exists_prefix_superset l (k - A.card) A (prefix_insert_above l t) (by omega)
    exact ⟨S, ⟨by omega, hSpre⟩, hsub (mem_insert_self t _)⟩

/-- vLLM's top-p after top-k: with the kept set `K` and weights `q`, `t` survives iff the
mass of kept tokens strictly above it is less than `p` of the kept mass. -/
def TopPMember (l q : V → ℝ) (K : Finset V) (p : ℝ) (t : V) : Prop :=
  t ∈ K ∧ mass q (K.filter (fun j => l t < l j)) < p * mass q K

/-- The true fraction of kept mass strictly above `t`. -/
noncomputable def massAbove (l q : V → ℝ) (K : Finset V) (t : V) : ℝ :=
  mass q (K.filter (fun j => l t < l j)) / mass q K

omit [Fintype V] [DecidableEq V] in
theorem topP_member_iff_fraction {l q : V → ℝ} {K : Finset V} {p : ℝ} {t : V}
    (hK : 0 < mass q K) : TopPMember l q K p t ↔ t ∈ K ∧ massAbove l q K t < p := by
  unfold TopPMember massAbove
  rw [div_lt_iff₀ hK]

omit [Fintype V] [DecidableEq V] in
/-- **Top-p membership, no false alarms**: a sample that meets the sampler's own guarantee
within `tol` passes the computed check `rhat < p + σ` whenever the slack covers `tol` plus
the checker's error `ε`. -/
theorem topP_member_sound {l q : V → ℝ} {K : Finset V} {p σ ε tol rhat : ℝ} {t : V}
    (hr : massAbove l q K t < p + tol) (hε : |rhat - massAbove l q K t| ≤ ε)
    (hσ : tol + ε ≤ σ) : rhat < p + σ := by
  have := (abs_le.mp hε).2
  linarith

omit [Fintype V] [DecidableEq V] in
/-- **Top-p membership, detection**: a rejected sample's true fraction of kept mass above
it is at least `p + tol`, so it is outside the nucleus by more than the sampler's own
tolerance (in particular not a `TopPMember` when `0 ≤ tol`). -/
theorem topP_member_detects {l q : V → ℝ} {K : Finset V} {p σ ε tol rhat : ℝ} {t : V}
    (hrej : p + σ ≤ rhat) (hε : |rhat - massAbove l q K t| ≤ ε) (hσ : tol + ε ≤ σ)
    (htol : 0 ≤ tol) (hK : 0 < mass q K) :
    p + tol ≤ massAbove l q K t ∧ ¬ TopPMember l q K p t := by
  have h := (abs_le.mp hε).2
  have hge : p + tol ≤ massAbove l q K t := by linarith
  refine ⟨hge, fun hm => ?_⟩
  have := ((topP_member_iff_fraction hK).mp hm).2
  linarith

end PCK
