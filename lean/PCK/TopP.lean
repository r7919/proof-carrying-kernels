/-
The top-p (nucleus) sampling check.

The checker (`pck.sampling.check_topp`, `sampling_kernel.cu`) accepts a selection `S` of
indices into the logits `l` with probabilities `q` iff `S` is a prefix (every selected logit
is at least every rejected one), the computed mass of `S` is at least `p - σ`, and the
computed mass of `S` without its smallest-logit element `i` is below `p + σ`, where `σ` is
the slack.

* `topP_minimal`: with exact masses, an accepted set has the fewest elements of any set
  whose mass reaches `p` (any tie-breaking allowed).
* `topP_computed_minimal`: with computed masses within `ε` of the true ones, an accepted set
  reaches mass `p - σ - ε` and is no larger than any set of mass `p + σ + ε`.
* `topP_computed_sound`: a sampler whose own masses are within `tol` of the truth, checked
  with `σ ≥ tol + ε`, always passes: no false alarms.
* `softmax_nonneg`, `softmax_mono`: softmax probabilities satisfy the hypotheses on `q`.

That the checker's FP64 masses are within `ε` of the true softmax masses is assumed, not
proven (it rests on the accuracy of `exp`); see docs/design.md.
-/
import Mathlib.Tactic
import PCK.TopK

namespace PCK

open Finset

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- Probability mass of a set. -/
def mass (q : V → ℝ) (S : Finset V) : ℝ := ∑ v ∈ S, q v

/-- The exact top-p check: `S` is a prefix of the logits, `i ∈ S` has the smallest selected
logit, `S` reaches mass `p` and `S` without `i` does not. -/
def TopPCheck (l q : V → ℝ) (p : ℝ) (S : Finset V) (i : V) : Prop :=
  IsPrefix l S ∧ i ∈ S ∧ (∀ s ∈ S, l i ≤ l s) ∧ p ≤ mass q S ∧ mass q (S.erase i) < p

/-- The check as computed: `mhat` gives the checker's masses, `σ` is the slack. -/
def TopPCheckComputed (l : V → ℝ) (mhat : Finset V → ℝ) (p σ : ℝ) (S : Finset V) (i : V) :
    Prop :=
  IsPrefix l S ∧ i ∈ S ∧ (∀ s ∈ S, l i ≤ l s) ∧ p - σ ≤ mhat S ∧ mhat (S.erase i) < p + σ

omit [Fintype V] [DecidableEq V] in
/-- A smaller set of nonnegative values, each at most every value of a larger set. -/
theorem sum_le_sum_of_card_le {q : V → ℝ} (hq : ∀ v, 0 ≤ q v) {A B : Finset V}
    (hcard : A.card ≤ B.card) (hle : ∀ a ∈ A, ∀ b ∈ B, q a ≤ q b) :
    ∑ a ∈ A, q a ≤ ∑ b ∈ B, q b := by
  rcases A.eq_empty_or_nonempty with rfl | hA
  · simpa using sum_nonneg fun b _ => hq b
  · set c := A.sup' hA q
    have hc : 0 ≤ c := (hq _).trans (le_sup' q hA.choose_spec)
    have h1 : ∑ a ∈ A, q a ≤ A.card • c := sum_le_card_nsmul A q c fun a ha => le_sup' q ha
    have h2 : B.card • c ≤ ∑ b ∈ B, q b := card_nsmul_le_sum B q c fun b hb =>
      sup'_le hA q fun a ha => hle a ha b hb
    have h3 : A.card • c ≤ B.card • c := by
      simp only [nsmul_eq_mul]
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast hcard) hc
    linarith

omit [Fintype V] in
/-- An accepted set without its smallest element `i` has the largest mass of any set of at
most its size. -/
theorem mass_le_mass_erase {l q : V → ℝ} (hq : ∀ v, 0 ≤ q v)
    (hmono : ∀ a b, l a ≤ l b → q a ≤ q b) {S : Finset V} {i : V} (hpre : IsPrefix l S)
    (hmin : ∀ s ∈ S, l i ≤ l s) (T : Finset V) (hT : T.card ≤ (S.erase i).card) :
    mass q T ≤ mass q (S.erase i) := by
  set E := S.erase i
  -- `E` is a prefix for `q`: `i` and every rejected index are below all of `E`.
  have hE : ∀ e ∈ E, ∀ j ∉ E, q j ≤ q e := by
    intro e he j hj
    have heS := mem_of_mem_erase he
    by_cases hji : j = i
    · exact hmono _ _ (hji ▸ hmin e heS)
    · exact hmono _ _ (hpre e heS j fun hjS => hj (mem_erase.mpr ⟨hji, hjS⟩))
  have hcard : (T \ E).card ≤ (E \ T).card := by
    have h1 := card_sdiff_add_card_inter T E
    have h2 := card_sdiff_add_card_inter E T
    rw [inter_comm] at h2
    omega
  have hle : ∑ v ∈ T \ E, q v ≤ ∑ v ∈ E \ T, q v :=
    sum_le_sum_of_card_le hq hcard fun a ha b hb =>
      hE b (mem_sdiff.mp hb).1 a (mem_sdiff.mp ha).2
  have hTs := sum_inter_add_sum_sdiff T E q
  have hEs := sum_inter_add_sum_sdiff E T q
  rw [inter_comm] at hEs
  unfold mass
  linarith

omit [Fintype V] in
/-- Any set whose mass reaches `p'`, where `S` without `i` stays below it, is at least as
large as `S`. -/
theorem card_le_of_mass {l q : V → ℝ} (hq : ∀ v, 0 ≤ q v)
    (hmono : ∀ a b, l a ≤ l b → q a ≤ q b) {S : Finset V} {i : V} {p' : ℝ}
    (hpre : IsPrefix l S) (hi : i ∈ S) (hmin : ∀ s ∈ S, l i ≤ l s)
    (hE : mass q (S.erase i) < p') (T : Finset V) (hT : p' ≤ mass q T) : S.card ≤ T.card := by
  by_contra h
  have hcard : T.card ≤ (S.erase i).card := by
    rw [card_erase_of_mem hi]
    omega
  have := mass_le_mass_erase hq hmono hpre hmin T hcard
  linarith

omit [Fintype V] in
/-- **Top-p, exact masses**: an accepted set is a prefix reaching mass `p` with the fewest
elements of any set reaching `p`. -/
theorem topP_minimal {l q : V → ℝ} (hq : ∀ v, 0 ≤ q v)
    (hmono : ∀ a b, l a ≤ l b → q a ≤ q b) {p : ℝ} {S : Finset V} {i : V}
    (h : TopPCheck l q p S i) :
    IsPrefix l S ∧ p ≤ mass q S ∧ ∀ T : Finset V, p ≤ mass q T → S.card ≤ T.card := by
  obtain ⟨hpre, hi, hmin, hS, hE⟩ := h
  exact ⟨hpre, hS, card_le_of_mass hq hmono hpre hi hmin hE⟩

omit [Fintype V] in
/-- **Top-p, as computed**: if the checker's masses of `S` and `S` without `i` are within
`ε` of the true ones, an accepted set reaches mass `p - σ - ε` and is no larger than any set
of mass at least `p + σ + ε`. -/
theorem topP_computed_minimal {l q : V → ℝ} (hq : ∀ v, 0 ≤ q v)
    (hmono : ∀ a b, l a ≤ l b → q a ≤ q b) {mhat : Finset V → ℝ} {p σ ε : ℝ}
    {S : Finset V} {i : V} (hS : |mhat S - mass q S| ≤ ε)
    (hE : |mhat (S.erase i) - mass q (S.erase i)| ≤ ε) (h : TopPCheckComputed l mhat p σ S i) :
    IsPrefix l S ∧ p - σ - ε ≤ mass q S ∧
      ∀ T : Finset V, p + σ + ε ≤ mass q T → S.card ≤ T.card := by
  obtain ⟨hpre, hi, hmin, hlo, hhi⟩ := h
  have hS' := (abs_le.mp hS).2
  have hE' := (abs_le.mp hE).1
  refine ⟨hpre, by linarith, card_le_of_mass hq hmono hpre hi hmin ?_⟩
  linarith

omit [Fintype V] in
/-- **Top-p, no false alarms**: a prefix whose true masses meet the sampler's own guarantee
within `tol` passes the computed check whenever the slack covers `tol` plus the checker's
error `ε`. -/
theorem topP_computed_sound {l q : V → ℝ} {mhat : Finset V → ℝ} {p σ ε tol : ℝ}
    {S : Finset V} {i : V} (hpre : IsPrefix l S) (hi : i ∈ S) (hmin : ∀ s ∈ S, l i ≤ l s)
    (hlo : p - tol ≤ mass q S) (hhi : mass q (S.erase i) < p + tol)
    (hS : |mhat S - mass q S| ≤ ε) (hE : |mhat (S.erase i) - mass q (S.erase i)| ≤ ε)
    (hσ : tol + ε ≤ σ) : TopPCheckComputed l mhat p σ S i := by
  have hS' := (abs_le.mp hS).1
  have hE' := (abs_le.mp hE).2
  exact ⟨hpre, hi, hmin, by linarith, by linarith⟩

/-- Softmax of logits `l` at temperature `t`. -/
noncomputable def softmax (t : ℝ) (l : V → ℝ) (v : V) : ℝ :=
  Real.exp (l v / t) / ∑ w, Real.exp (l w / t)

omit [DecidableEq V] in
theorem softmax_nonneg (t : ℝ) (l : V → ℝ) (v : V) : 0 ≤ softmax t l v :=
  div_nonneg (Real.exp_pos _).le (sum_nonneg fun _ _ => (Real.exp_pos _).le)

omit [DecidableEq V] in
/-- Softmax preserves the order of the logits, so a prefix of the logits is a prefix of the
probabilities and the smallest selected logit has the smallest selected probability. -/
theorem softmax_mono {t : ℝ} (ht : 0 < t) (l : V → ℝ) (a b : V) (h : l a ≤ l b) :
    softmax t l a ≤ softmax t l b := by
  unfold softmax
  apply div_le_div_of_nonneg_right _ (sum_nonneg fun _ _ => (Real.exp_pos _).le)
  exact Real.exp_le_exp.mpr (div_le_div_of_nonneg_right h ht.le)

/-- `topP_computed_minimal` for softmax probabilities at a positive temperature. -/
theorem topP_softmax_computed {t : ℝ} (ht : 0 < t) {l : V → ℝ} {mhat : Finset V → ℝ}
    {p σ ε : ℝ} {S : Finset V} {i : V} (hS : |mhat S - mass (softmax t l) S| ≤ ε)
    (hE : |mhat (S.erase i) - mass (softmax t l) (S.erase i)| ≤ ε)
    (h : TopPCheckComputed l mhat p σ S i) :
    IsPrefix l S ∧ p - σ - ε ≤ mass (softmax t l) S ∧
      ∀ T : Finset V, p + σ + ε ≤ mass (softmax t l) T → S.card ≤ T.card :=
  topP_computed_minimal (softmax_nonneg t l) (softmax_mono ht l) hS hE h

end PCK
