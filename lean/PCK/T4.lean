/-
T4 (randomized detection), the 2^-K part.

The checker's signs r ∈ {±1}^b are drawn uniformly at random after the output y is fixed
(commit-then-challenge in evaluation mode; any fixed fault in production). Let D be any
deviation vector, faulty or even adversarial. If one coordinate is large, |D i₀| > T, then
at most half of all sign vectors r satisfy |r · D| ≤ T: pairing r with r flipped at i₀
changes r · D by exactly 2 |D i₀| > 2T, so at most one of each pair can pass. With K
independent sign vectors, at most a 2^-K fraction of all K-tuples passes every check.

Probabilities are stated as counts over the finite sign cube, so no measure theory is needed:
`card (passing) * 2 ≤ 2^b` for one check and `card (passing K-tuples) * 2^K ≤ 2^(b K)`.

`t4_computed` connects this to the checker as computed: a block passing with signs r forces
|r · D| ≤ tau_hat + E + u * (a . |x|), by the residual split of T1.
-/
import PCK.T1

namespace PCK

open Finset

variable {b : ℕ}

/-- The sign encoded by a bit: `true ↦ 1`, `false ↦ -1`. -/
def sgn (s : Bool) : ℝ := if s then 1 else -1

theorem sgn_not (s : Bool) : sgn (!s) = -sgn s := by cases s <;> simp [sgn]

theorem sgn_eq (s : Bool) : sgn s = 1 ∨ sgn s = -1 := by cases s <;> simp [sgn]

/-- `r · D` for a sign vector given as bits. -/
noncomputable def signedSum (D : Fin b → ℝ) (r : Fin b → Bool) : ℝ := ∑ i, sgn (r i) * D i

/-- Flipping one sign changes `r · D` by exactly `2 * sgn (r i₀) * D i₀`. -/
theorem signedSum_flip (D : Fin b → ℝ) (r : Fin b → Bool) (i₀ : Fin b) :
    signedSum D r - signedSum D (Function.update r i₀ (!r i₀)) = 2 * sgn (r i₀) * D i₀ := by
  unfold signedSum
  rw [← add_sum_erase _ _ (mem_univ i₀), ← add_sum_erase _ _ (mem_univ i₀)]
  have hrest : ∑ i ∈ univ.erase i₀, sgn (Function.update r i₀ (!r i₀) i) * D i =
      ∑ i ∈ univ.erase i₀, sgn (r i) * D i :=
    sum_congr rfl fun i hi => by rw [Function.update_of_ne (ne_of_mem_erase hi)]
  rw [hrest, Function.update_self, sgn_not]
  ring

/-- The sign vectors with which a single check on deviation `D` can pass at level `T`. -/
noncomputable def passing (D : Fin b → ℝ) (T : ℝ) : Finset (Fin b → Bool) :=
  univ.filter fun r => |signedSum D r| ≤ T

/-- At most half of all sign vectors pass when one coordinate exceeds `T`. -/
theorem card_passing_mul_two_le (D : Fin b → ℝ) (T : ℝ) (i₀ : Fin b) (hbig : T < |D i₀|) :
    (passing D T).card * 2 ≤ 2 ^ b := by
  set flip : (Fin b → Bool) → (Fin b → Bool) := fun r => Function.update r i₀ (!r i₀)
  have hinj : Function.Injective flip := by
    intro r s h
    have h' := congrArg flip h
    simpa [flip, Function.update_idem, Function.update_eq_self] using h'
  have hdisj : Disjoint (passing D T) ((passing D T).image flip) := by
    rw [disjoint_left]
    intro r hr hr'
    obtain ⟨s, hs, rfl⟩ := mem_image.mp hr'
    have h1 := (mem_filter.mp hr).2
    have h2 := (mem_filter.mp hs).2
    have hd := signedSum_flip D s i₀
    have hflip : Function.update (flip s) i₀ (!(flip s) i₀) = s := by
      simp [flip, Function.update_idem, Function.update_eq_self]
    -- |sum s - sum (flip s)| = 2 |D i₀| > 2T, but both sums are at most T in size.
    have habs : |signedSum D s - signedSum D (flip s)| = 2 * |D i₀| := by
      rw [hd, abs_mul, abs_mul]
      rcases sgn_eq (s i₀) with h | h <;> simp [h]
    have := abs_sub (signedSum D s) (signedSum D (flip s))
    linarith
  have hunion := card_union_of_disjoint hdisj
  rw [card_image_of_injective _ hinj] at hunion
  have hle : (passing D T ∪ (passing D T).image flip).card ≤ 2 ^ b := by
    calc _ ≤ (univ : Finset (Fin b → Bool)).card := card_le_univ _
      _ = 2 ^ b := by simp
  linarith

/-- With `K` independent sign vectors, at most a `2^-K` fraction of all `K`-tuples passes
every check. -/
theorem card_all_pass_le {K : ℕ} (D : Fin b → ℝ) (T : ℝ) (i₀ : Fin b) (hbig : T < |D i₀|) :
    (Fintype.piFinset fun _ : Fin K => passing D T).card * 2 ^ K ≤ 2 ^ (b * K) := by
  rw [Fintype.card_piFinset, prod_const, card_univ, Fintype.card_fin, ← mul_pow, pow_mul]
  exact Nat.pow_le_pow_left (card_passing_mul_two_le D T i₀ hbig) K

section Computed

variable {n : ℕ}

/-- A computed pass with signs `r` forces `|r . D| <= tau + E + u * (a . |x|)`, where `D` is
the deviation of `y` from the exact product. -/
theorem computed_pass_subset {u : ℝ} (hu : 0 ≤ u)
    (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ) (y : Fin b → ℝ)
    (c : (Fin b → Bool) → Fin n → ℝ)
    (hc : ∀ r j, |c r j - ∑ i, sgn (r i) * W i j| ≤ u * |∑ i, sgn (r i) * W i j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j)
    (rhoHat : (Fin b → Bool) → ℝ) {E tau : ℝ}
    (herr : ∀ r, |rhoHat r - (∑ j, c r j * x j - ∑ i, sgn (r i) * y i)| ≤ E) :
    (univ.filter fun r => |rhoHat r| ≤ tau) ⊆
      passing (fun i => y i - ∑ j, W i j * x j) (tau + E + u * ∑ j, a j * |x j|) := by
  intro r hr
  have hpass := (mem_filter.mp hr).2
  refine mem_filter.mpr ⟨mem_univ r, ?_⟩
  have hsplit := residual_split W x y (fun i => sgn (r i)) (c r)
  have hchk := checksum_term_le (x := x) hu W (r := fun i => sgn (r i))
    (fun i => sgn_eq (r i)) (c r) (hc r) a ha
  have he := herr r
  have hrho : |∑ j, c r j * x j - ∑ i, sgn (r i) * y i| ≤ tau + E := by
    have := abs_sub_abs_le_abs_sub (∑ j, c r j * x j - ∑ i, sgn (r i) * y i) (rhoHat r)
    rw [abs_sub_comm] at he
    linarith
  rw [hsplit] at hrho
  have key := abs_sub_abs_le_abs_sub (∑ i, sgn (r i) * (y i - ∑ j, W i j * x j))
    (∑ j, (c r j - ∑ i, sgn (r i) * W i j) * x j)
  rw [abs_sub_comm] at key
  change |∑ i, sgn (r i) * (y i - ∑ j, W i j * x j)| ≤ _
  linarith

/-- T4 for the check as computed. Whatever the output `y`, if one output deviates from its
exact value by more than `tau + E + u * (a . |x|)`, the computed block check passes for at
most half of all sign vectors, and for at most a `2^-K` fraction of `K` independent ones.
`c r` is the stored checksum for signs `r` and `rhoHat r` the computed residual, within `E`
of the exact one. -/
theorem t4_computed {u : ℝ} (hu : 0 ≤ u) {K : ℕ}
    (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ) (y : Fin b → ℝ)
    (c : (Fin b → Bool) → Fin n → ℝ)
    (hc : ∀ r j, |c r j - ∑ i, sgn (r i) * W i j| ≤ u * |∑ i, sgn (r i) * W i j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j)
    (rhoHat : (Fin b → Bool) → ℝ) {E tau : ℝ}
    (herr : ∀ r, |rhoHat r - (∑ j, c r j * x j - ∑ i, sgn (r i) * y i)| ≤ E)
    (i₀ : Fin b)
    (hbig : tau + E + u * ∑ j, a j * |x j| < |y i₀ - ∑ j, W i₀ j * x j|) :
    (univ.filter fun r => |rhoHat r| ≤ tau).card * 2 ≤ 2 ^ b ∧
      (univ.filter fun R : Fin K → (Fin b → Bool) => ∀ k, |rhoHat (R k)| ≤ tau).card * 2 ^ K
        ≤ 2 ^ (b * K) := by
  have hsub := computed_pass_subset hu W x y c hc a ha rhoHat herr (tau := tau)
  refine ⟨(Nat.mul_le_mul_right 2 (card_le_card hsub)).trans
    (card_passing_mul_two_le _ _ i₀ hbig), ?_⟩
  have hsubK : (univ.filter fun R : Fin K → (Fin b → Bool) => ∀ k, |rhoHat (R k)| ≤ tau) ⊆
      Fintype.piFinset fun _ : Fin K =>
        passing (fun i => y i - ∑ j, W i j * x j) (tau + E + u * ∑ j, a j * |x j|) := by
    intro R hR
    rw [Fintype.mem_piFinset]
    exact fun k => hsub (mem_filter.mpr ⟨mem_univ _, (mem_filter.mp hR).2 k⟩)
  exact (Nat.mul_le_mul_right _ (card_le_card hsubK)).trans (card_all_pass_le _ _ i₀ hbig)

end Computed

end PCK
