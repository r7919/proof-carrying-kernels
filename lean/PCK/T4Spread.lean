/-
T4 (randomized detection), the Paley-Zygmund part: spread-out errors.

`PCK.T4` covers deviations with one large coordinate. Here the deviation D may be small in
every coordinate but large in total: if ||D||_2 >= 2T then at most 13/16 of all sign vectors
r satisfy |r . D| <= T, so K independent checks all pass for at most (13/16)^K of them.
This covers dropped work and low-precision partial sums, which spread error over a block.

The proof is counting over the sign cube {+-1}^b, by induction on b:
  * second moment:  sum_r (r . D)^2  = 2^b ||D||^2
  * fourth moment:  sum_r (r . D)^4 <= 3 * 2^b ||D||^4
and then Paley-Zygmund with Cauchy-Schwarz on the set where (r . D)^2 > ||D||^2 / 4.
-/
import PCK.T4

namespace PCK

open Finset

/-- Sum over the sign cube of dimension b + 1, split on the first sign. -/
theorem sum_cube_succ {b : ℕ} (F : (Fin (b + 1) → Bool) → ℝ) :
    ∑ r, F r = ∑ s : Bool, ∑ r : Fin b → Bool, F (Fin.cons s r) := by
  rw [← (Fin.consEquiv fun _ => Bool).sum_comp, Fintype.sum_prod_type]
  rfl

theorem signedSum_cons {b : ℕ} (D : Fin (b + 1) → ℝ) (s : Bool) (r : Fin b → Bool) :
    signedSum D (Fin.cons s r) = sgn s * D 0 + signedSum (fun i => D i.succ) r := by
  unfold signedSum
  rw [Fin.sum_univ_succ]
  simp only [Fin.cons_zero, Fin.cons_succ]

theorem card_cube (b : ℕ) : ((univ : Finset (Fin b → Bool)).card : ℝ) = 2 ^ b := by
  simp

/-- Second moment: `sum_r (r . D)^2 = 2^b * ||D||^2`. -/
theorem sum_signedSum_sq : ∀ (b : ℕ) (D : Fin b → ℝ),
    ∑ r, signedSum D r ^ 2 = 2 ^ b * ∑ i, D i ^ 2
  | 0, D => by simp [signedSum]
  | b + 1, D => by
    rw [sum_cube_succ, Fintype.sum_bool, ← sum_add_distrib]
    simp only [signedSum_cons, sgn, ite_true, Bool.false_eq_true, ite_false]
    have key : ∀ X : ℝ, (1 * D 0 + X) ^ 2 + (-1 * D 0 + X) ^ 2 = 2 * D 0 ^ 2 + 2 * X ^ 2 := by
      intro X; ring
    simp only [key, sum_add_distrib, ← mul_sum, sum_const, card_univ, Fintype.card_fun,
      Fintype.card_bool, Fintype.card_fin, nsmul_eq_mul]
    rw [sum_signedSum_sq b, Fin.sum_univ_succ]
    push_cast
    ring

/-- Fourth moment: `sum_r (r . D)^4 <= 3 * 2^b * ||D||^4`. -/
theorem sum_signedSum_pow_four_le : ∀ (b : ℕ) (D : Fin b → ℝ),
    ∑ r, signedSum D r ^ 4 ≤ 3 * 2 ^ b * (∑ i, D i ^ 2) ^ 2
  | 0, D => by simp [signedSum]
  | b + 1, D => by
    rw [sum_cube_succ, Fintype.sum_bool, ← sum_add_distrib]
    simp only [signedSum_cons, sgn, ite_true, Bool.false_eq_true, ite_false]
    have key : ∀ X : ℝ, (1 * D 0 + X) ^ 4 + (-1 * D 0 + X) ^ 4 =
        2 * D 0 ^ 4 + 12 * D 0 ^ 2 * X ^ 2 + 2 * X ^ 4 := by
      intro X; ring
    simp only [key, sum_add_distrib, ← mul_sum, sum_const, card_univ, Fintype.card_fun,
      Fintype.card_bool, Fintype.card_fin, nsmul_eq_mul]
    have ih := sum_signedSum_pow_four_le b fun i => D i.succ
    rw [sum_signedSum_sq b, Fin.sum_univ_succ]
    set q := ∑ i : Fin b, D i.succ ^ 2
    have hq : 0 ≤ q := sum_nonneg fun i _ => sq_nonneg _
    have hd : 0 ≤ D 0 ^ 2 := sq_nonneg _
    have h2 : (0 : ℝ) < 2 ^ b := by positivity
    have h4 : 0 ≤ 2 ^ b * D 0 ^ 4 := by positivity
    push_cast
    rw [show (2 : ℝ) ^ (b + 1) = 2 * 2 ^ b by ring]
    nlinarith [ih, h4]

variable {b : ℕ}

/-- Paley-Zygmund on the sign cube: if `||D||_2 >= 2T`, at most 13/16 of all sign vectors
satisfy `|r . D| <= T`. -/
theorem card_passing_spread (D : Fin b → ℝ) (T : ℝ) (hT : 0 < T)
    (hD : 4 * T ^ 2 ≤ ∑ i, D i ^ 2) :
    ((passing D T).card : ℝ) * 16 ≤ 13 * 2 ^ b := by
  set q := ∑ i, D i ^ 2
  have hq : 0 < q := by nlinarith
  set Z : (Fin b → Bool) → ℝ := fun r => signedSum D r ^ 2
  set A := univ.filter fun r => q / 4 < Z r
  -- Passing vectors have small Z, so they lie outside A.
  have hPA : passing D T ⊆ univ \ A := by
    intro r hr
    have h := (mem_filter.mp hr).2
    refine mem_sdiff.mpr ⟨mem_univ r, fun hA => ?_⟩
    have hZ := (mem_filter.mp hA).2
    have : Z r ≤ T ^ 2 := by
      simp only [Z]
      have := abs_le.mp h
      nlinarith [sq_abs (signedSum D r), abs_nonneg (signedSum D r)]
    linarith
  -- Second moment: most of the mass of Z sits on A.
  have hm2 : ∑ r, Z r = 2 ^ b * q := sum_signedSum_sq b D
  have hout : ∑ r ∈ univ \ A, Z r ≤ 2 ^ b * (q / 4) := by
    calc ∑ r ∈ univ \ A, Z r ≤ ∑ _r ∈ univ \ A, q / 4 :=
          sum_le_sum fun r hr => le_of_not_gt fun h =>
            (mem_sdiff.mp hr).2 (mem_filter.mpr ⟨mem_univ r, h⟩)
      _ ≤ ∑ _r : Fin b → Bool, q / 4 :=
          sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => by positivity
      _ = 2 ^ b * (q / 4) := by rw [sum_const, card_univ]; simp
  have hsplit := sum_sdiff (subset_univ A) (f := Z)
  have hinA : 3 / 4 * (2 ^ b * q) ≤ ∑ r ∈ A, Z r := by linarith
  -- Cauchy-Schwarz and the fourth moment bound the mass on A by |A|.
  have hcs := sum_mul_sq_le_sq_mul_sq A (fun _ => (1 : ℝ)) Z
  simp only [one_mul, one_pow, sum_const, nsmul_eq_mul, mul_one] at hcs
  have hZ2 : ∑ r ∈ A, Z r ^ 2 ≤ 3 * 2 ^ b * q ^ 2 := by
    calc ∑ r ∈ A, Z r ^ 2 ≤ ∑ r, Z r ^ 2 :=
          sum_le_sum_of_subset_of_nonneg (subset_univ _) fun _ _ _ => sq_nonneg _
      _ = ∑ r, signedSum D r ^ 4 := sum_congr rfl fun r _ => by simp only [Z]; ring
      _ ≤ 3 * 2 ^ b * q ^ 2 := sum_signedSum_pow_four_le b D
  have h2b : (0 : ℝ) < 2 ^ b := by positivity
  have hAcard : 3 / 16 * 2 ^ b ≤ (A.card : ℝ) := by
    have h1 : (3 / 4 * (2 ^ b * q)) ^ 2 ≤ A.card * (3 * 2 ^ b * q ^ 2) := by
      have h0 : 0 ≤ 3 / 4 * (2 ^ b * q) := by positivity
      calc (3 / 4 * (2 ^ b * q)) ^ 2 ≤ (∑ r ∈ A, Z r) ^ 2 := pow_le_pow_left₀ h0 hinA 2
        _ ≤ A.card * ∑ r ∈ A, Z r ^ 2 := hcs
        _ ≤ A.card * (3 * 2 ^ b * q ^ 2) :=
            mul_le_mul_of_nonneg_left hZ2 (Nat.cast_nonneg _)
    have hpos : 0 < 2 ^ b * q ^ 2 := by positivity
    have h1' : (3 / 16 * 2 ^ b) * (3 * (2 ^ b * q ^ 2)) ≤ A.card * (3 * (2 ^ b * q ^ 2)) := by
      nlinarith
    exact le_of_mul_le_mul_right h1' (by positivity)
  have hcardA : ((univ \ A).card : ℝ) + A.card = 2 ^ b := by
    have := card_sdiff_add_card_eq_card (subset_univ A)
    rw [card_univ] at this
    exact_mod_cast (by simpa using this)
  have hP : ((passing D T).card : ℝ) ≤ (univ \ A).card := by exact_mod_cast card_le_card hPA
  linarith

/-- With `K` independent sign vectors, a spread-out deviation passes every check for at
most a `(13/16)^K` fraction of all `K`-tuples. -/
theorem card_all_pass_spread {K : ℕ} (D : Fin b → ℝ) (T : ℝ) (hT : 0 < T)
    (hD : 4 * T ^ 2 ≤ ∑ i, D i ^ 2) :
    (Fintype.piFinset fun _ : Fin K => passing D T).card * 16 ^ K ≤ 13 ^ K * 2 ^ (b * K) := by
  have h1 : (passing D T).card * 16 ≤ 13 * 2 ^ b := by
    exact_mod_cast card_passing_spread D T hT hD
  rw [Fintype.card_piFinset, prod_const, card_univ, Fintype.card_fin, ← mul_pow, pow_mul,
    ← mul_pow]
  exact Nat.pow_le_pow_left h1 K

section Computed

variable {n : ℕ}

/-- T4 for spread-out deviations, as computed: if the deviation of `y` from the exact
product has `||D||_2 >= 2 (tau + E + u (a . |x|))`, the computed block check passes for at
most 13/16 of all sign vectors, and all of `K` independent checks for at most `(13/16)^K`. -/
theorem t4_spread_computed {u : ℝ} (hu : 0 ≤ u) {K : ℕ}
    (W : Fin b → Fin n → ℝ) (x : Fin n → ℝ) (y : Fin b → ℝ)
    (c : (Fin b → Bool) → Fin n → ℝ)
    (hc : ∀ r j, |c r j - ∑ i, sgn (r i) * W i j| ≤ u * |∑ i, sgn (r i) * W i j|)
    (a : Fin n → ℝ) (ha : ∀ j, ∑ i, |W i j| ≤ a j)
    (rhoHat : (Fin b → Bool) → ℝ) {E tau : ℝ}
    (herr : ∀ r, |rhoHat r - (∑ j, c r j * x j - ∑ i, sgn (r i) * y i)| ≤ E)
    (hpos : 0 < tau + E + u * ∑ j, a j * |x j|)
    (hspread : 4 * (tau + E + u * ∑ j, a j * |x j|) ^ 2 ≤
      ∑ i, (y i - ∑ j, W i j * x j) ^ 2) :
    ((univ.filter fun r => |rhoHat r| ≤ tau).card : ℝ) * 16 ≤ 13 * 2 ^ b ∧
      (univ.filter fun R : Fin K → (Fin b → Bool) => ∀ k, |rhoHat (R k)| ≤ tau).card * 16 ^ K
        ≤ 13 ^ K * 2 ^ (b * K) := by
  have hsub := computed_pass_subset hu W x y c hc a ha rhoHat herr (tau := tau)
  refine ⟨?_, ?_⟩
  · have h1 : ((univ.filter fun r => |rhoHat r| ≤ tau).card : ℝ) ≤
        (passing (fun i => y i - ∑ j, W i j * x j) (tau + E + u * ∑ j, a j * |x j|)).card := by
      exact_mod_cast card_le_card hsub
    have h2 := card_passing_spread _ _ hpos hspread
    linarith
  · have hsubK : (univ.filter fun R : Fin K → (Fin b → Bool) => ∀ k, |rhoHat (R k)| ≤ tau) ⊆
        Fintype.piFinset fun _ : Fin K =>
          passing (fun i => y i - ∑ j, W i j * x j) (tau + E + u * ∑ j, a j * |x j|) := by
      intro R hR
      rw [Fintype.mem_piFinset]
      exact fun k => hsub (mem_filter.mpr ⟨mem_univ _, (mem_filter.mp hR).2 k⟩)
    exact (Nat.mul_le_mul_right _ (card_le_card hsubK)).trans
      (card_all_pass_spread _ _ hpos hspread)

end Computed

end PCK
