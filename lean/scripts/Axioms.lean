/-
Axiom audit, run in CI after `lake build`:  lake env lean scripts/Axioms.lean
Fails if any declaration in the `PCK` namespace depends on an axiom other than Lean's
three standard ones. That includes `sorryAx`, so a proof with `sorry` cannot pass.
-/
import PCK

open Lean Elab Command

#eval show CommandElabM Unit from do
  let env ← getEnv
  let standard := [``propext, ``Classical.choice, ``Quot.sound]
  let mut checked : Nat := 0
  let mut bad : Array (Name × Name) := #[]
  for (name, _) in env.constants.toList do
    if (`PCK).isPrefixOf name && !name.isInternal then
      checked := checked + 1
      for ax in ← liftCoreM (collectAxioms name) do
        unless standard.contains ax do
          bad := bad.push (name, ax)
  if checked == 0 then
    throwError "no PCK declarations found"
  unless bad.isEmpty do
    throwError m!"unexpected axioms: {bad.toList}"
  logInfo m!"axiom audit passed: {checked} PCK declarations use only standard axioms"
