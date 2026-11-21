# Workbench design notes

**Idea: a lab bench for water.** The Workbench is an instrument panel, not a form. One screen holds the whole task; what you
change on the left is visible in the middle and on the right at once.

## Identity

- Two signature hues carry the science everywhere (3D view, charts, chips, tiles): **aquamarine = solvent / occupied**,
  **coral = solute / the zone's anchor**. One periwinkle accent is reserved for interaction. Everything else is a cool neutral scale.
- Dark ("deep-sea instrument") is the default; light ("lab notebook paper") is one click away. Both come from the same tokens
  (`theme.py`), so the 3D scene, pyqtgraph charts and widgets always agree.
- Type: a display face for headings, a text face for UI, a monospaced face for every number, unit and selection, so digits align and
  a selection reads as code. Families are looked up on the system; DejaVu (shipped with matplotlib) is the guaranteed fallback.

## Interaction principles

- **Defaults first, everything adjustable.** The first screen after import already shows a zone (the ligand if there is one) and a
  preview. *Advanced* regions are collapsed but are never more than one click away; each field shows a dot when it differs from its default,
  a reset button, and the card offers **Reset all**.
- **Units are always visible**: in spin boxes (Å, ps), axis labels, tiles and the definition popups.
- **Live previews are labelled previews.** The occupancy preview under the 3D view is computed on ~80 sampled frames and says so;
  results never reuse a preview number.
- **Errors say how to recover.** Validation is attached to the control that caused it and offers the fix as a button where one exists.
- **Trust is a first-class output.** Every tile has a badge; the Trust panel lists caveats in plain language; the definition of every number is one click away.
- **Motion explains change**: numbers count to their new value, the stepper fills as steps complete, the 3D camera glides, frames
  that just entered/left the zone are highlighted. *Reduce motion* turns all of it off.
- **Nothing is hidden by size**: the layout is verified at 1366×768 (no horizontal scrolling, Run button, tiles and chart tabs all on screen).

## Accessibility notes

Colour is never the only carrier: trust uses icon + border + text, and in-zone / bulk solvent differ in size as well as colour in the 3D view.
Buttons and inputs show a focus ring and have tooltips. Text colours were measured (WCAG relative luminance) and adjusted so that every
text token reaches ≥ 4.5:1 on the card and page surfaces of both palettes (≥ 3:1 on the hover surface); the 3D renderer additionally
adjusts its own colour classes to keep ≥ 3:1 against its background. Not done: screen-reader semantics for the custom-painted stepper and 3D view,
and full keyboard navigation of the 3D scene beyond the keys the renderer handles. These are the known accessibility gaps.
