⭐ COSMOSIM EDUCATION MODULE ROADMAP
(Phased, Promptable, Dependency-Aware, Implementation-Ready)

The roadmap is divided into:

Phases (E1–E8) — aligned with educational flow

Feature Blocks — what gets implemented

Technical Requirements — what files/engines change

Prompt Strategy — how to request it from the IDE

Deliverables — what you should expect to get back

Prerequisites — what must exist beforehand

Let’s begin.

🌟 PHASE E1 — Core UI Framework & Scenario Loader

Goal: A feature-complete UI shell so later scenarios are plug-and-play.

Feature Blocks
E1.1 — Concept Navigation Panel

left sidebar listing educational modules

collapsible sections: Concepts, Simulators, Diagnostics

button to load each scenario

highlight active scenario

E1.2 — Scenario Loader System

add “scenario configs” in a new folder:

education_scenarios/
    flat_space.json
    curved_space.json
    torus_world.json
    expansion_basic.json
    lightpaths_demo.json


viewer reads config and applies:

UniverseConfig overrides

overlay flags

camera positioning

environment toggles

E1.3 — Explainer HUD Framework

top or bottom panel

supports:

text

step numbers

opacity fading

next/previous explanation steps

E1.4 — Reset + Next Scenario Buttons
Technical Requirements

Modify test.html or main viewer to add UI elements

Add a new scenario_loader.js

Create scenario config schema

UI → calls → scenarioLoader.applyConfig()

Prompt Strategy for the IDE

You will eventually say:

“Implement E1.1: Concept Navigation Panel.
Add HTML/CSS/JS for a left sidebar with nested buttons that load scenario configs.”

“Implement E1.2: Scenario Loader System.
Create scenario_loader.js with loadScenario(name) that reads a JSON config and applies it to the viewer.”

etc.

Each E1 block is ~150–300 tokens → perfect for one IDE action.

Deliverables

UI shell

Config loader

Explainer HUD skeleton

Ready for all future scenarios

🌟 PHASE E2 — Light Path Demonstrator (Simple Photons)

Goal: An immediately compelling educational demo.

Feature Blocks
E2.1 — Massless Photon Entities

a new entity type:

mass=0

photon = true

moves at constant speed c (or scaled)

no gravitational attraction from photons

but photons are bent by gravitation (optional later)

E2.2 — Photon Launcher

UI button “emit photon”

click anywhere → photon spawns + direction arrow

E2.3 — Straight-line propagation

no substrate or expansion effects yet

just constant-velocity rays

E2.4 — Redshift Visualizer (simple)

photon has a wavelength property

expansion increases λ

E2.5 — Multi-photon tracer mode

trail lines

fade over time

Prompt Strategy

You ask for pieces:

“Implement E2.1: Add photon entity support to run_sim and viewer. Photons follow constant-speed direction vectors and ignore gravity.”

“Implement E2.4: Add simple redshift tracking based on scale factor expansion.”

Each self-contained.

🌟 PHASE E3 — Topology Explorer (High-Value Concept Module)

Goal: Let users understand torus, sphere, bubble through geodesics.

Feature Blocks
E3.1 — Geodesic Tracer Tool

user clicks → draws geodesic line

line wraps through torus

curves on sphere

reflects / curves in bubble interior

E3.2 — Boundary Teleport Visualizer

in torus mode: show teleport jumps

optional “ghost images” for wrap-around origins

E3.3 — Topology overlays

Already partially implemented, but now:

add label markers

coordinate grids that wrap seamlessly

Prompt Strategy

“Implement E3.1: Add geodesic tracer tool. A click emits a massless test particle whose path is drawn by line segments.”

“Implement E3.2: Add ghost-image visualization for torus wraparound.”

🌟 PHASE E4 — Expansion Explorer

Goal: Teach the most misunderstood concept in cosmology.

Feature Blocks
E4.1 — Comoving Grid Overlay

grid expands with scale factor disabled

camera remains fixed

E4.2 — Proper Grid Overlay

grid stretches in real space

shows different behavior

E4.3 — Hubble Flow Arrows

per-particle expansion vectors

magnitude ~ H × distance

E4.4 — Toggling Modes

no expansion

linear expansion

scale-factor expansion

bubble expansion

anisotropic expansion (later)

Prompt Strategy

“Implement E4.1: Add comoving/proper grid overlays that react to UniverseConfig.expansion_type.”

🌟 PHASE E5 — Curvature Explorer (Simple Newtonian Curvature)

Goal: Visualize how gravity bends space + light.

Feature Blocks
E5.1 — Scalar Curvature Map

Using Newtonian potential:

∇
2
Φ
=
4
𝜋
𝐺
𝜌
∇
2
Φ=4πGρ

Display as:

heatmap

contour lines

E5.2 — Photon bending

Use small-angle approximation:

Δ
𝑣
∝
−
∇
Φ
Δv∝−∇Φ
E5.3 — Curvature Overlay Toggle
Prompt Strategy

Ask for one block at a time:

“Implement E5.1 curvature map: compute Newtonian potential on a grid and display as heatmap overlay.”

🌟 PHASE E6 — Scalar Field Substrate (First Real Substrate Physics)

Goal: Start the substrate physics journey.

Feature Blocks
E6.1 — New physics_mode = 'FIELD'
E6.2 — Add lattice to UniverseState

phi[x,y]

phi_vel[x,y]

E6.3 — Klein-Gordon-like PDE
∂
𝑡
2
𝜙
=
𝑐
2
∇
2
𝜙
−
𝑚
2
𝜙
∂
t
2
	​

ϕ=c
2
∇
2
ϕ−m
2
ϕ
E6.4 — Viewer for field visualization

color heatmap

contour lines

vector arrows for gradients

E6.5 — Field ↔ entity coupling (optional later)
Prompt Strategy

“Implement E6.2: Add 2D scalar lattice fields to UniverseState with resolution configurable in UniverseConfig.”

🌟 PHASE E7 — Superfluid Substrate

Goal: Begin simulating exotic substrate physics.

Feature Blocks
E7.1 — Complex field state

psi_real
psi_imag

E7.2 — Simplified GPE integration
𝑖
∂
𝑡
𝜓
=
−
𝛼
∇
2
𝜓
+
𝛽
∣
𝜓
∣
2
𝜓
i∂
t
	​

ψ=−α∇
2
ψ+β∣ψ∣
2
ψ
E7.3 — Superfluid overlays

phase field (hue)

density field (brightness)

vortex detection

E7.4 — Expansion-coupled PDE
∇
2
→
1
𝑎
2
(
𝑡
)
∇
2
∇
2
→
a
2
(t)
1
	​

∇
2
Prompt Strategy

Each block is large → granular prompts like:

“Implement E7.1: Add complex field substrate and time stepping using explicit Euler (temporary) in kernel.update_superfluid.”

🌟 PHASE E8 — Superlattice Potential & Exotic Cosmology Tests

Goal: Realize the full “superfluid superlattice universe” concept.

Feature Blocks
E8.1 — Superlattice potential V(x,y)
𝑉
=
𝑉
0
cos
⁡
(
𝑘
1
𝑥
)
+
𝑉
1
cos
⁡
(
𝑘
2
𝑥
)
+
𝑉
2
cos
⁡
(
𝑘
3
𝑦
)
+
.
.
.
V=V
0
	​

cos(k
1
	​

x)+V
1
	​

cos(k
2
	​

x)+V
2
	​

cos(k
3
	​

y)+...
E8.2 — Add potential term to GPE
E8.3 — Diagnostics

vortex density

substrate power spectrum

stability index

E8.4 — Black-hole stress tests (vector+field)

drop particles into mass concentration

observe substrate reaction

detect divergence/instability

Prompt Strategy

High-complexity, but still discretizable:

“Implement E8.1: Add superlattice potential module generating multiscale V[x,y] grid based on parameters in UniverseConfig.”

⭐ COMPLEMENTARY TO ALL PHASES — Diagnostics System

This runs outside the educational module but is critical:

energy drift

momentum conservation

curvature statistics

redshift curves

wave spectrum analysis

field coherence lengths

Each diagnostic is a separate IDE prompt.

⭐ Summary: Stable Trajectory Toward Maximum Impact

This roadmap ensures:

quick wins early (UI + light + topology + expansion)

deep physics later (field → superfluid → superlattice)

strong compatibility with the existing CosmoSim engine

immediate YouTube demo readiness by Phase E2–E4

zero architectural dead-ends

everything is broken into AI IDE promptable units


⭐ COSMOSIM EDUCATION MODULE
FULL PREREQUISITE MATRIX

This will tell you:

what must already exist

what must be prepared

what must be refactored or stabilized

what new scaffolding must be created

when certain physics features must be implemented before later phases

This ensures we never run ahead of the engine’s capabilities.

⭐ GLOBAL PREREQUISITE LAYER

(Applies to ALL educational module phases)

These must be stable before we build ANY educational scenarios.

✔ PR1 — Working Simulation Loop

run_sim.py must produce safe, non-NaN frames.

Stabilization clamps must work.

Naming convention stable.

✔ PR2 — Cosmosim Viewer must be functional

You must be able to:

load JSON outputs

play frames

pause, seek

reset viewer state

switch simulations easily

This is mostly done.

✔ PR3 — Overlay System in Viewer

Already partially implemented. Must include:

grid

axes

bounds

topology overlays

Before Education Mode, we need:

reliable toggles

no UI conflicts

overlays synchronized with camera

✔ PR4 — Scenario Reset & Load APIs

Viewer must have:

initializePlayer(frames)

resetSimulation()

loadFramesFromJSON()

ability to apply config overrides

This is now partially working.

✔ PR5 — UniverseConfig MUST support overrides

Every educational scenario needs to override:

topology

expansion

substrate

dt

entity count

physics_mode

Your engine must safely handle:

missing params

unused fields

different physics modes

✔ PR6 — Safe Camera Behavior

Camera must:

center correctly

auto-rescale or at least not break for expansion

reset when new scenario loads

This prevents confusing visual artifacts.

✔ PR7 — Repository Folder Layout Locked In

You need a stable structure before adding scenarios:

/viewer
    test.html
    viewer.js
    overlays.js
    scenario_loader.js        (new)
    ui_components/            (new)

/education_scenarios
    flat_space.json
    torus_world.json
    expansion_basic.json
    ... etc

/sim_output
    (gitignored)

✔ PR8 — Core Physics Modes Stable

These physics systems must be stable enough for educational demos:

vector substrate (N-body)

topology engine

expansion engine

No need for fields or superfluids yet.

⭐ PER-PHASE PREREQUISITES

Now the critical piece: What each phase requires BEFORE you attempt it.

This is where the roadmap becomes development-friendly.

🌟 PHASE E1 — UI Framework & Scenario Loader
PREREQUISITES:
✔ PR1 – Sim loop stable
✔ PR2 – Viewer functional
✔ PR3 – Overlays stable
✔ PR4 – Reset/Load API in viewer
✔ PR5 – UniverseConfig override-safe
✔ PR6 – Camera resets correctly
✔ PR7 – Repo structure ready

No physics prereqs — this is pure UI.

🌟 PHASE E2 — Light Path Demonstrator
PREREQUISITES:
✔ E1 completed (UI navigation + scenario loader)
✔ PR5 — UniverseConfig override-safe
✔ PR8 — N-body physics stable
PLUS:
✔ PR9 — Entity Renderer must support new entity types

viewer must differentiate photons from particles

photon rendering style (line, glow, small dot)

✔ PR10 — Basic line drawing overlay

Photons leave trails → must have line segment rendering system.

🌟 PHASE E3 — Topology Explorer
PREREQUISITES:
✔ E1 (navigation + loader)
✔ E2 (photon path drawing)
✔ PR3 — Topology overlays working
✔ PR10 — Line/curve tracer working
✔ PR11 — Reliable distance + wrap functions

You already have this in topology engine — must confirm behavior.

Extra:

✔ PR12 — Geodesic tracer scaffolding

ability to simulate a test particle in one step without physics loop

or to override gravitational forces

🌟 PHASE E4 — Expansion Explorer
PREREQUISITES:
✔ E3
✔ PR3 — Overlays
✔ PR6 — Camera must not break during expansion
✔ PR13 — Expansion engine must be stable

linear

scale-factor

bubble

✔ PR14 — Grid overlays must scale correctly

This ensures:

comoving grid

proper grid

Hubble arrows

can animate independently.

🌟 PHASE E5 — Curvature Explorer
PREREQUISITES:
✔ E4 (expansion)
✔ PR15 — Newtonian gravity stable

before we visualize curvature.

✔ PR16 — 2D curvature grid computation

ability to compute ∇²Φ on a grid

requires simple lattice baked into viewer or engine

✔ PR17 — Heatmap renderer

Viewer must render a color grid.

🌟 PHASE E6 — Scalar Field Substrate
PREREQUISITES:
✔ E5 (curvature)
✔ PR18 — Add lattice to UniverseState

(H, W) grid

dt, dx must be stable

boundaries must obey topology

✔ PR19 — PDE stepping system

separate from N-body

synchronous with main sim loop

safe dt constraints

🌟 PHASE E7 — Superfluid Substrate
PREREQUISITES:
✔ E6 — Scalar field substrate fully working
✔ PR20 — Complex number lattice

psi_real

psi_imag

✔ PR21 — Stable Laplacian operator

needed for nonlinear Schrödinger / GPE

✔ PR22 — Phase visualization system

hue-mapped based on angle

magnitude mapped to brightness

🌟 PHASE E8 — Superlattice Potential + Exotic Cosmology
PREREQUISITES:
✔ E7
✔ PR23 — Superlattice potential generator

consistent grid resolution

periodic or toroidal metric

multi-scale modulation (k1, k2, etc.)

✔ PR24 — Field–Entity coupling

integrate field gradients into particle forces

feed particle mass distribution into field potential

⭐ GLOBAL PREREQUISITES FOR YOUTUBE-READY DEMOS

To produce videos as soon as possible:

You need only:

PR1 → PR7

plus E1, E2, E3, E4

This yields:

flat universe demo

curved space demo

topology demo

bubble universe visual explanation

light path demo

expansion demo

Meaning:

⚡ You can start producing educational demos BEFORE doing any field or substrate physics.

⭐ Summary:

Below is the entire prerequisite structure condensed:

GLOBAL PREREQS (before any E-phase)

PR1–PR7

E1 — UI Framework

requires: PR1–PR7

E2 — Light Paths

requires: E1 + PR9–PR10

E3 — Topology Explorer

requires: E2 + PR11–PR12

E4 — Expansion Explorer

requires: E3 + PR13–PR14

E5 — Curvature Explorer

requires: E4 + PR15–PR17

E6 — Scalar Field

requires: E5 + PR18–PR19

E7 — Superfluid Field

requires: E6 + PR20–PR22

E8 — Superlattice + Exotic Cosmology

requires: E7 + PR23–PR24