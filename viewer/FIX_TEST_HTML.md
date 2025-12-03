# E1.4 Test.html Fix Instructions

## Problem
The test.html file has a duplicated/nested scenario panel that needs to be removed.

## Fix Steps

### 1. Find and Delete Lines 203-239

In `viewer/test.html`, **DELETE** these lines (approximately lines 203-239):

```html
            <!-- Scenario Configuration Panel -->
            <section id="scenarioPanel" class="panel" style="...">
                ...
            </section>
```

This is the **INNER/DUPLICATE** panel that's incorrectly nested inside `<div id="paramsContainer">`.

### 2. Fix Line 201

Change line 201 from:
```html
        <div id="paramsContainer">
```

To:
```html
        <div id="paramsContainer"></div>
```

(Just close the div immediately - it should be empty initially)

### 3. Move `<script type="importmap">` Back

After deleting the duplicate section, the `<script type="importmap">` tag should be at the correct indentation level (not nested inside the scenario panel).

## Expected Result

After fixes, lines 182-250 should look like:

```html
    <!-- Scenario Configuration Panel (E1.4) -->
    <section id="scenarioPanel" style="...">
        <h3>Scenario Setup</h3>
        <label>Scenario:</label>
        <select id="scenarioSelector" style="..."></select>
        <div id="paramsContainer"></div>  <!-- EMPTY -->
        <button id="runScenarioBtn" style="...">Generate Simulation Command</button>
        <div id="scenarioCommandContainer" style="display:none;">...</div>
    </section>

    <script type="importmap">  <!-- NOT nested -->
        {
            "imports": {
                "three": "...",
                ...
            }
        }
    </script>
```

## Verification

1. Only **ONE** `<section id="scenarioPanel">` should exist
2. The `<div id="paramsContainer"></div>` should be **EMPTY** (closed immediately)
3. The `<script type="importmap">` should NOT be indented inside the scenario panel

## Alternative: Use Clean File

I can create a completely clean test.html file if manual editing is too error-prone.
