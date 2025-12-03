/**
 * Scenario Configuration UI for CosmoSim Web Viewer
 * 
 * Provides dynamic parameter editing and CLI command generation
 * based on PSS (Parameterized Scenario System) schemas.
 */

// PSS Schema Registry (JS-side only, matches Python SCENARIO_PARAMS)
export const ScenarioSchemas = {
    "bulk_ring": {
        "N": {
            type: "int",
            default: 64,
            min: 1,
            max: 128,
            description: "Number of entities in the ring"
        },
        "radius": {
            type: "float",
            default: 8.0,
            min: 1.0,
            max: 50.0,
            description: "Orbital radius of the ring"
        },
        "speed": {
            type: "float",
            default: 0.8,
            min: 0.1,
            max: 5.0,
            description: "Tangential velocity"
        },
        "mass": {
            type: "float",
            default: 1.0,
            min: 0.1,
            max: 10.0,
            description: "Mass per entity"
        }
    },
    "random_nbody": {
        "N": {
            type: "int",
            default: 300,
            min: 1,
            max: 5000,
            description: "Number of random entities"
        }
    }
};

/**
 * ScenarioConfigUI Class
 * 
 * Dynamically generates parameter input forms and CLI commands.
 */
export class ScenarioConfigUI {
    constructor(rootElement, schemas) {
        this.root = rootElement;
        this.schemas = schemas;

        // DOM References
        this.selector = document.getElementById("scenarioSelector");
        this.paramsContainer = document.getElementById("paramsContainer");
        this.runBtn = document.getElementById("runScenarioBtn");
        this.commandOutput = document.getElementById("scenarioCommandOutput");
        this.commandContainer = document.getElementById("scenarioCommandContainer");
        this.copyBtn = document.getElementById("copyScenarioCommandBtn");

        // State
        this.currentParams = {};

        // Initialize
        this.populateScenarios();

        // Event Listeners
        this.selector.onchange = () => this.renderParams();
        this.runBtn.onclick = () => this.generateCommand();
        this.copyBtn.onclick = () => this.copyCommand();
    }

    /**
     * Populate scenario dropdown with available scenarios
     */
    populateScenarios() {
        const names = Object.keys(this.schemas).sort();

        for (const name of names) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            this.selector.appendChild(opt);
        }

        if (names.length > 0) {
            this.selector.value = names[0];
            this.renderParams();
        }
    }

    /**
     * Render parameter input fields for selected scenario
     */
    renderParams() {
        const scenario = this.selector.value;
        const schema = this.schemas[scenario] || null;

        // Clear previous
        this.paramsContainer.innerHTML = "";
        this.currentParams = {};
        this.commandContainer.style.display = "none";

        if (!schema) {
            const msg = document.createElement("div");
            msg.textContent = "No editable parameters for this scenario.";
            msg.style.color = "#aaa";
            msg.style.fontSize = "12px";
            msg.style.marginTop = "10px";
            this.paramsContainer.appendChild(msg);
            return;
        }

        // Generate input fields
        for (const [key, spec] of Object.entries(schema)) {
            const wrapper = document.createElement("div");
            wrapper.style.marginBottom = "12px";

            const label = document.createElement("label");
            label.textContent = `${key}`;
            label.style.display = "block";
            label.style.marginBottom = "4px";
            label.style.fontSize = "13px";
            label.style.color = "#fff";
            wrapper.appendChild(label);

            // Description tooltip
            if (spec.description) {
                const desc = document.createElement("div");
                desc.textContent = spec.description;
                desc.style.fontSize = "10px";
                desc.style.color = "#aaa";
                desc.style.marginBottom = "4px";
                wrapper.appendChild(desc);
            }

            // Input field
            let input;

            if (spec.type === "bool") {
                input = document.createElement("input");
                input.type = "checkbox";
                input.checked = !!spec.default;
                input.style.transform = "scale(1.2)";
            } else {
                input = document.createElement("input");
                input.type = "number";
                input.value = spec.default;
                input.style.width = "100%";
                input.style.padding = "6px";
                input.style.borderRadius = "4px";
                input.style.border = "1px solid #555";
                input.style.background = "#222";
                input.style.color = "#fff";

                if (spec.type === "int") {
                    input.step = "1";
                }
                if (spec.min !== undefined) {
                    input.min = spec.min;
                }
                if (spec.max !== undefined) {
                    input.max = spec.max;
                }
            }

            // Update handler
            input.onchange = () => {
                this.currentParams[key] = this.readParamValue(spec, input);
            };

            // Initialize current params
            this.currentParams[key] = spec.default;

            wrapper.appendChild(input);
            this.paramsContainer.appendChild(wrapper);
        }
    }

    /**
     * Read typed value from input element
     */
    readParamValue(spec, input) {
        if (spec.type === "bool") {
            return input.checked;
        } else if (spec.type === "int") {
            return parseInt(input.value, 10);
        } else if (spec.type === "float") {
            return parseFloat(input.value);
        }
        return input.value;
    }

    /**
     * Generate CLI command with current parameters
     */
    generateCommand() {
        const scenario = this.selector.value;
        const pieces = [];

        for (const [key, value] of Object.entries(this.currentParams)) {
            pieces.push(`${key}=${value}`);
        }

        const paramStr = pieces.join(",");
        const cmd = `python cosmosim.py --scenario ${scenario} --params "${paramStr}" --steps 500 --view web`;

        this.commandOutput.value = cmd;
        this.commandContainer.style.display = "block";
    }

    /**
     * Copy command to clipboard
     */
    copyCommand() {
        this.commandOutput.select();
        this.commandOutput.setSelectionRange(0, 99999); // For mobile

        // Modern clipboard API
        if (navigator.clipboard) {
            navigator.clipboard.writeText(this.commandOutput.value)
                .then(() => {
                    this.copyBtn.textContent = "✓ Copied!";
                    setTimeout(() => {
                        this.copyBtn.textContent = "Copy to Clipboard";
                    }, 2000);
                })
                .catch(() => {
                    // Fallback to document.execCommand
                    document.execCommand("copy");
                });
        } else {
            // Fallback for older browsers
            document.execCommand("copy");
        }
    }
}
