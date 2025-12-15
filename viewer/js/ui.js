/**
 * CosmoSim Viewer - UI Management
 * Handles all UI interactions, modals, HUD updates, keyboard shortcuts
 */

import { loadFramesByFetch, promptForDirectory } from '../frameLoader.js';
import { FramePlayer } from '../player.js';
import { setVisualMode } from '../visualModes.js';
import { ScenarioConfigUI, ScenarioSchemas } from '../scenarioConfig.js';

export class UIManager {
    constructor(viewer) {
        this.viewer = viewer;
        this.scenarioPanelVisible = false;

        this.initDOMReferences();
        this.initEventListeners();
        this.initScenarioPanel();

        // Register HUD update callback with viewer's animation loop
        this.viewer.onFrameUpdate = () => this.updateHUD();
    }

    initDOMReferences() {
        // Loading overlay
        this.loadingOverlay = document.getElementById('loading');
        this.loadingProgress = document.getElementById('loading-progress');
        this.errorMessage = document.getElementById('error-message');

        // Load buttons
        this.btnLoadFetch = document.getElementById('btn-load-fetch');
        this.btnLoadDirectory = document.getElementById('btn-load-directory');
        this.btnLoadSimfile = document.getElementById('btn-load-simfile');
        this.btnLoadJson = document.getElementById('btn-load-json');
        this.btnLoadDir = document.getElementById('btn-load-dir');
        this.btnReset = document.getElementById('btn-reset');

        // Playback controls
        this.btnPlay = document.getElementById('btn-play');
        this.btnRestart = document.getElementById('btn-restart');
        this.btnPrev = document.getElementById('btn-prev');
        this.btnNext = document.getElementById('btn-next');
        this.fpsInput = document.getElementById('fps-input');

        // Visual controls
        this.visualModeSelect = document.getElementById('visual-mode');
        this.toggleTopology = document.getElementById('toggleTopology');
        this.topologyLabel = document.getElementById('topologyLabel');
        this.toggleAutoCamera = document.getElementById('toggleAutoCamera');
        this.btnResetCamera = document.getElementById('btn-reset-camera');

        // HUD elements
        this.hudFrame = document.getElementById('hud-frame');
        this.hudTotal = document.getElementById('hud-total');
        this.hudTime = document.getElementById('hud-time');
        this.hudEntities = document.getElementById('hud-entities');

        // Scenario panel
        this.btnOpenScenario = document.getElementById('btn-open-scenario');
        this.scenarioPanel = document.getElementById('scenarioPanel');
        this.btnCloseScenario = document.getElementById('btn-close-scenario');
    }

    initEventListeners() {
        // Frame load progress
        window.addEventListener('frameLoadProgress', (e) => {
            const { current, total } = e.detail;
            this.loadingProgress.textContent = `Loading frames: ${current}${total !== '?' ? ` / ${total}` : '...'}`;
        });

        // Load buttons
        this.btnLoadFetch?.addEventListener('click', () => this.handleLoadFetch());
        this.btnLoadDirectory?.addEventListener('click', () => this.handleLoadDirectory());
        this.btnLoadSimfile?.addEventListener('click', () => this.handleLoadSimfile());
        this.btnLoadJson?.addEventListener('click', () => this.btnLoadSimfile.click());
        this.btnLoadDir?.addEventListener('click', () => this.btnLoadDirectory.click());
        this.btnReset?.addEventListener('click', () => this.handleReset());

        // Playback controls
        this.btnPlay?.addEventListener('click', () => this.handlePlayPause());
        this.btnRestart?.addEventListener('click', () => this.handleRestart());
        this.btnPrev?.addEventListener('click', () => this.handlePrev());
        this.btnNext?.addEventListener('click', () => this.handleNext());
        this.fpsInput?.addEventListener('input', () => this.handleFPSChange());

        // Visual controls
        this.visualModeSelect?.addEventListener('change', (e) => this.handleVisualModeChange(e));
        this.toggleTopology?.addEventListener('change', (e) => this.handleTopologyToggle(e));
        this.toggleAutoCamera?.addEventListener('change', (e) => this.handleAutoCameraToggle(e));
        this.btnResetCamera?.addEventListener('click', () => this.handleResetCamera());

        // Scenario panel
        this.btnOpenScenario?.addEventListener('click', () => this.openScenarioPanel());
        this.btnCloseScenario?.addEventListener('click', () => this.closeScenarioPanel());

        // Keyboard shortcuts
        window.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }

    initScenarioPanel() {
        if (this.scenarioPanel) {
            try {
                new ScenarioConfigUI(this.scenarioPanel, ScenarioSchemas);
                console.log('[UI] Scenario Config Panel initialized');
            } catch (error) {
                console.error('[UI] Failed to initialize Scenario Config Panel:', error);
            }
        }
    }

    // ============================================
    // File Loading
    // ============================================

    async handleLoadFetch() {
        try {
            this.errorMessage.textContent = '';
            this.loadingProgress.textContent = 'Loading frames...';
            this.btnLoadFetch.disabled = true;
            this.btnLoadDirectory.disabled = true;

            const frames = await loadFramesByFetch('../frames/frame_', 1000);

            if (frames.length === 0) {
                throw new Error('No frames found in ../frames/');
            }

            this.initializePlayer(frames);
        } catch (error) {
            this.errorMessage.textContent = `Error: ${error.message}`;
            console.error(error);
            this.btnLoadFetch.disabled = false;
            this.btnLoadDirectory.disabled = false;
        }
    }

    async handleLoadDirectory() {
        try {
            this.errorMessage.textContent = '';
            this.loadingProgress.textContent = 'Select directory...';

            const frames = await promptForDirectory();

            if (frames.length === 0) {
                throw new Error('No JSON frames found in selected directory');
            }

            this.initializePlayer(frames);
        } catch (error) {
            this.errorMessage.textContent = `Error: ${error.message}`;
            console.error(error);
        }
    }

    async handleLoadSimfile() {
        try {
            this.errorMessage.textContent = '';
            this.loadingProgress.textContent = 'Select a .json simulation file...';

            const [fileHandle] = await window.showOpenFilePicker({
                types: [{
                    description: 'CosmoSim JSON Simulation',
                    accept: { 'application/json': ['.json'] }
                }]
            });

            const file = await fileHandle.getFile();
            const text = await file.text();
            const sim = JSON.parse(text);

            if (!sim.frames || !Array.isArray(sim.frames)) {
                throw new Error('Invalid simulation file — missing frames[]');
            }

            const frames = sim.frames.map((frame, idx) => ({
                positions: frame.positions,
                velocities: frame.velocities,
                masses: frame.masses || [],
                active: frame.active || [],
                topology: frame.topology || null,
                index: idx
            }));

            this.initializePlayer(frames);

        } catch (err) {
            this.errorMessage.textContent = `Error loading simulation: ${err.message}`;
            console.error(err);
        }
    }

    handleReset() {
        console.log('[Viewer] Resetting simulation...');
        this.cleanupCurrentSimulation();

        // Reset loading overlay
        this.errorMessage.textContent = '';
        this.loadingProgress.textContent = 'Select a file to load...';
        this.loadingOverlay.classList.remove('hidden');

        console.log('[Viewer] Simulation state cleared.');
    }

    cleanupCurrentSimulation() {
        console.log('[UI] Cleaning up current simulation...');

        // Stop playback if running
        if (this.viewer.player) {
            this.viewer.player.pause();
            console.log('[UI] Paused player');
        }

        // Remove rendered particle system (instancedMesh)
        if (this.viewer.player && this.viewer.player.instancedMesh) {
            this.viewer.scene.remove(this.viewer.player.instancedMesh);
            this.viewer.player.instancedMesh.geometry.dispose();
            this.viewer.player.instancedMesh.material.dispose();
            console.log('[UI] Removed instancedMesh');
        }

        // Remove topology overlay
        if (this.viewer.player && this.viewer.player.topologyOverlay && this.viewer.player.topologyOverlay.group) {
            this.viewer.scene.remove(this.viewer.player.topologyOverlay.group);
            console.log('[UI] Removed topology overlay');
        }

        // Clear player
        this.viewer.player = null;
        console.log('[UI] Cleared player reference');

        // Reset HUD
        this.hudFrame.textContent = '0';
        this.hudTotal.textContent = '0';
        this.hudTime.textContent = '0.000';
        this.hudEntities.textContent = '0';
    }

    initializePlayer(frames) {
        // Clean up any existing simulation first
        this.cleanupCurrentSimulation();

        const player = new FramePlayer(frames, this.viewer.scene);
        player.initializeEntities();

        // Wire up topology UI
        player.topologyOverlay.onModeChange = (mode) => {
            this.topologyLabel.textContent = `(${mode})`;
        };

        // Initial topology state
        const initialTopo = frames[0]?.topology;
        if (initialTopo) {
            player.topologyOverlay.update(initialTopo);
        } else {
            this.topologyLabel.textContent = '(none)';
        }

        player.updateFrame();

        this.viewer.setPlayer(player);
        this.loadingOverlay.classList.add('hidden');
        this.updateHUD();
    }

    // ============================================
    // Playback Controls
    // ============================================

    handlePlayPause() {
        if (!this.viewer.player) return;
        this.viewer.player.toggle();
        this.updatePlayButton();
    }

    handleRestart() {
        if (!this.viewer.player) return;
        this.viewer.player.restart();
        this.updatePlayButton();
        this.updateHUD();
    }

    handlePrev() {
        if (!this.viewer.player) return;
        this.viewer.player.prev();
        this.updateHUD();
    }

    handleNext() {
        if (!this.viewer.player) return;
        this.viewer.player.next();
        this.updateHUD();
    }

    handleFPSChange() {
        if (!this.viewer.player) return;
        this.viewer.player.setFPS(parseInt(this.fpsInput.value) || 30);
    }

    updatePlayButton() {
        if (!this.viewer.player) return;
        this.btnPlay.textContent = this.viewer.player.isPlaying ? '⏸ Pause' : '▶ Play';
    }

    // ============================================
    // Visual Controls
    // ============================================

    handleVisualModeChange(e) {
        setVisualMode(e.target.value);
        if (this.viewer.player) this.viewer.player.refreshVisuals();
        // Remove focus so keyboard shortcuts still work
        this.visualModeSelect.blur();
    }

    handleTopologyToggle(e) {
        if (this.viewer.player) {
            this.viewer.player.topologyOverlay.setVisible(e.target.checked);
        }
    }

    handleAutoCameraToggle(e) {
        this.viewer.toggleAutoCamera(e.target.checked);
    }

    handleResetCamera() {
        this.viewer.resetCameraToDefault();
    }

    // ============================================
    // HUD Updates
    // ============================================

    updateHUD() {
        if (!this.viewer.player) return;
        const info = this.viewer.player.getFrameInfo();
        this.hudFrame.textContent = info.frame;
        this.hudTotal.textContent = info.total;
        this.hudTime.textContent = info.time.toFixed(3);
        this.hudEntities.textContent = info.activeCount;
    }

    // ============================================
    // Scenario Panel
    // ============================================

    openScenarioPanel() {
        this.scenarioPanel.classList.add('visible');
        this.scenarioPanelVisible = true;
    }

    closeScenarioPanel() {
        this.scenarioPanel.classList.remove('visible');
        this.scenarioPanelVisible = false;
    }

    // ============================================
    // Keyboard Shortcuts
    // ============================================

    handleKeyboard(e) {
        if (!this.viewer.player) return;

        // Visual Mode shortcuts (1-5)
        if (['1', '2', '3', '4', '5'].includes(e.key)) {
            const modes = {
                '1': 'mass-color',
                '2': 'type-color',
                '3': 'velocity-color',
                '4': 'active-mask',
                '5': 'uniform'
            };
            const mode = modes[e.key];
            setVisualMode(mode);
            this.visualModeSelect.value = mode;
            this.viewer.player.refreshVisuals();
            return;
        }

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                this.handlePlayPause();
                break;
            case 'ArrowRight':
                e.preventDefault();
                this.handleNext();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                this.handlePrev();
                break;
            case 'KeyR':
                e.preventDefault();
                this.handleRestart();
                break;
            case 'KeyC':
                e.preventDefault();
                this.handleResetCamera();
                break;
        }
    }
}
