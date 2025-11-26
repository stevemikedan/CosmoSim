// Frame Player Module for CosmoSim Viewer
// Handles playback control and frame updates

import * as THREE from 'three';

export class FramePlayer {
    constructor(frames, scene) {
        this.frames = frames;
        this.scene = scene;
        this.currentFrame = 0;
        this.isPlaying = false;
        this.fps = 30;
        this.lastFrameTime = 0;
        this.entities = []; // Array of { mesh, initialIndex }

        console.log(`FramePlayer initialized with ${frames.length} frames`);
    }

    /**
     * Create Three.js meshes for all entities in the first frame
     */
    initializeEntities() {
        if (this.frames.length === 0) return;

        const firstFrame = this.frames[0];
        const positions = firstFrame.positions || [];
        const masses = firstFrame.masses || [];
        const active = firstFrame.active || [];

        // Clear existing entities
        this.entities.forEach(entity => this.scene.remove(entity.mesh));
        this.entities = [];

        // Create a mesh for each entity (even inactive ones, we'll hide them)
        for (let i = 0; i < positions.length; i++) {
            const mass = masses[i] || 1.0;
            const radius = 0.05 + Math.cbrt(mass) * 0.02;

            const geometry = new THREE.SphereGeometry(radius, 16, 16);
            const material = new THREE.MeshPhongMaterial({ color: 0x55aaff });
            const mesh = new THREE.Mesh(geometry, material);

            // Set initial position
            const pos = positions[i];
            mesh.position.set(pos[0], pos[1], pos[2]);
            mesh.visible = active[i];

            this.scene.add(mesh);
            this.entities.push({ mesh, initialIndex: i });
        }

        console.log(`Created ${this.entities.length} entity meshes`);
    }

    /**
     * Update entity positions from current frame
     */
    updateFrame() {
        if (this.frames.length === 0 || this.currentFrame >= this.frames.length) {
            return;
        }

        const frame = this.frames[this.currentFrame];
        const positions = frame.positions || [];
        const active = frame.active || [];

        // Update each entity's position and visibility
        for (let i = 0; i < this.entities.length && i < positions.length; i++) {
            const entity = this.entities[i];
            const pos = positions[i];

            entity.mesh.position.set(pos[0], pos[1], pos[2]);
            entity.mesh.visible = active[i];
        }
    }

    /**
     * Start playback
     */
    play() {
        this.isPlaying = true;
        this.lastFrameTime = performance.now();
    }

    /**
     * Pause playback
     */
    pause() {
        this.isPlaying = false;
    }

    /**
     * Toggle play/pause
     */
    toggle() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    /**
     * Restart from frame 0
     */
    restart() {
        this.currentFrame = 0;
        this.updateFrame();
        this.pause();
    }

    /**
     * Step to next frame
     */
    next() {
        if (this.currentFrame < this.frames.length - 1) {
            this.currentFrame++;
            this.updateFrame();
        }
    }

    /**
     * Step to previous frame
     */
    prev() {
        if (this.currentFrame > 0) {
            this.currentFrame--;
            this.updateFrame();
        }
    }

    /**
     * Update player state (call every animation frame)
     * @param {number} currentTime - Current timestamp from requestAnimationFrame
     */
    update(currentTime) {
        if (!this.isPlaying || this.frames.length === 0) {
            return;
        }

        const frameDuration = 1000 / this.fps;
        const elapsed = currentTime - this.lastFrameTime;

        if (elapsed >= frameDuration) {
            this.lastFrameTime = currentTime;

            // Advance to next frame
            this.currentFrame++;

            // Loop back to start if at end
            if (this.currentFrame >= this.frames.length) {
                this.currentFrame = 0;
            }

            this.updateFrame();
        }
    }

    /**
     * Get current frame info for HUD
     */
    getFrameInfo() {
        if (this.frames.length === 0) {
            return { frame: 0, total: 0, time: 0, activeCount: 0 };
        }

        const frame = this.frames[this.currentFrame];
        const activeCount = (frame.active || []).filter(a => a).length;
        const time = frame.time || 0;

        return {
            frame: this.currentFrame,
            total: this.frames.length,
            time: time,
            activeCount: activeCount
        };
    }

    /**
     * Set playback speed (frames per second)
     */
    setFPS(fps) {
        this.fps = Math.max(1, Math.min(60, fps));
    }
}
