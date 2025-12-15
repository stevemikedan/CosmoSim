/**
 * CosmoSim Viewer - Core Three.js Management
 * Handles scene, renderer, camera, lights, animation loop
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { updateCameraView, resetCamera } from '../autoCamera.js';

export class ViewerManager {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.player = null;
        this.onFrameUpdate = null; // Callback for UI updates every frame

        // Auto-camera configuration
        this.config = {
            autoCamera: true,
            cameraPadding: 1.4,
            defaultCamera: {
                position: [3, 3, 6],
                target: [0, 0, 0]
            }
        };

        this.initRenderer();
        this.initScene();
        this.initCamera();
        this.initLights();
        this.initControls();
        this.initEventListeners();
    }

    initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
    }

    initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
    }

    initCamera() {
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.set(3, 3, 6);
        this.camera.lookAt(0, 0, 0);
    }

    initLights() {
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);
    }

    initControls() {
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
    }

    initEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setPlayer(player) {
        this.player = player;
    }

    animate(currentTime) {
        requestAnimationFrame((time) => this.animate(time));

        if (this.player) {
            this.player.update(currentTime);

            // Call UI update callback (e.g., for HUD updates)
            if (this.onFrameUpdate) {
                this.onFrameUpdate();
            }

            // Auto-camera adjustment
            if (this.config.autoCamera) {
                const frame = this.player.frames[this.player.currentFrame];
                if (frame && frame.positions) {
                    updateCameraView(this.camera, frame.positions, this.config);
                }
            }
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    resetCameraToDefault() {
        resetCamera(this.camera, this.controls, this.config);
    }

    toggleAutoCamera(enabled) {
        this.config.autoCamera = enabled;
        console.log('Auto Camera:', this.config.autoCamera ? 'enabled' : 'disabled');
    }
}
