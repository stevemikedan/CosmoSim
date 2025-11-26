// CosmoSim Three.js Viewer Prototype
// Loads and displays a single JSON frame

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Scene setup
const canvas = document.getElementById('cosmosim-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

// Camera setup
const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
);
camera.position.set(3, 3, 6);
camera.lookAt(0, 0, 0);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(10, 10, 10);
scene.add(directionalLight);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Load JSON frame
async function loadFrame() {
    try {
        const response = await fetch('../frames/frame_00000.json');
        if (!response.ok) {
            throw new Error(`Failed to load frame: ${response.status}`);
        }

        const data = await response.json();

        // Extract data
        const positions = data.positions || [];
        const masses = data.masses || [];
        const active = data.active || [];

        console.log(`Loaded frame with ${positions.length} entities`);

        // Render active entities
        let activeCount = 0;
        for (let i = 0; i < positions.length; i++) {
            // Skip inactive entities
            if (!active[i]) continue;

            const pos = positions[i];
            const mass = masses[i];

            // Calculate radius based on mass
            const radius = 0.05 + Math.cbrt(mass) * 0.02;

            // Create sphere geometry and material
            const geometry = new THREE.SphereGeometry(radius, 16, 16);
            const material = new THREE.MeshPhongMaterial({ color: 0x55aaff });
            const mesh = new THREE.Mesh(geometry, material);

            // Set position
            mesh.position.set(pos[0], pos[1], pos[2]);

            // Add to scene
            scene.add(mesh);
            activeCount++;
        }

        console.log(`Rendered ${activeCount} active entities`);

    } catch (error) {
        console.error('Error loading frame:', error);
        console.error('Make sure frames/frame_00000.json exists in the parent directory');
    }
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    controls.update();
    renderer.render(scene, camera);
}

// Window resize handler
function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', onWindowResize);

// Initialize
loadFrame();
animate();
