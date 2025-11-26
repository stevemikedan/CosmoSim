// Frame Loader Module for CosmoSim Viewer
// Handles loading JSON frame sequences from directory or via fetch

/**
 * Load frames by scanning a user-selected directory
 * @param {FileSystemDirectoryHandle} dirHandle - Directory handle from file picker
 * @returns {Promise<Array>} Array of parsed frame objects
 */
export async function loadFramesFromDirectory(dirHandle) {
    const frames = [];
    const fileList = [];

    // Collect all JSON files
    for await (const entry of dirHandle.values()) {
        if (entry.kind === 'file' && entry.name.endsWith('.json')) {
            fileList.push({ name: entry.name, handle: entry });
        }
    }

    // Sort by frame number
    fileList.sort((a, b) => {
        const numA = parseInt(a.name.match(/\d+/)?.[0] || '0');
        const numB = parseInt(b.name.match(/\d+/)?.[0] || '0');
        return numA - numB;
    });

    console.log(`Found ${fileList.length} JSON files`);

    // Load each file
    for (let i = 0; i < fileList.length; i++) {
        try {
            const file = await fileList[i].handle.getFile();
            const text = await file.text();
            const data = JSON.parse(text);
            frames.push(data);

            // Dispatch progress event
            window.dispatchEvent(new CustomEvent('frameLoadProgress', {
                detail: { current: i + 1, total: fileList.length }
            }));
        } catch (error) {
            console.warn(`Failed to load ${fileList[i].name}:`, error);
        }
    }

    return frames;
}

/**
 * Load frames by fetching sequentially from a path prefix
 * @param {string} prefix - Path prefix (e.g., '../frames/frame_')
 * @param {number} maxFrames - Maximum frames to attempt loading
 * @returns {Promise<Array>} Array of parsed frame objects
 */
export async function loadFramesByFetch(prefix = '../frames/frame_', maxFrames = 1000) {
    const frames = [];
    let frameIndex = 0;
    let consecutiveFailures = 0;

    while (frameIndex < maxFrames && consecutiveFailures < 3) {
        const framePath = `${prefix}${String(frameIndex).padStart(5, '0')}.json`;

        try {
            const response = await fetch(framePath);
            if (!response.ok) {
                consecutiveFailures++;
                if (frameIndex === 0) {
                    throw new Error(`No frames found at ${framePath}`);
                }
                break;
            }

            const data = await response.json();
            frames.push(data);
            consecutiveFailures = 0;

            // Dispatch progress event
            window.dispatchEvent(new CustomEvent('frameLoadProgress', {
                detail: { current: frameIndex + 1, total: '?' }
            }));

            frameIndex++;
        } catch (error) {
            if (frameIndex === 0) {
                throw error;
            }
            consecutiveFailures++;
        }
    }

    console.log(`Loaded ${frames.length} frames via fetch`);
    return frames;
}

/**
 * Show directory picker and load frames
 * @returns {Promise<Array>} Array of parsed frame objects
 */
export async function promptForDirectory() {
    if (!window.showDirectoryPicker) {
        throw new Error('Directory picker not supported in this browser');
    }

    const dirHandle = await window.showDirectoryPicker({
        mode: 'read',
        startIn: 'downloads'
    });

    return loadFramesFromDirectory(dirHandle);
}
