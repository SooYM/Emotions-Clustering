// Application Configuration and State
const CLUSTER_COLORS = [
    '#3b82f6', // Cluster 0: Blue
    '#10b981', // Cluster 1: Emerald
    '#f43f5e', // Cluster 2: Rose
    '#a855f7', // Cluster 3: Purple
    '#eab308', // Cluster 4: Amber
    '#06b6d4', // Cluster 5: Cyan
    '#ec4899'  // Cluster 6: Pink
];

const CLUSTER_NAMES = [
    "Cluster Alpha",
    "Cluster Beta",
    "Cluster Gamma",
    "Cluster Delta",
    "Cluster Epsilon",
    "Cluster Zeta",
    "Cluster Eta"
];

const EMOTION_COLORS = {
    "happy": "#10b981",
    "sad": "#3b82f6",
    "anxiety": "#a855f7",
    "angry": "#f43f5e",
    "surprised": "#eab308",
    "disgust": "#06b6d4",
    "neutral": "#6366f1"
};

let appState = {
    coords: [],          // Downsampled coordinates
    representatives: {}, // Representatives per cluster
    activeTab: 0,        // Selected cluster tab
    predictedPoint: null,// Coords of predicted image {x, y, cluster}
    animationFrame: null,// Animation frame ID
    pulseRadius: 6,
    pulseGrowing: true,
    
    // Mode and Webcam States
    currentMode: "image", // "image" or "webcam"
    webcamActive: false,
    faceLandmarker: null,
    stream: null,
    lastInferenceTime: 0,
    inferenceThrottleMs: 800
};

// DOM Elements
const canvas = document.getElementById('clusterPlot');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('plotTooltip');
const tooltipImg = document.getElementById('tooltipImg');
const tooltipText = document.getElementById('tooltipText');
const plotLegend = document.getElementById('plotLegend');

// Mode toggle buttons
const btnModeImage = document.getElementById('btnModeImage');
const btnModeWebcam = document.getElementById('btnModeWebcam');
const imageModeContainer = document.getElementById('imageModeContainer');
const webcamModeContainer = document.getElementById('webcamModeContainer');

// Image upload DOMs
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const btnReset = document.getElementById('btnReset');
const processStatus = document.getElementById('processStatus');

// Webcam DOMs
const webcamVideo = document.getElementById('webcamVideo');
const webcamOverlay = document.getElementById('webcamOverlay');
const webcamOverlayCtx = webcamOverlay.getContext('2d');
const btnStartWebcam = document.getElementById('btnStartWebcam');
const btnStopWebcam = document.getElementById('btnStopWebcam');
const webcamStatusOverlay = document.getElementById('webcamStatusOverlay');
const webcamStatusText = document.getElementById('webcamStatusText');

// Output DOMs
const resultsBox = document.getElementById('resultsBox');
const predictedBadge = document.getElementById('predictedBadge');
const emotionBadge = document.getElementById('emotionBadge');
const distanceList = document.getElementById('distanceList');
const clusterTabs = document.getElementById('clusterTabs');
const repsGrid = document.getElementById('repsGrid');
const galleryTitle = document.getElementById('galleryTitle');
const galleryDesc = document.getElementById('galleryDesc');

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    fetchClusterData();
    setupUploadHandlers();
    setupModeToggle();
    setupWebcamControls();
    setupResizeHandler();
});

// Fetch Clusters and Coordinates
async function fetchClusterData() {
    try {
        const response = await fetch('/api/clusters');
        if (!response.ok) throw new Error("Failed to load clustering data");
        const data = await response.json();
        
        appState.coords = data.coords;
        appState.representatives = data.representatives;
        
        buildLegend();
        buildTabs();
        selectTab(0);
        
        initCanvas();
        startAnimationLoop();
        
    } catch (error) {
        console.error("Error loading cluster data:", error);
        alert("Could not load cluster data. Please run feature extraction first.");
    }
}

// Build Plot Legend
function buildLegend() {
    plotLegend.innerHTML = '';
    CLUSTER_COLORS.forEach((color, idx) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        item.style.color = color;
        item.innerHTML = `
            <span class="legend-color" style="background-color: ${color}"></span>
            <span class="legend-label">C${idx} (${CLUSTER_NAMES[idx].split(' ')[1]})</span>
        `;
        item.addEventListener('click', () => {
            selectTab(idx);
            document.querySelector('.cluster-reps-card').scrollIntoView({ behavior: 'smooth' });
        });
        plotLegend.appendChild(item);
    });
}

// Build Cluster Tabs for bottom viewer
function buildTabs() {
    clusterTabs.innerHTML = '';
    CLUSTER_COLORS.forEach((color, idx) => {
        const tab = document.createElement('button');
        tab.className = 'tab-btn';
        tab.innerText = `C${idx}: ${CLUSTER_NAMES[idx].split(' ')[1]}`;
        tab.style.setProperty('--btn-color', color);
        tab.style.setProperty('--glow-color', color + '40');
        tab.addEventListener('click', () => selectTab(idx));
        clusterTabs.appendChild(tab);
    });
}

// Select Cluster Tab and show representatives
function selectTab(idx) {
    appState.activeTab = idx;
    
    // Update tab classes
    const tabs = clusterTabs.querySelectorAll('.tab-btn');
    tabs.forEach((tab, i) => {
        if (i === idx) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    
    // Update gallery details
    galleryTitle.innerText = `${CLUSTER_NAMES[idx]} (Cluster ${idx})`;
    galleryDesc.innerText = `Displaying the 15 closest training face images to the centroid.`;
    
    // Set custom visual highlighting on reps container
    repsGrid.style.setProperty('--active-cluster-color', CLUSTER_COLORS[idx]);
    repsGrid.style.setProperty('--active-cluster-glow', CLUSTER_COLORS[idx] + '40');
    
    // Populate representatives
    repsGrid.innerHTML = '';
    const reps = appState.representatives[idx] || [];
    reps.forEach(rep => {
        const wrapper = document.createElement('div');
        wrapper.className = 'rep-img-wrapper';
        wrapper.innerHTML = `
            <img src="/images/${rep.name}" alt="${rep.name}" loading="lazy">
        `;
        wrapper.title = `${rep.name} (Dist: ${rep.distance.toFixed(3)})`;
        repsGrid.appendChild(wrapper);
    });
}

// --- Canvas Plot Rendering ---
let mapBounds = { minX: 0, maxX: 0, minY: 0, maxY: 0 };

function initCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth * window.devicePixelRatio;
    canvas.height = container.clientHeight * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    if (appState.coords.length === 0) return;
    
    const xs = appState.coords.map(c => c.x);
    const ys = appState.coords.map(c => c.y);
    
    mapBounds.minX = Math.min(...xs);
    mapBounds.maxX = Math.max(...xs);
    mapBounds.minY = Math.min(...ys);
    mapBounds.maxY = Math.max(...ys);
    
    const xPad = (mapBounds.maxX - mapBounds.minX) * 0.08;
    const yPad = (mapBounds.maxY - mapBounds.minY) * 0.08;
    mapBounds.minX -= xPad;
    mapBounds.maxX += xPad;
    mapBounds.minY -= yPad;
    mapBounds.maxY += yPad;
    
    setupCanvasInteractions();
}

function setupResizeHandler() {
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            initCanvas();
        }, 150);
    });
}

function toScreen(x, y) {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    
    const screenX = ((x - mapBounds.minX) / (mapBounds.maxX - mapBounds.minX)) * w;
    const screenY = h - (((y - mapBounds.minY) / (mapBounds.maxY - mapBounds.minY)) * h);
    
    return { x: screenX, y: screenY };
}

// Draw Plot Frame
function drawPlot() {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    
    ctx.clearRect(0, 0, w, h);
    
    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
    ctx.lineWidth = 1;
    const gridCount = 8;
    for (let i = 1; i < gridCount; i++) {
        const xPos = (w / gridCount) * i;
        ctx.beginPath();
        ctx.moveTo(xPos, 0);
        ctx.lineTo(xPos, h);
        ctx.stroke();
        
        const yPos = (h / gridCount) * i;
        ctx.beginPath();
        ctx.moveTo(0, yPos);
        ctx.lineTo(w, yPos);
        ctx.stroke();
    }
    
    // Draw data points
    appState.coords.forEach(pt => {
        const screenPt = toScreen(pt.x, pt.y);
        ctx.fillStyle = CLUSTER_COLORS[pt.cluster] + 'bb'; // Increased opacity for better visibility
        ctx.beginPath();
        ctx.arc(screenPt.x, screenPt.y, 4.5, 0, 2 * Math.PI); // Slightly larger points
        ctx.fill();
    });
    
    // Draw predicted point if active
    if (appState.predictedPoint) {
        const pPt = toScreen(appState.predictedPoint.x, appState.predictedPoint.y);
        const color = CLUSTER_COLORS[appState.predictedPoint.cluster];
        
        // Animated Pulse Ring
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(pPt.x, pPt.y, appState.pulseRadius, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Glowing Core
        const gradient = ctx.createRadialGradient(pPt.x, pPt.y, 1, pPt.x, pPt.y, 9);
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.3, color);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        
        ctx.beginPath();
        ctx.arc(pPt.x, pPt.y, 9, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// Animation loop
function startAnimationLoop() {
    function animate() {
        if (appState.pulseGrowing) {
            appState.pulseRadius += 0.35;
            if (appState.pulseRadius >= 20) appState.pulseGrowing = false;
        } else {
            appState.pulseRadius -= 0.35;
            if (appState.pulseRadius <= 5) appState.pulseGrowing = true;
        }
        
        drawPlot();
        appState.animationFrame = requestAnimationFrame(animate);
    }
    animate();
}

// Canvas Hover Interactions
function setupCanvasInteractions() {
    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        let nearest = null;
        let minDist = 20; // Max hover distance in pixels
        
        appState.coords.forEach(pt => {
            const screenPt = toScreen(pt.x, pt.y);
            const dist = Math.hypot(screenPt.x - mouseX, screenPt.y - mouseY);
            if (dist < minDist) {
                minDist = dist;
                nearest = pt;
            }
        });
        
        if (nearest) {
            tooltipImg.src = `/images/${nearest.name}`;
            tooltipText.innerHTML = `File: ${nearest.name}<br>Cluster: ${nearest.cluster}`;
            tooltip.classList.remove('hidden');
            
            const tooltipW = tooltip.offsetWidth || 90;
            const tooltipH = tooltip.offsetHeight || 110;
            let left = mouseX + 15;
            let top = mouseY - tooltipH / 2;
            
            if (left + tooltipW > rect.width) {
                left = mouseX - tooltipW - 15;
            }
            if (top < 10) top = 10;
            if (top + tooltipH > rect.height - 10) {
                top = rect.height - tooltipH - 10;
            }
            
            tooltip.style.left = `${left}px`;
            tooltip.style.top = `${top}px`;
            canvas.style.cursor = 'pointer';
        } else {
            tooltip.classList.add('hidden');
            canvas.style.cursor = 'crosshair';
        }
    });
    
    canvas.addEventListener('mouseleave', () => {
        tooltip.classList.add('hidden');
    });
    
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        let nearest = null;
        let minDist = 22;
        
        appState.coords.forEach(pt => {
            const screenPt = toScreen(pt.x, pt.y);
            const dist = Math.hypot(screenPt.x - mouseX, screenPt.y - mouseY);
            if (dist < minDist) {
                minDist = dist;
                nearest = pt;
            }
        });
        
        if (nearest) {
            selectTab(nearest.cluster);
            document.querySelector('.cluster-reps-card').scrollIntoView({ behavior: 'smooth' });
        }
    });
}

// --- Mode Toggle Setup ---
function setupModeToggle() {
    btnModeImage.addEventListener('click', () => {
        if (appState.currentMode === "image") return;
        switchMode("image");
    });
    
    btnModeWebcam.addEventListener('click', () => {
        if (appState.currentMode === "webcam") return;
        switchMode("webcam");
    });
}

function switchMode(mode) {
    appState.currentMode = mode;
    
    if (mode === "image") {
        btnModeImage.classList.add('active');
        btnModeWebcam.classList.remove('active');
        imageModeContainer.classList.remove('hidden');
        webcamModeContainer.classList.add('hidden');
        
        stopWebcam();
        resetUpload();
    } else {
        btnModeWebcam.classList.add('active');
        btnModeImage.classList.remove('active');
        webcamModeContainer.classList.remove('hidden');
        imageModeContainer.classList.add('hidden');
        
        resetUpload();
        // Prompt loading models
        initMediaPipe();
    }
}

// --- Single Image Upload Handling ---
function setupUploadHandlers() {
    dropZone.addEventListener('dragenter', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('dragover'); });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const dt = e.dataTransfer;
        if (dt.files.length > 0) handleUploadedFile(dt.files[0]);
    });
    
    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleUploadedFile(e.target.files[0]);
    });
    
    btnReset.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });
}

function handleUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        processStatus.classList.remove('hidden');
        resultsBox.classList.add('hidden');
        
        uploadAndPredict(file);
    };
    reader.readAsDataURL(file);
}

async function uploadAndPredict(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Inference failed");
        }
        
        const result = await response.json();
        showPredictionResults(result);
        
    } catch (error) {
        console.error("Error predicting:", error);
        processStatus.innerHTML = `<span style="color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> ${error.message}</span>`;
    }
}

function showPredictionResults(result) {
    processStatus.classList.add('hidden');
    resultsBox.classList.remove('hidden');
    
    // Render Cluster Badge
    const clusterIdx = result.predicted_cluster;
    predictedBadge.innerText = `${CLUSTER_NAMES[clusterIdx]} (Cluster ${clusterIdx})`;
    predictedBadge.style.backgroundColor = CLUSTER_COLORS[clusterIdx];
    predictedBadge.style.boxShadow = `0 0 20px ${CLUSTER_COLORS[clusterIdx]}50`;
    
    // Render Emotion Badge with matching color
    const emotion = result.emotion_label;
    emotionBadge.innerText = emotion;
    emotionBadge.style.background = EMOTION_COLORS[emotion] || '#6366f1';
    emotionBadge.style.boxShadow = `0 0 20px ${EMOTION_COLORS[emotion] || '#6366f1'}50`;
    
    // Set prediction coordinates overlay on plot
    appState.predictedPoint = {
        x: result.x,
        y: result.y,
        cluster: clusterIdx
    };
    
    // Update Representatives Gallery tab
    selectTab(clusterIdx);
    
    // Render progress bars
    renderDistances(result.distances, clusterIdx);
}

function renderDistances(distances, predictedClusterIdx) {
    distanceList.innerHTML = '';
    
    const distValues = Object.values(distances);
    const maxDist = Math.max(...distValues);
    const minDist = Math.min(...distValues);
    
    const sortedClusters = Object.keys(distances).map(k => parseInt(k)).sort((a, b) => distances[a] - distances[b]);
    
    sortedClusters.forEach(idx => {
        const dist = distances[idx];
        
        let pct = 100;
        if (maxDist > minDist) {
            pct = 15 + 85 * (1 - (dist - minDist) / (maxDist - minDist));
        }
        
        const color = CLUSTER_COLORS[idx];
        const isMatched = idx === predictedClusterIdx;
        
        const item = document.createElement('div');
        item.className = 'dist-item';
        item.innerHTML = `
            <div class="dist-meta">
                <span class="dist-name" style="color: ${isMatched ? '#ffffff' : 'var(--text-muted)'}">
                    ${isMatched ? '<i class="fa-solid fa-chevron-right"></i> ' : ''}Cluster ${idx} (${CLUSTER_NAMES[idx].split(' ')[1]})
                </span>
                <span class="dist-val" style="color: ${color}">Dist: ${dist.toFixed(2)} (${Math.round(pct)}% Match)</span>
            </div>
            <div class="dist-bar-bg">
                <div class="dist-bar-fill" style="width: 0%; background-color: ${color}; color: ${color}"></div>
            </div>
        `;
        
        distanceList.appendChild(item);
        
        setTimeout(() => {
            item.querySelector('.dist-bar-fill').style.width = `${pct}%`;
        }, 100);
    });
}

function resetUpload() {
    fileInput.value = '';
    dropZone.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    resultsBox.classList.add('hidden');
    appState.predictedPoint = null;
}

// --- Webcam & MediaPipe Facial Scan Handlers ---
function setupWebcamControls() {
    btnStartWebcam.addEventListener('click', startWebcam);
    btnStopWebcam.addEventListener('click', stopWebcam);
}

async function initMediaPipe() {
    if (appState.faceLandmarker) return; // Already loaded
    
    webcamStatusOverlay.classList.remove('hidden');
    webcamStatusText.innerText = "Loading MediaPipe WebAssembly...";
    
    try {
        const visionBundle = await createFilesetResolver();
        appState.faceLandmarker = await createFaceLandmarker(visionBundle);
        webcamStatusText.innerText = "Ready to scan face.";
        setTimeout(() => {
            if (!appState.webcamActive) {
                // Show prompt to start camera
                webcamStatusText.innerText = "Camera scan is ready. Click 'Start Camera Scan'.";
            }
        }, 500);
    } catch (error) {
        console.error("Error loading MediaPipe FaceLandmarker:", error);
        webcamStatusText.innerHTML = `<span style="color: #ef4444;"><i class="fa-solid fa-circle-exclamation"></i> Wasm compilation failed.</span>`;
    }
}

// Helpers wrappers for MediaPipe Vision Web SDK
async function createFilesetResolver() {
    return await createFilesetResolverForVision(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
}

async function createFilesetResolverForVision(url) {
    // MediaPipe Vision Tasks resolver global namespace wrapper from Vision bundle script
    if (typeof Module !== 'undefined' && typeof createFilesetResolver !== 'undefined') {
        return await createFilesetResolver(url);
    }
    // Standard module import mapping fallback
    const { FilesetResolver } = dummy_mediapipe_bundle_vision_namespace();
    return await FilesetResolver.forVisionTasks(url);
}

function dummy_mediapipe_bundle_vision_namespace() {
    // If the bundle loaded, it injects "mp" or "mediapipe" globals on window
    const mpTasks = window.mp || window.mediapipe || {};
    return mpTasks.tasks || {};
}

async function createFaceLandmarker(visionBundle) {
    const mpTasks = window.mp || window.mediapipe || {};
    const FaceLandmarker = mpTasks.tasks.FaceLandmarker;
    
    return await FaceLandmarker.createFromOptions(visionBundle, {
        baseOptions: {
            modelAssetPath: "/face_landmarker.task",
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: false,
        runningMode: "VIDEO",
        numFaces: 1
    });
}

async function startWebcam() {
    if (!appState.faceLandmarker) {
        await initMediaPipe();
    }
    
    webcamStatusOverlay.classList.remove('hidden');
    webcamStatusText.innerText = "Accessing camera feed...";
    
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        };
        
        appState.stream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamVideo.srcObject = appState.stream;
        
        // Start play stream
        webcamVideo.addEventListener('loadedmetadata', () => {
            webcamVideo.play();
            appState.webcamActive = true;
            btnStartWebcam.classList.add('hidden');
            btnStopWebcam.classList.remove('hidden');
            webcamStatusOverlay.classList.add('hidden');
            
            // Adjust overlay sizes
            adjustOverlayCanvas();
            
            // Trigger processing loop
            requestAnimationFrame(processWebcamFrame);
        });
        
    } catch (error) {
        console.error("Camera access denied:", error);
        webcamStatusText.innerHTML = `<span style="color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Camera permission denied. Check your browser security.</span>`;
    }
}

function stopWebcam() {
    if (appState.stream) {
        appState.stream.getTracks().forEach(track => track.stop());
    }
    
    webcamVideo.srcObject = null;
    appState.webcamActive = false;
    btnStopWebcam.classList.add('hidden');
    btnStartWebcam.classList.remove('hidden');
    webcamStatusOverlay.classList.remove('hidden');
    webcamStatusText.innerText = "Camera scan is stopped.";
    
    // Clear overlay
    webcamOverlayCtx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
    
    // Hide prediction results
    resultsBox.classList.add('hidden');
    appState.predictedPoint = null;
}

function adjustOverlayCanvas() {
    webcamOverlay.width = webcamVideo.clientWidth;
    webcamOverlay.height = webcamVideo.clientHeight;
}

// MediaPipe frame loop
function processWebcamFrame() {
    if (!appState.webcamActive || !appState.faceLandmarker) return;
    
    // Sync sizes if wrapper changed sizes
    if (webcamOverlay.width !== webcamVideo.clientWidth) {
        adjustOverlayCanvas();
    }
    
    let now = performance.now();
    let result = null;
    
    try {
        if (webcamVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
            result = appState.faceLandmarker.detectForVideo(webcamVideo, now);
        }
    } catch (e) {
        console.error("Face landmarker tracking error:", e);
    }
    
    // Draw landmarks overlay
    webcamOverlayCtx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
    
    if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
        const landmarks = result.faceLandmarks[0];
        
        // Draw facial wireframe outlines (eyes, eyebrows, mouth)
        drawFaceWireframe(landmarks);
        
        // Check inference throttle
        const nowMs = Date.now();
        if (nowMs - appState.lastInferenceTime > appState.inferenceThrottleMs) {
            appState.lastInferenceTime = nowMs;
            
            if (result.faceBlendshapes && result.faceBlendshapes.length > 0) {
                const categories = result.faceBlendshapes[0].categories;
                const blendshapesDict = {};
                categories.forEach(item => {
                    blendshapesDict[item.categoryName] = item.score;
                });
                
                // Submit blendshape scores array to API
                submitBlendshapes(blendshapesDict);
            }
        }
    }
    
    // Recursive frame capture loop
    requestAnimationFrame(processWebcamFrame);
}

// Draw key connections (eyes, lips, brows) to give a neat scan effect
function drawFaceWireframe(landmarks) {
    const ctx = webcamOverlayCtx;
    const w = webcamOverlay.width;
    const h = webcamOverlay.height;
    
    ctx.strokeStyle = '#10b98188'; // Neon green outline
    ctx.lineWidth = 1;
    
    // Define helper to draw outlines
    function drawPath(indices, close = false) {
        ctx.beginPath();
        indices.forEach((idx, i) => {
            const pt = landmarks[idx];
            if (i === 0) ctx.moveTo(pt.x * w, pt.y * h);
            else ctx.lineTo(pt.x * w, pt.y * h);
        });
        if (close) ctx.closePath();
        ctx.stroke();
    }
    
    // Indices references for key face outlines
    const L_EYE_OUTLINE = [33, 160, 158, 133, 153, 144];
    const R_EYE_OUTLINE = [362, 385, 387, 263, 373, 380];
    const L_BROW = [70, 63, 105, 66, 107];
    const R_BROW = [300, 293, 334, 296, 336];
    const LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
    const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
    
    drawPath(L_EYE_OUTLINE, true);
    drawPath(R_EYE_OUTLINE, true);
    drawPath(L_BROW);
    drawPath(R_BROW);
    drawPath(LIPS_OUTER, true);
    
    // Draw bounding box
    const xs = landmarks.map(p => p.x * w);
    const ys = landmarks.map(p => p.y * h);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    
    // Draw square scanning box
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(minX - 10, minY - 15, (maxX - minX) + 20, (maxY - minY) + 30);
    
    // Scanner Text tag
    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 10px Plus Jakarta Sans';
    ctx.fillText("AI REALTIME SCAN", minX - 10, minY - 22);
}

async function submitBlendshapes(blendshapesDict) {
    try {
        const response = await fetch('/api/predict_blendshapes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ blendshapes: blendshapesDict })
        });
        
        if (!response.ok) throw new Error("Blendshape API prediction failed");
        
        const result = await response.json();
        showPredictionResults(result);
        
    } catch (e) {
        console.error("Failed to run blendshape prediction:", e);
    }
}
