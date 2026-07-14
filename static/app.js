// Application Configuration and State
const CLUSTER_COLORS = [
    '#38bdf8', // Cluster 0: Bright Neon Blue
    '#4ade80', // Cluster 1: Bright Neon Green
    '#f43f5e', // Cluster 2: Bright Neon Coral/Rose
    '#c084fc', // Cluster 3: Bright Neon Purple
    '#facc15', // Cluster 4: Bright Neon Yellow
    '#22d3ee', // Cluster 5: Bright Neon Cyan
    '#ff79c6'  // Cluster 6: Bright Neon Hot Pink
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
    "happy": "#4ade80",
    "sad": "#38bdf8",
    "anxiety": "#c084fc",
    "angry": "#f43f5e",
    "surprised": "#facc15",
    "disgust": "#22d3ee",
    "neutral": "#818cf8"
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
        ctx.fillStyle = CLUSTER_COLORS[pt.cluster]; // Solid bright colors
        ctx.beginPath();
        ctx.arc(screenPt.x, screenPt.y, 5.0, 0, 2 * Math.PI); // Solid 5px radius
        ctx.fill();
        
        // Add dark stroke around points to prevent bleeding and improve contrast/separation
        ctx.strokeStyle = 'rgba(11, 9, 20, 0.8)';
        ctx.lineWidth = 1.2;
        ctx.stroke();
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
    
    // Save to appState to color the face mesh
    const emotion = result.emotion_label;
    appState.predictedEmotion = emotion;
    
    // Render Cluster Badge
    const clusterIdx = result.predicted_cluster;
    predictedBadge.innerText = `${CLUSTER_NAMES[clusterIdx]} (Cluster ${clusterIdx})`;
    predictedBadge.style.backgroundColor = CLUSTER_COLORS[clusterIdx];
    predictedBadge.style.boxShadow = `0 0 20px ${CLUSTER_COLORS[clusterIdx]}50`;
    
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
        "/static/mediapipe_wasm"
    );
}

async function createFilesetResolverForVision(url) {
    // Resolve from global 'vision' namespace (vision_bundle.js CDN)
    if (window.vision && window.vision.FilesetResolver) {
        return await window.vision.FilesetResolver.forVisionTasks(url);
    }
    // Fallback: Resolve from window.mp or window.mediapipe
    const mpTasks = window.mp || window.mediapipe || {};
    const tasksNamespace = mpTasks.tasks || window.vision || {};
    if (tasksNamespace.FilesetResolver) {
        return await tasksNamespace.FilesetResolver.forVisionTasks(url);
    }
    // Fallback for global createFilesetResolver helper
    if (typeof createFilesetResolver !== 'undefined') {
        return await createFilesetResolver(url);
    }
    throw new Error("FilesetResolver is not defined in any loaded namespace.");
}

async function createFaceLandmarker(visionBundle) {
    const mpTasks = window.mp || window.mediapipe || {};
    const tasksNamespace = mpTasks.tasks || window.vision || {};
    const FaceLandmarker = tasksNamespace.FaceLandmarker;
    
    if (!FaceLandmarker) {
        throw new Error("FaceLandmarker is not defined in any loaded namespace.");
    }
    
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
        
        // Match active emotion color or fallback to indigo accent
        const emotionColor = EMOTION_COLORS[appState.predictedEmotion] || '#6366f1';
        
        // Draw the full face mesh point cloud and connections matching Classification layout
        drawFaceMesh(landmarks, emotionColor);
        
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

// Visual face connections mapping from the Classification visualizer
const FACE_CONTOUR = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109
];
const LIPS = [
  78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
  308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
  308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
];
const L_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466];
const R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const L_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const R_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];

function drawFaceMesh(landmarks, emotionColor) {
    const ctx = webcamOverlayCtx;
    const w = webcamOverlay.width;
    const h = webcamOverlay.height;
    
    // 1. Draw delicate point cloud (all 478 landmarks)
    ctx.fillStyle = "rgba(224, 242, 254, 0.22)"; // Translucent light blue
    for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        ctx.beginPath();
        ctx.arc(pt.x * w, pt.y * h, 1.2, 0, 2 * Math.PI);
        ctx.fill();
    }
    
    // 2. Draw connections (outlines for lips, eyes, brows, facial contours)
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.shadowBlur = 6;
    ctx.shadowColor = emotionColor;
    ctx.strokeStyle = emotionColor;
    
    const drawPath = (indices, close = false) => {
        if (indices.length === 0) return;
        ctx.beginPath();
        const first = landmarks[indices[0]];
        ctx.moveTo(first.x * w, first.y * h);
        for (let i = 1; i < indices.length; i++) {
            const pt = landmarks[indices[i]];
            ctx.lineTo(pt.x * w, pt.y * h);
        }
        if (close) ctx.closePath();
        ctx.stroke();
    };
    
    drawPath(FACE_CONTOUR, false);
    drawPath(LIPS, true);
    drawPath(L_EYE, true);
    drawPath(R_EYE, true);
    drawPath(L_BROW, false);
    drawPath(R_BROW, false);
    
    // 3. Draw glowing pupils
    ctx.fillStyle = emotionColor;
    ctx.shadowBlur = 8;
    ctx.shadowColor = "#FFFFFF";
    
    const lPupil = landmarks[468];
    const rPupil = landmarks[473];
    if (lPupil) {
        ctx.beginPath();
        ctx.arc(lPupil.x * w, lPupil.y * h, 2.5, 0, 2 * Math.PI);
        ctx.fill();
    }
    if (rPupil) {
        ctx.beginPath();
        ctx.arc(rPupil.x * w, rPupil.y * h, 2.5, 0, 2 * Math.PI);
        ctx.fill();
    }
    ctx.shadowBlur = 0; // Reset
    
    // 4. Draw bounding box with padding and header tag
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (const lm of landmarks) {
        if (lm.x < minX) minX = lm.x;
        if (lm.x > maxX) maxX = lm.x;
        if (lm.y < minY) minY = lm.y;
        if (lm.y > maxY) maxY = lm.y;
    }
    
    let boxX = minX * w;
    let boxY = minY * h;
    let boxW = (maxX - minX) * w;
    let boxH = (maxY - minY) * h;
    
    // Padding (12%)
    const paddingX = boxW * 0.12;
    const paddingY = boxH * 0.12;
    boxX -= paddingX;
    boxY -= paddingY;
    boxW += paddingX * 2;
    boxH += paddingY * 2;
    
    // Clamp to canvas borders
    boxX = Math.max(0, boxX);
    boxY = Math.max(0, boxY);
    boxW = Math.min(w - boxX, boxW);
    boxH = Math.min(h - boxY, boxH);
    
    // Draw box
    ctx.strokeStyle = emotionColor;
    ctx.lineWidth = 3;
    ctx.shadowBlur = 8;
    ctx.shadowColor = emotionColor;
    ctx.strokeRect(boxX, boxY, boxW, boxH);
    ctx.shadowBlur = 0;
    
    // Label header
    const labelHeight = 22;
    let labelY = boxY - labelHeight;
    if (labelY < 0) labelY = boxY;
    
    ctx.fillStyle = emotionColor;
    ctx.fillRect(boxX, labelY, boxW, labelHeight);
    
    ctx.fillStyle = "#FFFFFF";
    ctx.font = "bold 11px 'Plus Jakarta Sans', sans-serif";
    ctx.textBaseline = "middle";
    ctx.fillText(`👤 FACE SCAN: ${appState.predictedEmotion ? appState.predictedEmotion.toUpperCase() : "DETECTING"}`, boxX + 8, labelY + labelHeight / 2);
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
