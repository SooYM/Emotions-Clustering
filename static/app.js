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

let appState = {
    coords: [],          // Downsampled coordinates
    representatives: {}, // Representatives per cluster
    activeTab: 0,        // Selected cluster tab
    predictedPoint: null,// Coords of predicted image {x, y, cluster}
    animationFrame: null,// Animation frame ID
    pulseRadius: 6,
    pulseGrowing: true,
    scale: 1,
    offsetX: 0,
    offsetY: 0
};

// DOM Elements
const canvas = document.getElementById('clusterPlot');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('plotTooltip');
const tooltipImg = document.getElementById('tooltipImg');
const tooltipText = document.getElementById('tooltipText');
const plotLegend = document.getElementById('plotLegend');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const btnReset = document.getElementById('btnReset');
const processStatus = document.getElementById('processStatus');
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
        // Set CSS variables for tab coloring
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
        
        // Show file name in tooltip on hover inside gallery
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
    
    // Determine bounds
    if (appState.coords.length === 0) return;
    
    const xs = appState.coords.map(c => c.x);
    const ys = appState.coords.map(c => c.y);
    
    mapBounds.minX = Math.min(...xs);
    mapBounds.maxX = Math.max(...xs);
    mapBounds.minY = Math.min(...ys);
    mapBounds.maxY = Math.max(...ys);
    
    // Pad bounds slightly to avoid edge clipping
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

// Transform Data to Screen Coordinates
function toScreen(x, y) {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    
    const screenX = ((x - mapBounds.minX) / (mapBounds.maxX - mapBounds.minX)) * w;
    // Invert Y axis for standard cartesian visualization
    const screenY = h - (((y - mapBounds.minY) / (mapBounds.maxY - mapBounds.minY)) * h);
    
    return { x: screenX, y: screenY };
}

// Transform Screen Coordinates to Latent Space
function toLatent(screenX, screenY) {
    const w = canvas.width / window.devicePixelRatio;
    const h = canvas.height / window.devicePixelRatio;
    
    const x = mapBounds.minX + (screenX / w) * (mapBounds.maxX - mapBounds.minX);
    const y = mapBounds.minY + ((h - screenY) / h) * (mapBounds.maxY - mapBounds.minY);
    
    return { x, y };
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
        ctx.fillStyle = CLUSTER_COLORS[pt.cluster] + 'aa'; // With transparency
        ctx.beginPath();
        ctx.arc(screenPt.x, screenPt.y, 3, 0, 2 * Math.PI);
        ctx.fill();
    });
    
    // Draw predicted point if active
    if (appState.predictedPoint) {
        const pPt = toScreen(appState.predictedPoint.x, appState.predictedPoint.y);
        const color = CLUSTER_COLORS[appState.predictedPoint.cluster];
        
        // Animated Pulse Ring
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(pPt.x, pPt.y, appState.pulseRadius, 0, 2 * Math.PI);
        ctx.stroke();
        
        // Glowing Core
        const gradient = ctx.createRadialGradient(pPt.x, pPt.y, 1, pPt.x, pPt.y, 7);
        gradient.addColorStop(0, '#ffffff');
        gradient.addColorStop(0.3, color);
        gradient.addColorStop(1, 'transparent');
        ctx.fillStyle = gradient;
        
        ctx.beginPath();
        ctx.arc(pPt.x, pPt.y, 7, 0, 2 * Math.PI);
        ctx.fill();
    }
}

// Animation loop
function startAnimationLoop() {
    function animate() {
        // Update pulse radius
        if (appState.pulseGrowing) {
            appState.pulseRadius += 0.3;
            if (appState.pulseRadius >= 18) appState.pulseGrowing = false;
        } else {
            appState.pulseRadius -= 0.3;
            if (appState.pulseRadius <= 6) appState.pulseGrowing = true;
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
        
        // Find nearest point
        let nearest = null;
        let minDist = 18; // Max hover distance in pixels
        
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
            
            // Position tooltip next to cursor
            const tooltipW = tooltip.offsetWidth || 90;
            const tooltipH = tooltip.offsetHeight || 110;
            let left = mouseX + 15;
            let top = mouseY - tooltipH / 2;
            
            // Check canvas bounds
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
    
    // Navigate on click
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        let nearest = null;
        let minDist = 20;
        
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

// --- Upload & Prediction Handlers ---
function setupUploadHandlers() {
    // Drag & Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });
    
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });
    
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadedFile(e.target.files[0]);
        }
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
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dropZone.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        processStatus.classList.remove('hidden');
        resultsBox.classList.add('hidden');
        
        // Execute Prediction API
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
        
        if (!response.ok) throw new Error("Inference failed");
        
        const result = await response.json();
        
        // Hide loader & show results
        processStatus.classList.add('hidden');
        resultsBox.classList.remove('hidden');
        
        // Render Cluster Badge
        const clusterIdx = result.predicted_cluster;
        predictedBadge.innerText = `${CLUSTER_NAMES[clusterIdx]} (Cluster ${clusterIdx})`;
        predictedBadge.style.backgroundColor = CLUSTER_COLORS[clusterIdx];
        predictedBadge.style.boxShadow = `0 0 20px ${CLUSTER_COLORS[clusterIdx]}50`;
        
        // Set prediction point to overlay on plot
        appState.predictedPoint = {
            x: result.x,
            y: result.y,
            cluster: clusterIdx
        };
        
        // Update Representative Gallery Tab to Match Prediction
        selectTab(clusterIdx);
        
        // Render Distances
        renderDistances(result.distances, clusterIdx);
        
    } catch (error) {
        console.error("Error running prediction:", error);
        processStatus.innerHTML = `<span style="color: #ef4444;"><i class="fa-solid fa-triangle-exclamation"></i> Analysis failed. Please try again.</span>`;
    }
}

function renderDistances(distances, predictedClusterIdx) {
    distanceList.innerHTML = '';
    
    // Find max distance to normalize progress bars (furthest centroid = shortest bar)
    const distValues = Object.values(distances);
    const maxDist = Math.max(...distValues);
    const minDist = Math.min(...distValues);
    
    // Sort clusters by distance (closest first)
    const sortedClusters = Object.keys(distances).map(k => parseInt(k)).sort((a, b) => distances[a] - distances[b]);
    
    sortedClusters.forEach(idx => {
        const dist = distances[idx];
        
        // Calculate similarity percentage (closeness)
        // Normalize: closest is 100%, furthest is near 10%
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
        
        // Animate fill width after DOM insertion
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
    
    // Clear prediction dot
    appState.predictedPoint = null;
}
