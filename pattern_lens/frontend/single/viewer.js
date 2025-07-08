/**
 * Attention Pattern Viewer Module - Simplified version
 * Displays PNG directly with overlay for highlights
 */

class AttentionPatternViewer {
    constructor(containerId) {
        // Constants from config
        this.SIZE = CONFIG.layout.canvasSize;
        this.HM_highlight_strokeStyle = CONFIG.visualization.highlightStrokeStyle;
        this.HM_highlight_lineWidth = CONFIG.visualization.highlightLineWidth;
        this.HM_grid_strokeStyle = CONFIG.visualization.gridStrokeStyle;
        this.HM_grid_lineWidth = CONFIG.visualization.gridLineWidth;
        this.THROTTLE_DELAY = CONFIG.visualization.throttleDelay;

        // State
        this.n = 0;
        this.tokens = [];
        this.pixelSize = 0;
        this.cellBoundaries = [];
        this.lastMouseTime = 0;
        this.animationFrame = null;
        this.labelElements = { x: [], y: [] };
        this.pngImage = null;
        this.attentionMatrix = null; // Store the actual matrix data
        this.selectedCell = null; // { x, y } or null
        this.keyboardMode = false;
        this.keysPressed = new Set(); // Track multiple key presses
        this.keyRepeatInterval = null;

        // DOM elements
        this.container = document.getElementById(containerId);

        // Set up grid layout with constants
        this.container.style.gridTemplateColumns = `${CONFIG.layout.yLabelWidth}px ${CONFIG.layout.canvasSize}px`;
        this.container.style.gridTemplateRows = `${CONFIG.layout.canvasSize}px ${CONFIG.layout.xLabelHeight}px`;

        // Create main canvas for PNG display
        this.canvas = document.getElementById('heatmapCanvas');
        this.ctx = this.canvas.getContext('2d');

        // no image smoothing for pixelated
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.webkitImageSmoothingEnabled = false; // Safari
        this.ctx.mozImageSmoothingEnabled = false;   // Firefox
        this.ctx.msImageSmoothingEnabled = false;   // old Edge/IE

        // Create overlay canvas for highlights
        this.overlayCanvas = document.createElement('canvas');
        this.overlayCanvas.style.position = 'absolute';
        this.overlayCanvas.style.left = '0';
        this.overlayCanvas.style.top = '0';
        this.overlayCanvas.style.pointerEvents = 'none';
        this.overlayCanvas.style.zIndex = '10';
        this.overlayCtx = this.overlayCanvas.getContext('2d');

        // Add overlay to container
        this.canvas.parentElement.style.position = 'relative';
        this.canvas.parentElement.appendChild(this.overlayCanvas);

        this.tooltip = document.getElementById('tooltip');
        this.cellInfo = document.getElementById('cellInfo');
        this.tokensDisplay = document.getElementById('tokensDisplay');
        this.xLabelsContainer = document.getElementById('xLabels');
        this.yLabelsContainer = document.getElementById('yLabels');

        // Set up event listeners
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => this.handleMouseLeave());
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        document.addEventListener('keyup', (e) => this.handleKeyUp(e));

        // Set up token click handlers once
        this.tokensDisplay.addEventListener('click', (e) => this.handleTokenClick(e, 'k'));
        this.tokensDisplay.addEventListener('contextmenu', (e) => this.handleTokenClick(e, 'q'));
    }

    precalculateBoundaries() {
        this.cellBoundaries = [];
        for (let i = 0; i <= this.n; i++) {
            this.cellBoundaries.push(i * this.pixelSize);
        }
    }


    renderHighlights(hoverX, hoverY) {
        // Clear overlay
        this.overlayCtx.clearRect(0, 0, this.SIZE, this.SIZE);

        // Draw grid lines
        this.overlayCtx.strokeStyle = this.HM_grid_strokeStyle;
        this.overlayCtx.lineWidth = this.HM_grid_lineWidth;
        this.overlayCtx.beginPath();

        for (let i = 0; i <= this.n; i++) {
            const pos = this.cellBoundaries[i] + 0.5;
            // Horizontal line
            this.overlayCtx.moveTo(0, pos);
            this.overlayCtx.lineTo(this.SIZE, pos);
            // Vertical line
            this.overlayCtx.moveTo(pos, 0);
            this.overlayCtx.lineTo(pos, this.SIZE);
        }

        this.overlayCtx.stroke();

        // Draw highlights if hovering
        if (hoverX >= 0 && hoverY >= 0 && hoverX < this.n && hoverY < this.n) {
            const x1 = hoverX * this.pixelSize;
            const y1 = hoverY * this.pixelSize;

            // Highlight the cell's own borders (red)
            this.overlayCtx.strokeStyle = CONFIG.visualization.colors.kAxis;
            this.overlayCtx.lineWidth = this.HM_highlight_lineWidth;
            this.overlayCtx.strokeRect(x1, y1, this.pixelSize, this.pixelSize);

            // Highlight row (only to the left of hovered cell) - green for Q
            if (hoverX > 0) {
                this.overlayCtx.strokeStyle = CONFIG.visualization.colors.qAxis;
                this.overlayCtx.beginPath();
                this.overlayCtx.moveTo(0, y1);
                this.overlayCtx.lineTo(x1, y1);
                this.overlayCtx.moveTo(0, y1 + this.pixelSize);
                this.overlayCtx.lineTo(x1, y1 + this.pixelSize);
                this.overlayCtx.stroke();
            }

            // Highlight column (only below hovered cell) - red for K
            if (hoverY < this.n - 1) {
                this.overlayCtx.strokeStyle = CONFIG.visualization.colors.kAxis;
                this.overlayCtx.beginPath();
                this.overlayCtx.moveTo(x1, y1 + this.pixelSize);
                this.overlayCtx.lineTo(x1, this.SIZE);
                this.overlayCtx.moveTo(x1 + this.pixelSize, y1 + this.pixelSize);
                this.overlayCtx.lineTo(x1 + this.pixelSize, this.SIZE);
                this.overlayCtx.stroke();
            }
        }
    }

    createAxisLabels() {
        // Clear existing labels
        this.xLabelsContainer.innerHTML = '';
        this.yLabelsContainer.innerHTML = '';
        this.labelElements.x = [];
        this.labelElements.y = [];

        // Hide labels if too many tokens
        if (this.n > CONFIG.layout.maxTokensForLabels) {
            // Adjust grid layout to account for missing labels
            this.container.style.gridTemplateColumns = `0px ${CONFIG.layout.canvasSize}px`;
            this.container.style.gridTemplateRows = `${CONFIG.layout.canvasSize}px 0px`;
            return;
        }

        // Reset grid layout for labels
        this.container.style.gridTemplateColumns = `${CONFIG.layout.yLabelWidth}px ${CONFIG.layout.canvasSize}px`;
        this.container.style.gridTemplateRows = `${CONFIG.layout.canvasSize}px ${CONFIG.layout.xLabelHeight}px`;

        this.tokens.forEach((token) => {
            const displayToken = this.renderWhitespace(token);

            const xLabel = document.createElement('div');
            xLabel.className = 'label x-label';
            xLabel.textContent = displayToken;
            xLabel.style.width = this.pixelSize + 'px';
            xLabel.style.height = CONFIG.layout.xLabelHeight + 'px';
            this.xLabelsContainer.appendChild(xLabel);
            this.labelElements.x.push(xLabel);

            const yLabel = document.createElement('div');
            yLabel.className = 'label y-label';
            yLabel.textContent = displayToken;
            yLabel.style.width = CONFIG.layout.yLabelWidth + 'px';
            yLabel.style.height = this.pixelSize + 'px';
            yLabel.style.lineHeight = this.pixelSize + 'px';
            this.yLabelsContainer.appendChild(yLabel);
            this.labelElements.y.push(yLabel);
        });
    }

    updateHighlights(x, y) {
        // Update label highlights
        this.labelElements.x.forEach(label => label.classList.remove('highlight-k'));
        this.labelElements.y.forEach(label => label.classList.remove('highlight-q'));

        if (x >= 0 && x < this.n && y >= 0 && y < this.n) {
            if (this.labelElements.x.length > 0) {
                this.labelElements.x[x].classList.add('highlight-k');
                this.labelElements.y[y].classList.add('highlight-q');
            }
        }

        // Update token highlights
        this.updateTokenHighlights(x, y);

        // Render highlights
        this.renderHighlights(x, y);
    }

    updateTokenHighlights(x, y) {
        const tokens = this.tokensDisplay.querySelectorAll('.token');
        tokens.forEach((token, idx) => {
            token.classList.remove('highlight-k', 'highlight-q');
            token.style.backgroundColor = '';

            if (idx === x) {
                token.classList.add('highlight-k');
            }
            if (idx === y) {
                token.classList.add('highlight-q');
            }

            // Add value-based highlighting based on attention values
            if (x >= 0 && y >= 0 && x < this.n && y < this.n) {
                // Get the attention value for this token from the selected row
                const attentionValue = this.getPixelValue(idx, y);
                if (attentionValue > 0) {
                    // Apply intensity-based background color
                    const intensity = Math.min(1, attentionValue * CONFIG.visualization.tokenHighlight.intensityScale);
                    const alpha = intensity * CONFIG.visualization.tokenHighlight.maxOpacity;
                    token.style.backgroundColor = CONFIG.visualization.tokenHighlight.backgroundColor.replace('{alpha}', alpha);
                }
            }
        });
    }

    getPixelValue(x, y) {
        // Get value from the attention matrix
        if (!this.attentionMatrix || y >= this.attentionMatrix.length || x >= this.attentionMatrix[y].length) {
            return 0;
        }
        return this.attentionMatrix[y][x];
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * this.n);
        const y = Math.floor((e.clientY - rect.top) / rect.height * this.n);

        if (x >= 0 && x < this.n && y >= 0 && y < this.n) {
            // Update cell info with hover position
            this.updateCellInfo(x, y, false);

            // Don't update highlights in keyboard mode
            if (!this.keyboardMode) {
                this.updateHighlightsFromMouse(e);
            }
        }
    }

    updateHighlightsFromMouse(e) {

        const now = Date.now();
        if (now - this.lastMouseTime < this.THROTTLE_DELAY) {
            return;
        }
        this.lastMouseTime = now;

        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * this.n);
        const y = Math.floor((e.clientY - rect.top) / rect.height * this.n);

        if (x >= 0 && x < this.n && y >= 0 && y < this.n) {
            // Cancel any pending animation frame
            if (this.animationFrame) {
                cancelAnimationFrame(this.animationFrame);
            }

            // Schedule highlight update
            this.animationFrame = requestAnimationFrame(() => {
                this.updateHighlights(x, y);
                this.animationFrame = null;
            });
        }
    }


    handleMouseLeave() {
        // Clear cell info if not in keyboard mode
        if (!this.keyboardMode) {
            this.cellInfo.innerHTML = '';
        }

        // In keyboard mode, don't clear highlights
        if (this.keyboardMode) {
            return;
        }

        // Cancel any pending animation frame
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }

        // Remove highlights
        this.labelElements.x.forEach(label => label.classList.remove('highlight-k'));
        this.labelElements.y.forEach(label => label.classList.remove('highlight-q'));

        // Clear token highlights
        this.updateTokenHighlights(-1, -1);

        // Render without highlights
        this.renderHighlights(-1, -1);
    }

    handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = Math.floor((e.clientX - rect.left) / rect.width * this.n);
        const y = Math.floor((e.clientY - rect.top) / rect.height * this.n);

        if (x >= 0 && x < this.n && y >= 0 && y < this.n) {
            if (this.keyboardMode) {
                // Exit keyboard mode on second click
                this.keyboardMode = false;
                this.selectedCell = null;
                this.cellInfo.innerHTML = '';
                // Let mouse position take over
                this.handleMouseMove(e);
            } else {
                // Enter keyboard mode and select cell
                this.keyboardMode = true;
                this.selectedCell = { x, y };
                this.updateHighlights(x, y);
                this.updateCellInfo(x, y);
            }
        }
    }

    handleKeyDown(e) {
        // Only handle in keyboard mode
        if (!this.keyboardMode || !this.selectedCell) {
            return;
        }

        // Track key press
        this.keysPressed.add(e.key);

        if (e.key === 'Escape') {
            // Exit keyboard mode
            this.keyboardMode = false;
            this.selectedCell = null;
            this.updateHighlights(-1, -1);
            this.cellInfo.innerHTML = '';
            this.keysPressed.clear();
            if (this.keyRepeatInterval) {
                clearInterval(this.keyRepeatInterval);
                this.keyRepeatInterval = null;
            }
            return;
        }

        // Start continuous movement if not already running
        if (!this.keyRepeatInterval && this.isArrowKey(e.key)) {
            e.preventDefault();
            // Add initial delay before continuous movement
            this.moveSelection(); // Initial move
            setTimeout(() => {
                if (this.hasArrowKeyPressed() && !this.keyRepeatInterval) {
                    this.keyRepeatInterval = setInterval(() => this.moveSelection(), 100);
                }
            }, 300); // 300ms delay before repeat
        }
    }

    handleKeyUp(e) {
        this.keysPressed.delete(e.key);

        // Stop continuous movement if no arrow keys pressed
        if (this.keyRepeatInterval && !this.hasArrowKeyPressed()) {
            clearInterval(this.keyRepeatInterval);
            this.keyRepeatInterval = null;
        }
    }

    isArrowKey(key) {
        return ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(key);
    }

    hasArrowKeyPressed() {
        return Array.from(this.keysPressed).some(key => this.isArrowKey(key));
    }

    moveSelection() {
        if (!this.selectedCell) return;

        let dx = 0, dy = 0;
        const step = this.keysPressed.has('Control') ? CONFIG.visualization.keyboard.ctrlMoveStep : CONFIG.visualization.keyboard.moveStep;

        if (this.keysPressed.has('ArrowLeft')) dx -= step;
        if (this.keysPressed.has('ArrowRight')) dx += step;
        if (this.keysPressed.has('ArrowUp')) dy -= step;
        if (this.keysPressed.has('ArrowDown')) dy += step;

        const newX = Math.max(0, Math.min(this.n - 1, this.selectedCell.x + dx));
        const newY = Math.max(0, Math.min(this.n - 1, this.selectedCell.y + dy));

        if (newX !== this.selectedCell.x || newY !== this.selectedCell.y) {
            this.selectedCell = { x: newX, y: newY };
            this.updateHighlights(newX, newY);
            this.updateCellInfo(newX, newY);
        }
    }

    updateCellInfo(x, y) {
        if (x >= 0 && x < this.n && y >= 0 && y < this.n) {
            const xToken = this.renderWhitespace(this.tokens[x]);
            const yToken = this.renderWhitespace(this.tokens[y]);
            const value = this.getPixelValue(x, y).toFixed(2);

            this.cellInfo.innerHTML = `
                <table class="cell-info-table">
                    <tr>
                        <td>K[${x}]: <span class="right">${xToken}</span></td>
                        <td>Q[${y}]: <span class="right">${yToken}</span></td>
                        <td>Value: <span class="right">${value}</span></td>
                    </tr>
                </table>
            `;
        }
    }

    renderWhitespace(token) {
        // Convert whitespace characters to visible symbols
        return token.replace(/ /g, '␣')
            .replace(/\t/g, '␉')
            .replace(/\n/g, '␤')
            .replace(/\r/g, '␍');
    }

    renderTokensDisplay() {
        // Create individual token spans for click handling
        const tokenSpans = this.tokens.map((token, idx) => {
            // Check if token is purely whitespace
            const isWhitespace = /^[\s\n\r\t]+$/.test(token);
            const displayToken = this.renderWhitespace(token);
            const className = isWhitespace ? 'token whitespace' : 'token';
            const span = `<span class="${className}" data-index="${idx}">${displayToken}</span>`;

            // Add line break after tokens that are purely newlines
            if (token === '\n') {
                return span + '<br>';
            }
            return span;
        }).join('');

        this.tokensDisplay.innerHTML = tokenSpans;
    }

    handleTokenClick(e, axis) {
        e.preventDefault();
        const tokenEl = e.target.closest('.token');
        if (!tokenEl) return;

        const index = parseInt(tokenEl.dataset.index);
        if (isNaN(index) || index < 0 || index >= this.n) return;

        // Enter keyboard mode if not already
        if (!this.keyboardMode) {
            this.keyboardMode = true;
        }

        // Update selection
        if (!this.selectedCell) {
            this.selectedCell = { x: 0, y: 0 };
        }

        if (axis === 'k') {
            this.selectedCell.x = index;
        } else {
            this.selectedCell.y = index;
        }

        this.updateHighlights(this.selectedCell.x, this.selectedCell.y);
        this.updateCellInfo(this.selectedCell.x, this.selectedCell.y);
    }

    normalizeTokens(tokens) {
        return tokens.map(token => {
            // Replace unicode sequences within tokens
            let normalized = token;
            normalized = normalized.replace(/\u0120/g, ' ');    // GPT-2 space token
            normalized = normalized.replace(/\u010a/g, '\n');   // GPT-2 newline token
            return normalized;
        });
    }

    async displayPattern(dataLoader, model, promptHash, layerIdx, headIdx) {
        // Load prompt metadata
        const metadata = await dataLoader.loadPromptMetadata(model, promptHash);
        // Add boundary tokens as specified in config
        const startTokens = CONFIG.data.tokenBoundary.start || [];
        const endTokens = CONFIG.data.tokenBoundary.end || [];
        const tokensWithBounds = startTokens.concat(metadata.tokens).concat(endTokens);
        this.tokens = this.normalizeTokens(tokensWithBounds);
        this.n = this.tokens.length;
        this.pixelSize = this.SIZE / this.n;

        // Load attention matrix data and PNG path
        const { matrix, pngPath } = await dataLoader.loadAttentionPattern(model, promptHash, layerIdx, headIdx);
        this.attentionMatrix = matrix;


        return new Promise((resolve, reject) => {
            this.pngImage = new Image();
            this.pngImage.crossOrigin = 'anonymous';

            console.log(`Attempting to load PNG image from: ${pngPath}`);

            this.pngImage.onload = () => {
                console.log(`Successfully loaded PNG image: ${pngPath}`);
                console.log(`Image dimensions: ${this.pngImage.width}x${this.pngImage.height}`);
                // Precalculate boundaries
                this.precalculateBoundaries();

                // Set canvas dimensions to fixed size
                this.canvas.width = this.SIZE;
                this.canvas.height = this.SIZE;
                this.overlayCanvas.width = this.SIZE;
                this.overlayCanvas.height = this.SIZE;

                // every canvas reside restores defaults, so we remove the smoothing again
                this.ctx.imageSmoothingEnabled = false;
                this.ctx.webkitImageSmoothingEnabled = false; // Safari
                this.ctx.mozImageSmoothingEnabled = false; // Firefox
                this.ctx.msImageSmoothingEnabled = false; // old Edge/IE

                // Calculate pixel size based on fixed canvas size
                this.pixelSize = this.SIZE / this.n;

                // Render PNG scaled to canvas size
                this.ctx.drawImage(this.pngImage, 0, 0, this.SIZE, this.SIZE);

                // Initial render of grid
                this.renderHighlights(-1, -1);

                // Create labels
                this.createAxisLabels();

                // Render tokens display
                this.renderTokensDisplay();

                // Update page title
                document.title = `${model} L${layerIdx}H${headIdx} - ${promptHash.substring(0, 8)}`;

                resolve();
            };

            this.pngImage.onerror = (event) => {
                console.error(`Failed to load PNG image: ${pngPath}`);
                console.error('Error event:', event);
                reject(new Error(`Failed to load image: ${pngPath}`));
            };

            this.pngImage.src = pngPath;
        });
    }
}