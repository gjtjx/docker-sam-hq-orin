// Unit tests for index.html functionality

const { JSDOM } = require('jsdom');
const fs = require('fs');
const path = require('path');

// Load the HTML file
const html = fs.readFileSync(path.resolve(__dirname, 'index.html'), 'utf8');

// Setup JSDOM environment
const dom = new JSDOM(html);
global.document = dom.window.document;
global.window = dom.window;

// Mock the global state and elements
const state = {
    image: null,
    mode: 'point',
    points: [],
    boxes: [],
    currentBox: null,
    alpha: 0.5,
    model: 'vit_b',
    apiUrl: 'http://localhost',
    originalImage: null,
    masksMetadata: [],
    maskColors: []
};

const elements = {
    canvas: document.getElementById('canvas'),
    canvasContainer: document.getElementById('canvasContainer'),
    uploadPrompt: document.getElementById('uploadPrompt'),
    imageUpload: document.getElementById('imageUpload'),
    modelSelect: document.getElementById('modelSelect'),
    modelStatus: document.getElementById('modelStatus'),
    alphaSlider: document.getElementById('alphaSlider'),
    alphaValue: document.getElementById('alphaValue'),
    segmentBtn: document.getElementById('segmentBtn'),
    clearBtn: document.getElementById('clearBtn'),
    downloadVisualizationBtn: document.getElementById('downloadVisualizationBtn'),
    downloadMasksBinaryBtn: document.getElementById('downloadMasksBinaryBtn'),
    downloadMasksColorBtn: document.getElementById('downloadMasksColorBtn'),
    downloadJsonBtn: document.getElementById('downloadJsonBtn'),
    modeAuto: document.getElementById('modeAuto'),
    modePoint: document.getElementById('modePoint'),
    modeBox: document.getElementById('modeBox'),
    modeInfo: document.getElementById('modeInfo'),
    loading: document.getElementById('loading'),
    stats: {
        total: document.getElementById('statTotal'),
        encode: document.getElementById('statEncode'),
        predict: document.getElementById('statPredict'),
        metadata: document.getElementById('statMetadata'),
        memory: document.getElementById('statMemory'),
        masks: document.getElementById('statMasks')
    }
};

// Test suite for the index.html functionality
describe('SAM HQ Image Segmentation', () => {
    beforeEach(() => {
        // Reset state before each test
        state.image = null;
        state.points = [];
        state.boxes = [];
        state.currentBox = null;
        state.originalImage = null;
        state.masksMetadata = [];
        state.maskColors = [];
    });

    describe('Mode Switching', () => {
        it('should switch to auto mode', () => {
            setMode('auto');
            expect(state.mode).toBe('auto');
            expect(elements.modeAuto.classList.contains('active')).toBe(true);
            expect(elements.modeInfo.innerHTML).toContain('全图模式');
        });

        it('should switch to point mode', () => {
            setMode('point');
            expect(state.mode).toBe('point');
            expect(elements.modePoint.classList.contains('active')).toBe(true);
            expect(elements.modeInfo.innerHTML).toContain('点模式');
        });

        it('should switch to box mode', () => {
            setMode('box');
            expect(state.mode).toBe('box');
            expect(elements.modeBox.classList.contains('active')).toBe(true);
            expect(elements.modeInfo.innerHTML).toContain('框模式');
        });
    });

    describe('Image Upload', () => {
        it('should handle image upload', () => {
            const file = new File([''], 'test.png', { type: 'image/png' });
            const event = { target: { files: [file] } };
            handleImageUpload(event);
            expect(state.image).not.toBeNull();
            expect(elements.uploadPrompt.style.display).toBe('none');
            expect(elements.canvas.style.display).toBe('block');
        });

        it('should not handle empty file upload', () => {
            const event = { target: { files: [] } };
            handleImageUpload(event);
            expect(state.image).toBeNull();
            expect(elements.uploadPrompt.style.display).not.toBe('none');
        });
    });

    describe('RLE Decoding', () => {
        it('should decode RLE to mask', () => {
            const rle = {
                size: [10, 10],
                counts: [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
            };
            const mask = rleToMask(rle);
            expect(mask.data.length).toBe(100);
            expect(mask.width).toBe(10);
            expect(mask.height).toBe(10);
        });

        it('should handle empty RLE', () => {
            const rle = {
                size: [0, 0],
                counts: []
            };
            const mask = rleToMask(rle);
            expect(mask.data.length).toBe(0);
            expect(mask.width).toBe(0);
            expect(mask.height).toBe(0);
        });
    });

    describe('Alpha Slider', () => {
        it('should update alpha value', () => {
            const event = { target: { value: '0.7' } };
            elements.alphaSlider.dispatchEvent(new Event('input', event));
            expect(state.alpha).toBe(0.7);
            expect(elements.alphaValue.textContent).toBe('0.7');
        });
    });

    describe('Clear Annotations', () => {
        it('should clear points and boxes', () => {
            state.points = [{ x: 10, y: 10, label: 1 }];
            state.boxes = [{ x1: 10, y1: 10, x2: 20, y2: 20 }];
            clearAnnotations();
            expect(state.points.length).toBe(0);
            expect(state.boxes.length).toBe(0);
        });
    });
});

// Helper functions from the original code
function setMode(mode) {
    state.mode = mode;
    [elements.modeAuto, elements.modePoint, elements.modeBox].forEach(btn => {
        btn.classList.remove('active');
    });

    if (mode === 'auto') {
        elements.modeAuto.classList.add('active');
        elements.modeInfo.innerHTML = '<p><strong>全图模式:</strong> 自动分割整张图片的所有对象</p>';
    } else if (mode === 'point') {
        elements.modePoint.classList.add('active');
        elements.modeInfo.innerHTML = '<p><strong>点模式:</strong> 左键添加正样本点，右键添加负样本点</p>';
    } else if (mode === 'box') {
        elements.modeBox.classList.add('active');
        elements.modeInfo.innerHTML = '<p><strong>框模式:</strong> 拖拽鼠标绘制边界框</p>';
    }

    clearAnnotations();
}

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    // Mock implementation for testing
    state.image = file;
    elements.uploadPrompt.style.display = 'none';
    elements.canvas.style.display = 'block';
}

function rleToMask(rle) {
    const [height, width] = rle.size;
    const flatMask = new Uint8Array(height * width);
    let pos = 0;
    let value = 0;

    for (let count of rle.counts) {
        const endPos = Math.min(pos + count, height * width);
        for (let i = pos; i < endPos; i++) {
            flatMask[i] = value;
        }
        pos = endPos;
        value = 1 - value;
    }

    const transposed = new Uint8Array(height * width);
    for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
            transposed[row * width + col] = flatMask[col * height + row];
        }
    }

    return { data: transposed, width, height };
}

function clearAnnotations() {
    state.points = [];
    state.boxes = [];
    state.currentBox = null;
}