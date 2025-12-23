/**
 * LFM2-VL Image Processor for WebGPU/ONNX Runtime Web
 *
 * Implements the image preprocessing logic from Lfm2VlImageProcessorFast:
 * 1. Split image into tiles (512x512)
 * 2. Extract 16x16 patches from each tile (32x32 = 1024 patches per tile)
 * 3. Flatten each patch to 768 values (16*16*3)
 * 4. Normalize: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1
 *
 * Output shapes match Python processor:
 * - pixel_values: [num_tiles, 1024, 768]
 * - pixel_attention_mask: [num_tiles, 1024]
 */

// Configuration from preprocessor_config.json
const CONFIG = {
  tileSize: 512,
  maxTiles: 10,
  minTiles: 2,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  rescaleFactor: 1 / 255,
  useThumbnail: true,
  patchSize: 16,  // Each patch is 16x16 pixels
  patchesPerTile: 32,  // 512 / 16 = 32 patches per side = 1024 per tile
  downsampleFactor: 2,
};

/**
 * Calculate optimal tile grid for an image
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {{rows: number, cols: number}} - Tile grid dimensions
 */
function calculateTileGrid(width, height) {
  const tileSize = CONFIG.tileSize;
  const maxTiles = CONFIG.maxTiles - (CONFIG.useThumbnail ? 1 : 0);

  // Calculate aspect ratio preserving tile layout
  const aspectRatio = width / height;

  // Try different grid sizes
  let bestRows = 1, bestCols = 1;
  let bestScore = Infinity;

  for (let tiles = 1; tiles <= maxTiles; tiles++) {
    for (let r = 1; r <= tiles; r++) {
      if (tiles % r !== 0) continue;
      const c = tiles / r;
      const gridAspect = c / r;
      const score = Math.abs(Math.log(gridAspect / aspectRatio));
      if (score < bestScore) {
        bestScore = score;
        bestRows = r;
        bestCols = c;
      }
    }
  }

  return { rows: bestRows, cols: bestCols };
}

/**
 * Process an image into flattened patches for VL model
 * @param {HTMLImageElement|HTMLCanvasElement} image - Input image
 * @returns {Promise<{pixelValues: Float32Array, attentionMask: BigInt64Array, numTiles: number, shape: number[]}>}
 */
export async function processImage(image) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  let width, height;
  if (image instanceof HTMLImageElement) {
    width = image.naturalWidth;
    height = image.naturalHeight;
  } else {
    width = image.width;
    height = image.height;
  }

  // Calculate tile grid
  const { rows, cols } = calculateTileGrid(width, height);
  const tileSize = CONFIG.tileSize;
  const patchSize = CONFIG.patchSize;
  const patchesPerSide = CONFIG.patchesPerTile;
  const patchesPerTile = patchesPerSide * patchesPerSide;  // 1024
  const patchDim = patchSize * patchSize * 3;  // 768

  // Number of tiles: grid + thumbnail
  const numTiles = rows * cols + (CONFIG.useThumbnail ? 1 : 0);

  // Output arrays
  const pixelValues = new Float32Array(numTiles * patchesPerTile * patchDim);
  const attentionMask = new BigInt64Array(numTiles * patchesPerTile);

  let tileIdx = 0;

  // Add thumbnail first if enabled
  if (CONFIG.useThumbnail) {
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = tileSize;
    thumbCanvas.height = tileSize;
    const thumbCtx = thumbCanvas.getContext('2d');

    // Resize to fit in tileSize x tileSize maintaining aspect ratio
    const scale = Math.min(tileSize / width, tileSize / height);
    const thumbWidth = Math.round(width * scale);
    const thumbHeight = Math.round(height * scale);
    const offsetX = Math.floor((tileSize - thumbWidth) / 2);
    const offsetY = Math.floor((tileSize - thumbHeight) / 2);

    thumbCtx.fillStyle = 'black';
    thumbCtx.fillRect(0, 0, tileSize, tileSize);
    thumbCtx.drawImage(image, offsetX, offsetY, thumbWidth, thumbHeight);

    const thumbData = thumbCtx.getImageData(0, 0, tileSize, tileSize);
    extractPatches(thumbData, pixelValues, attentionMask, tileIdx, offsetX, offsetY, thumbWidth, thumbHeight);
    tileIdx++;
  }

  // Calculate tile dimensions based on source image
  const srcTileWidth = width / cols;
  const srcTileHeight = height / rows;

  // Extract tiles from image
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const srcX = col * srcTileWidth;
      const srcY = row * srcTileHeight;

      // Create tile canvas at full tile size
      const tileCanvas = document.createElement('canvas');
      tileCanvas.width = tileSize;
      tileCanvas.height = tileSize;
      const tileCtx = tileCanvas.getContext('2d');

      // Fill with black padding
      tileCtx.fillStyle = 'black';
      tileCtx.fillRect(0, 0, tileSize, tileSize);

      // Draw portion of source image, scaled to tile size
      tileCtx.drawImage(
        image,
        srcX, srcY, srcTileWidth, srcTileHeight,  // source rect
        0, 0, tileSize, tileSize  // dest rect (full tile)
      );

      const tileData = tileCtx.getImageData(0, 0, tileSize, tileSize);
      extractPatches(tileData, pixelValues, attentionMask, tileIdx, 0, 0, tileSize, tileSize);
      tileIdx++;
    }
  }

  return {
    pixelValues,
    attentionMask,
    numTiles,
    shape: [numTiles, patchesPerTile, patchDim],  // [tiles, 1024, 768]
  };
}

/**
 * Extract 16x16 patches from a tile and flatten to the output array
 * @param {ImageData} tileData - Tile image data (512x512)
 * @param {Float32Array} pixelValues - Output pixel values array
 * @param {BigInt64Array} attentionMask - Output attention mask array
 * @param {number} tileIdx - Index of this tile
 * @param {number} validOffsetX - Start X of valid region (for padding detection)
 * @param {number} validOffsetY - Start Y of valid region
 * @param {number} validWidth - Width of valid region
 * @param {number} validHeight - Height of valid region
 */
function extractPatches(tileData, pixelValues, attentionMask, tileIdx, validOffsetX, validOffsetY, validWidth, validHeight) {
  const tileSize = CONFIG.tileSize;
  const patchSize = CONFIG.patchSize;
  const patchesPerSide = CONFIG.patchesPerTile;
  const patchesPerTile = patchesPerSide * patchesPerSide;
  const patchDim = patchSize * patchSize * 3;

  const pixels = tileData.data;
  const tileOffset = tileIdx * patchesPerTile * patchDim;
  const maskOffset = tileIdx * patchesPerTile;

  let patchIdx = 0;

  for (let py = 0; py < patchesPerSide; py++) {
    for (let px = 0; px < patchesPerSide; px++) {
      const patchStartX = px * patchSize;
      const patchStartY = py * patchSize;

      // Check if patch is in valid region
      const isValid = patchStartX < validOffsetX + validWidth &&
                      patchStartX + patchSize > validOffsetX &&
                      patchStartY < validOffsetY + validHeight &&
                      patchStartY + patchSize > validOffsetY;

      attentionMask[maskOffset + patchIdx] = isValid ? 1n : 0n;

      // Extract and normalize patch pixels
      const patchOffset = tileOffset + patchIdx * patchDim;
      let outIdx = 0;

      // Flatten patch: iterate over pixels in patch, then channels
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const imgX = patchStartX + dx;
          const imgY = patchStartY + dy;
          const srcIdx = (imgY * tileSize + imgX) * 4;

          for (let c = 0; c < 3; c++) {
            const pixelValue = pixels[srcIdx + c] * CONFIG.rescaleFactor;
            const normalized = (pixelValue - CONFIG.imageMean[c]) / CONFIG.imageStd[c];
            pixelValues[patchOffset + outIdx] = normalized;
            outIdx++;
          }
        }
      }

      patchIdx++;
    }
  }
}

/**
 * Load an image from URL or data URL
 * @param {string} src - Image URL or data URL
 * @returns {Promise<HTMLImageElement>}
 */
export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
