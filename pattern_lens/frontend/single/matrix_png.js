/* matrix_png.js
 *
 * Decode a square PNG created from a lower‑triangular, row‑stochastic matrix
 * (values scaled linearly to Matplotlib’s Blues colormap) and recover the
 * numeric matrix.
 *
 * Exported async function:
 *
 *     pngToMatrix(url: string, n?: number) -> Promise<number[][]>
 *
 *   • `url`  – URL (or data‑URI) of the PNG.
 *   • `n`    – optional matrix size; validated against the image if supplied.
 *
 * Assumptions
 *   • The PNG is n×n.
 *   • Pixel (0,0) encodes 1; pixel (1,0) encodes 0.
 *   • Mapping between these is linear in perceived luminance.
 *   • The matrix is strictly lower‑triangular; entries above the diagonal are 0.
 *
 * The function runs entirely in the browser, using OffscreenCanvas when
 * available.  It returns the matrix as an Array<Array<number>> with rows in
 * natural order (row 0 at index 0).
 */

function _pixel_avg(data, idx) {
	// Convert pixel at `idx` to perceived luminance (gray value).
	// Assumes data is Uint8ClampedArray with RGBA channels.
	return (
		(data[idx] + data[idx + 1] + data[idx + 2]) / 3
	) / 255; // scale to [0,1]
}

async function pngToMatrix(url, n = null) {
	// ---------- load & sanity‑check ------------------------------------------------
	console.log(`pngToMatrix: Loading image from ${url}`);
	const img = new Image();
	img.crossOrigin = 'anonymous';   // allow CORS / data URIs
	img.src = url;

	try {
		await img.decode();
		console.log(`pngToMatrix: Successfully decoded image`);
	} catch (error) {
		console.error(`pngToMatrix: Failed to decode image from ${url}:`, error);
		throw error;
	}

	const size = img.width;
	if (img.height !== size) throw new Error('PNG must be square');
	if (n !== null && n !== size) throw new Error('given n does not match PNG size');
	n = size;

	// ---------- raster to RGBA -----------------------------------------------------
	const canvas = typeof OffscreenCanvas !== 'undefined'
		? new OffscreenCanvas(size, size)
		: Object.assign(document.createElement('canvas'), { width: size, height: size });

	const ctx = canvas.getContext('2d');
	ctx.drawImage(img, 0, 0);
	const { data } = ctx.getImageData(0, 0, size, size); // Uint8ClampedArray

	// ---------- calibration --------------------------------------------------------
	const gMax = _pixel_avg(data, 0);          // pixel (0,0)  -> scalar 1
	const gMin = _pixel_avg(data, 4);          // pixel (1,0)  -> scalar 0
	const denom = gMax - gMin || 1;

	const pix_to_scalar = (data, idx) => (
		_pixel_avg(data, idx) - gMin
	) / denom; // scale to [0,1]

	// ---------- extract matrix -----------------------------------------------------
	const matrix = new Array(n);
	let rowStart = 0;               // byte offset of first pixel in current row

	try {
		for (let y = 0; y < n; ++y) {
			const row = new Float32Array(n); // zero‑filled
			for (let x = 0; x <= y; ++x) { // lower triangle incl. diag
				let v = pix_to_scalar(data, rowStart + x * 4);
				if (v < 0 || v > 1) {
					console.error(`Invalid pixel value at (${x},${y}): ${v} (will clam to [0,1])`);
					if (v < 0) {
						v = 0; // clamp to 0
					}
					else if (v > 1) {
						v = 1; // clamp to 1
					}
				}
				row[x] = v;
			}
			// normalize row to sum to 1
			const rowSum = row.reduce((sum, val) => sum + val, 0);
			for (let x = 0; x <= y; ++x) {
				row[x] /= rowSum; // normalize to sum to 1
			}
			matrix[y] = Array.from(row);
			rowStart += n * 4;
		}
	} catch (error) {
		console.error(`pngToMatrix: Error processing image data:`, error);
		console.log(`gMax: ${gMax}, gMin: ${gMin}, denom: ${denom}`);
		console.log(data)
		console.log(matrix);
		return matrix;
	}

	return matrix;
}
