/**
 * Configuration Handler
 * 
 * Provides a flexible configuration system with multiple override levels:
 * 1. Default configuration (lowest priority)
 * 2. Inline configuration override (INLINE_CONFIG)
 * 3. External config.json file
 * 4. URL parameters (highest priority)
 * 
 * Features:
 * - Deep merging of configuration objects
 * - URL parameter parsing with dot notation support
 * - Automatic URL synchronization with debouncing
 * - Configuration export functionality
 * - Reset to loaded state capability
 */

// Configuration constants
const CONFIG_FILE_PATH = "sg_cfg.json";
const URL_UPDATE_DEBOUNCE_DELAY = 500; // ms
const FLOAT_COMPARISON_EPSILON = 0.001;

// Keys to skip during URL serialization
const URL_SKIP_PATHS = [];

// Keys to skip during config comparison
const COMPARISON_SKIP_KEYS = [];

// For inline config overrides - replace this with external script if needed
var INLINE_CONFIG = null;

// the line below might be replaced by an external build script to inject a config
/*$$$INLINE_CONFIG$$$*/

// Global variables for configuration management
let CONFIG = null;
let LOADED_CONFIG = null; // Store the config as loaded from file for comparison
let URL_UPDATE_TIMEOUT = null;

/**
 * Get default configuration object
 * @returns {object} Default configuration
 */
function getDefaultConfig() {
	let default_cfg = {
		// Layout configuration
		layout: {
			yLabelWidth: 100,
			xLabelHeight: 100,
			canvasSize: 500,
			maxTokensForLabels: 30  // Hide labels if more than this many tokens
		},

		// Data configuration
		data: {
			basePath: ".",
			attentionFilename: "raw.png",  // Filename for attention pattern PNG files
			tokenBoundary: {
				start: ["<BOS>"],  // Tokens to add at start
				end: []            // Tokens to add at end
			},
			// Default values for URL parameters when not specified
			defaults: {
				promptHash: "LQc1qlQHZHOVpI7zEWAeEA",
				head: "gpt2-small.L0.H0"
			},
			// Link templates for head and prompt
			links: {
				// Use {model}, {layer}, {head} placeholders for head link
				head: "https://miv.name/pattern-lens/demo/index.html?models={model}&heads-{model}=L{layer}H{head}",
				// Use {prompt_hash} placeholder for prompt link
				prompt: "https://miv.name/pattern-lens/demo/index.html?prompts={prompt_hash}"
			}
		},

		// Visualization configuration
		visualization: {
			// Canvas styling
			highlightStrokeStyle: "#ff0000",
			highlightLineWidth: 0.5,
			gridStrokeStyle: "#ddd",
			gridLineWidth: 0.2,

			// Colors for different axes
			colors: {
				kAxis: "#ff0000",      // Red for K (key) axis
				qAxis: "#00aa00",      // Green for Q (query) axis
				kAxisLight: "#ffcccc", // Light red for K axis labels
				qAxisLight: "#ccffcc"  // Light green for Q axis labels
			},

			// Performance settings
			throttleDelay: 16,  // ~60fps for mouse updates

			// Keyboard navigation
			keyboard: {
				moveStep: 1,
				ctrlMoveStep: 10,
				repeatDelay: 300,    // Initial delay before key repeat
				repeatInterval: 100  // Interval between repeats
			},

			// Token highlighting
			tokenHighlight: {
				maxOpacity: 0.9,
				intensityScale: 5.0,
				backgroundColor: "rgba(173, 216, 230, {alpha})"
			}
		}
	};

	if (INLINE_CONFIG) {
		// If INLINE_CONFIG is set, merge it into the default config
		deepMerge(default_cfg, INLINE_CONFIG);
		console.log("Merged inline config overrides");
	}

	return default_cfg;
}

/**
 * Load config.json (if present) and merge into CONFIG.
 * Also parse URL parameters and apply them to CONFIG.
 * Priority: URL params > config.json > inline config > defaults
 * @returns {Promise<object>} resolved CONFIG object
 */
async function getConfig() {
	// Initialize with defaults
	CONFIG = getDefaultConfig();

	try {
		// First, try to load config.json
		const r = await fetch(CONFIG_FILE_PATH);
		if (r.ok) {
			const loaded = await r.json();
			// Deep merge loaded config into CONFIG
			deepMerge(CONFIG, loaded);
			// Store a deep copy of the loaded config for URL comparison
			LOADED_CONFIG = JSON.parse(JSON.stringify(CONFIG));
			console.log("Loaded config.json");
		} else {
			console.warn("config.json not found, using defaults");
			// If no config.json, use defaults for comparison
			LOADED_CONFIG = JSON.parse(JSON.stringify(CONFIG));
		}
	} catch (e) {
		// if the inline config is null, then failing to find config.json is fine
		if (!INLINE_CONFIG) {
			console.error("Config load error:", e);
		} else {
			console.warn("Failed to load config.json, but it's fine because an inline config was provided");
		}
		// On error, use defaults for comparison
		LOADED_CONFIG = JSON.parse(JSON.stringify(CONFIG));
	}

	// Parse URL parameters and override CONFIG values (highest priority)
	parseURLParams();

	return CONFIG;
}

/**
 * Deep merge source object into target object
 * @param {object} target - Target object to merge into
 * @param {object} source - Source object to merge from
 */
function deepMerge(target, source) {
	for (const key in source) {
		if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
			if (!target[key]) target[key] = {};
			deepMerge(target[key], source[key]);
		} else {
			target[key] = source[key];
		}
	}
}

/**
 * Parse URL parameters and update CONFIG
 * Supports nested paths like: ?theme=light&ui.showToolbar=false&performance.maxItems=2000
 * Also supports arrays like: ?data.sources=file1.json,file2.json,file3.json
 * @param {URLSearchParams} [params] - Optional URLSearchParams object, defaults to current URL
 */
function parseURLParams(params = null) {
	if (!params) {
		params = new URLSearchParams(window.location.search);
	}

	for (const [key, value] of params) {
		setNestedConfigValue(CONFIG, key, parseConfigValue(value));
	}
}

/**
 * Set a nested configuration value using dot notation
 * Example: setNestedConfigValue(CONFIG, "ui.showToolbar", false)
 * @param {object} obj - Object to modify
 * @param {string} path - Dot-separated path
 * @param {any} value - Value to set
 */
function setNestedConfigValue(obj, path, value) {
	const keys = path.split('.');
	let current = obj;

	for (let i = 0; i < keys.length - 1; i++) {
		const key = keys[i];
		if (!(key in current) || typeof current[key] !== 'object') {
			current[key] = {};
		}
		current = current[key];
	}

	const finalKey = keys[keys.length - 1];
	current[finalKey] = value;
	console.log(`URL param override: ${path} = ${value}`);
}

/**
 * Parse a string value from URL params into appropriate type
 * Handles arrays (comma-separated values), booleans, numbers, and strings
 * @param {string} value - String value from URL parameter
 * @returns {any} Parsed value
 */
function parseConfigValue(value) {
	// Boolean
	if (value === 'true') return true;
	if (value === 'false') return false;

	// Array (comma-separated) - but handle single values too
	if (value.includes(',')) {
		return value.split(',').map(v => v.trim()).filter(v => v.length > 0);
	}

	// Number
	if (!isNaN(value) && !isNaN(parseFloat(value))) {
		return parseFloat(value);
	}

	// String (including hex colors, URLs, etc.)
	return value;
}

/**
 * Update the URL with current CONFIG state
 * Debounced to avoid excessive URL updates
 * @param {number} [delay] - Debounce delay in milliseconds (uses global constant if not provided)
 */
function updateURL(delay = URL_UPDATE_DEBOUNCE_DELAY) {
	if (URL_UPDATE_TIMEOUT) {
		clearTimeout(URL_UPDATE_TIMEOUT);
	}

	URL_UPDATE_TIMEOUT = setTimeout(() => {
		const params = generateURLParams();
		const newURL = window.location.pathname + (params.toString() ? '?' + params.toString() : '');
		window.history.replaceState({}, '', newURL);
		URL_UPDATE_TIMEOUT = null;
	}, delay);
}

/**
 * Generate URL search params from current CONFIG state
 * Only includes values that differ from the loaded config (not defaults)
 * @returns {URLSearchParams} URL parameters representing config differences
 */
function generateURLParams() {
	if (!LOADED_CONFIG) {
		// Fallback to default config if loaded config not available
		return new URLSearchParams();
	}

	const params = new URLSearchParams();
	const differences = findConfigDifferences(CONFIG, LOADED_CONFIG);

	for (const [path, value] of differences) {
		// Skip certain fields that shouldn't be in URLs
		if (shouldSkipInURL(path)) {
			continue;
		}

		// Special handling for arrays
		if (Array.isArray(value)) {
			if (value.length > 0) {
				params.set(path, value.join(','));
			}
		} else {
			params.set(path, value.toString());
		}
	}

	return params;
}

/**
 * Check if a config path should be skipped when generating URL parameters
 * @param {string} path - Config path (dot notation)
 * @returns {boolean} True if should be skipped
 */
function shouldSkipInURL(path) {
	return URL_SKIP_PATHS.some(skipPath => path.startsWith(skipPath));
}

/**
 * Find differences between current config and loaded config
 * Returns array of [path, value] tuples
 * Uses epsilon comparison for floats
 * @param {object} current - Current configuration
 * @param {object} base - Base configuration to compare against
 * @param {string} [prefix=''] - Current path prefix
 * @returns {Array<[string, any]>} Array of [path, value] differences
 */
function findConfigDifferences(current, base, prefix = '') {
	const differences = [];

	for (const key in current) {
		// Skip certain keys that shouldn't be compared
		if (shouldSkipInComparison(key)) {
			continue;
		}

		const currentPath = prefix ? `${prefix}.${key}` : key;
		const currentValue = current[key];
		const baseValue = base[key];

		if (Array.isArray(currentValue)) {
			// Special handling for arrays
			if (!Array.isArray(baseValue) || !arraysEqual(currentValue, baseValue)) {
				differences.push([currentPath, currentValue]);
			}
		} else if (typeof currentValue === 'object' && currentValue !== null) {
			if (typeof baseValue === 'object' && !Array.isArray(baseValue) && baseValue !== null) {
				differences.push(...findConfigDifferences(currentValue, baseValue, currentPath));
			} else {
				// Base doesn't have this object, include all of current
				differences.push([currentPath, JSON.stringify(currentValue)]);
			}
		} else {
			// Compare primitive values with epsilon for floats
			let valuesEqual = false;

			if (typeof currentValue === 'number' && typeof baseValue === 'number') {
				// Use epsilon comparison for floats
				valuesEqual = Math.abs(currentValue - baseValue) < FLOAT_COMPARISON_EPSILON;
			} else {
				// Direct comparison for other types
				valuesEqual = currentValue === baseValue;
			}

			if (!valuesEqual) {
				differences.push([currentPath, currentValue]);
			}
		}
	}

	return differences;
}

/**
 * Check if a config key should be skipped during comparison
 * @param {string} key - Configuration key
 * @returns {boolean} True if should be skipped
 */
function shouldSkipInComparison(key) {
	return COMPARISON_SKIP_KEYS.includes(key);
}

/**
 * Helper function to compare arrays for equality
 * @param {Array} arr1 - First array
 * @param {Array} arr2 - Second array
 * @returns {boolean} True if arrays are equal
 */
function arraysEqual(arr1, arr2) {
	if (arr1.length !== arr2.length) return false;
	for (let i = 0; i < arr1.length; i++) {
		if (arr1[i] !== arr2[i]) return false;
	}
	return true;
}

/**
 * Get the current configuration as a formatted JSON string
 * @param {number} [indent=2] - JSON indentation spaces
 * @returns {string} Formatted JSON configuration
 */
function getConfigAsJSON(indent = 2) {
	return JSON.stringify(CONFIG, null, indent);
}

/**
 * Export current configuration to a new browser tab
 * Creates a downloadable JSON file with current config
 */
function exportConfigToNewTab() {
	const configText = getConfigAsJSON();
	const blob = new Blob([configText], { type: 'application/json' });
	const url = URL.createObjectURL(blob);
	window.open(url, '_blank');

	// Clean up the object URL after a delay
	setTimeout(() => {
		URL.revokeObjectURL(url);
	}, 1000);
}

/**
 * Reset CONFIG to the loaded config.json state and clear URL parameters
 * Useful for reverting all changes back to the original loaded state
 */
function resetConfigToLoaded() {
	if (!LOADED_CONFIG) {
		console.warn("No loaded config available, resetting to defaults");
		CONFIG = getDefaultConfig();
	} else {
		// Deep copy the loaded config back to CONFIG
		CONFIG = JSON.parse(JSON.stringify(LOADED_CONFIG));
	}

	// Clear URL parameters by navigating to clean URL
	const cleanURL = window.location.pathname;
	window.history.replaceState({}, '', cleanURL);

	// Clear the URL update timeout if it exists
	if (URL_UPDATE_TIMEOUT) {
		clearTimeout(URL_UPDATE_TIMEOUT);
		URL_UPDATE_TIMEOUT = null;
	}

	console.log("Config reset to loaded state and URL cleared");
}

/**
 * Get a nested configuration value using dot notation
 * Example: getConfigValue("ui.showToolbar")
 * @param {string} path - Dot-separated path to config value
 * @param {any} [defaultValue] - Default value if path doesn't exist
 * @returns {any} Configuration value or default
 */
function getConfigValue(path, defaultValue = undefined) {
	const keys = path.split('.');
	let current = CONFIG;

	for (const key of keys) {
		if (current && typeof current === 'object' && key in current) {
			current = current[key];
		} else {
			return defaultValue;
		}
	}

	return current;
}

/**
 * Set a nested configuration value and optionally update URL
 * Example: setConfigValue("theme", "light", true)
 * @param {string} path - Dot-separated path to config value
 * @param {any} value - Value to set
 * @param {boolean} [updateUrl=true] - Whether to update URL parameters
 */
function setConfigValue(path, value, updateUrl = true) {
	setNestedConfigValue(CONFIG, path, value);

	if (updateUrl) {
		updateURL();
	}
}

/**
 * Initialize the configuration system
 * Call this once when your application starts
 * @returns {Promise<object>} Resolved configuration object
 */
async function initConfig() {
	try {
		return await getConfig();
	} catch (error) {
		console.error("Failed to initialize configuration:", error);
		// Fallback to defaults
		CONFIG = getDefaultConfig();
		LOADED_CONFIG = JSON.parse(JSON.stringify(CONFIG));
		return CONFIG;
	}
}