/**
 * Data Loader Module
 * Handles fetching attention patterns and prompt metadata
 */

class AttentionDataLoader {
    constructor() { }

    async loadAttentionPattern(model, promptHash, layerIdx, headIdx) {
        const pngPath = `${CONFIG.data.basePath}/${model}/prompts/${promptHash}/L${layerIdx}/H${headIdx}/${CONFIG.data.attentionFilename}`;
        console.log(`Loading attention pattern from: ${pngPath}`);
        const matrix = await pngToMatrix(pngPath);
        return { matrix, pngPath };
    }

    async loadPromptMetadata(model, promptHash) {
        const jsonPath = `${CONFIG.data.basePath}/${model}/prompts/${promptHash}/prompt.json`;
        console.log(`Loading prompt metadata from: ${jsonPath}`);
        const response = await fetch(jsonPath);
        if (!response.ok) {
            throw new Error(`Failed to load prompt metadata:\n${response.statusText}\nPath: ${jsonPath}`);
        }
        return await response.json();
    }

    /**
     * Load the model-level config (model_cfg.json) for a given model.
     * This contains model metadata such as `default_prepend_bos`, which
     * indicates whether the model's tokenizer prepends a BOS token.
     * Returns null if the config is missing or fails to load, so callers
     * should fall back to safe defaults.
     */
    async loadModelConfig(model) {
        const jsonPath = `${CONFIG.data.basePath}/${model}/model_cfg.json`;
        try {
            const response = await fetch(jsonPath);
            if (response.ok) return await response.json();
        } catch (e) {
            console.warn(`Failed to load model config: ${e}`);
        }
        return null;
    }
}