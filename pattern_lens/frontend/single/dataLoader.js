/**
 * Data Loader Module
 * Handles fetching attention patterns and prompt metadata
 */

class AttentionDataLoader {
    constructor(basePath = null) {
        this.basePath = basePath || CONFIG.data.basePath;
    }

    async loadAttentionPattern(model, promptHash, layerIdx, headIdx) {
        const pngPath = `${this.basePath}${model}/prompts/${promptHash}/L${layerIdx}/H${headIdx}/attn.png`;
        return await pngToMatrix(pngPath);
    }

    async loadPromptMetadata(model, promptHash) {
        const jsonPath = `${this.basePath}/${model}/prompts/${promptHash}/prompt.json`;
        const response = await fetch(jsonPath);
        if (!response.ok) {
            throw new Error(`Failed to load prompt metadata:\n${response.statusText}\nPath: ${jsonPath}`);
        }
        return await response.json();
    }
}