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
}