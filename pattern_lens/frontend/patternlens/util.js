const fileOps = {
	async getDirectoryContents(path) {
		const response = await fetch(`${path}/index.txt`);
		const text = await response.text();
		return text.trim().split('\n');
	},
	async fileExists(path) {
		const response = await fetch(path, { method: 'HEAD' });
		return response.ok;
	},
	async fetchJson(path) {
		const response = await fetch(path);
		return response.json();
	},
	async fetchJsonL(path) {
		const response = await fetch(path);
		const text = await response.text();
		// allow for the last line being incomplete
		const text_split = text.trim().split('\n');
		let output = text_split.slice(0, -1).map(JSON.parse);
		try {
			output.push(JSON.parse(text_split[text_split.length - 1]));
		} catch (error) {
			console.error('Error parsing last line of JSONL:', error);
		}
		return output;
	},
	async fetchAndDecompressSvgz(path) {
		// returns null if file does not exist
		const response = await fetch(path);
		if (!response.ok) {
			return null;
		} else {
			const arrayBuffer = await response.arrayBuffer();
			const uint8Array = new Uint8Array(arrayBuffer);
			return pako.inflate(uint8Array, { to: 'string' });
		}
	},
	async figureExists(path) {
		for (const format of CONFIG.data.figureFormats) {
			fig_path = `${path}.${format}`;
			if (await this.fileExists(fig_path)) {
				return format;
			}
		}
		return null;
	}
};


const colorUtils = {
	getRandomColor() {
		// Generate vibrant colors with good contrast
		const hue = Math.floor(Math.random() * 360);
		return `hsl(${hue}, 70%, 60%)`;
	},
};

