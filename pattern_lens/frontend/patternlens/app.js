const app = Vue.createApp({

	// ########     ###    ########    ###
	// ##     ##   ## ##      ##      ## ##
	// ##     ##  ##   ##     ##     ##   ##
	// ##     ## ##     ##    ##    ##     ##
	// ##     ## #########    ##    #########
	// ##     ## ##     ##    ##    ##     ##
	// ########  ##     ##    ##    ##     ##

	data() {
		return {
			isDarkMode: false,
			prompts: {
				all: {},        // hash -> prompt mapping
				selected: [],   // selected from table
				grid: {
					api: null,
					isReady: false
				},
			},
			loading: false,
			images: {
				visible: [],
				expected: 0,
				requested: false,
				upToDate: false,
				perRow: 4,
			},
			models: {
				configs: {},    // model -> config mapping
				grid: {
					api: null,
				},
			},
			filters: {
				available: {    // all available options
					models: [],
					functions: [],
					layers: [],
					heads: [],
				},
				selected: {     // currently selected options
					models: [],
					functions: [],
					layers: [],
					heads: [],
				},
			},
			head_selections_str: {}, // model -> selection string mapping
			visualization: {
				colorBy: '',
				sortBy: '',
				sortOrder: 'asc',
				colorMap: {},
			},
		};
	},

	methods: {

		// ##     ## ########    ###    ########   ######
		// ##     ## ##         ## ##   ##     ## ##    ##
		// ##     ## ##        ##   ##  ##     ## ##
		// ######### ######   ##     ## ##     ##  ######
		// ##     ## ##       ######### ##     ##       ##
		// ##     ## ##       ##     ## ##     ## ##    ##
		// ##     ## ######## ##     ## ########   ######

		// Parse head selection string and return a 2D array of booleans
		parseHeadString(str, maxLayer, maxHead) {
			try {
				const result = Array(maxLayer).fill().map(() => Array(maxHead).fill(false));
				if (!str || str.trim() === '') return result;

				const selections = str.replaceAll("x", "*").split(',').map(s => s.trim());

				for (const selection of selections) {
					const match = selection.match(/^L(\d+|\d+-\d+|\*)(H\d+|H\*|Hx)?$/);
					if (!match) return null;

					const layerPart = match[1];
					let headPart = match[2];

					// If the user typed only "L8" (no head specification), default to H*
					if (!headPart) {
						headPart = 'H*';
					}

					let layers = [];
					if (layerPart === '*') {
						layers = Array.from({ length: maxLayer }, (_, i) => i);
					} else if (layerPart.includes('-')) {
						const [start, end] = layerPart.split('-').map(Number);
						if (start > end || end >= maxLayer) return null;
						layers = Array.from({ length: end - start + 1 }, (_, i) => start + i);
					} else {
						const layer = Number(layerPart);
						if (layer >= maxLayer) return null;
						layers = [layer];
					}

					const headStr = headPart.substring(1);
					if (headStr === '*' || headStr === 'x') {
						for (const layer of layers) {
							result[layer].fill(true);
						}
					} else {
						const head = Number(headStr);
						if (head >= maxHead) return null;
						for (const layer of layers) {
							result[layer][head] = true;
						}
					}
				}

				return result;
			} catch (e) {
				console.error('Error parsing head string:', e);
				return null;
			}
		},

		isHeadSelected(model, layer, head) {
			// First check if we have parsed selections for this model
			if (!this.head_selections_arr[model]) {
				console.warn(`No parsed head selections found for model: ${model}`);
				return false;
			}

			try {
				// Verify layer and head are within bounds
				const parsedSelections = this.head_selections_arr[model];
				if (!Array.isArray(parsedSelections) ||
					!Array.isArray(parsedSelections[layer]) ||
					typeof parsedSelections[layer][head] === 'undefined') {
					console.warn(
						`Invalid layer/head combination for ${model}: L${layer}H${head}`,
						`Max bounds: L${parsedSelections.length - 1}H${parsedSelections[0]?.length - 1}`
					);
					return false;
				}

				return parsedSelections[layer][head];
			} catch (e) {
				console.error('Error checking head selection:', e);
				console.log('Model:', model, 'Layer:', layer, 'Head:', head);
				return false;
			}
		},

		isValidHeadSelection(model) {
			return this.head_selections_arr[model] !== null;
		},
		// ##     ## ########  ##
		// ##     ## ##     ## ##
		// ##     ## ##     ## ##
		// ##     ## ########  ##
		// ##     ## ##   ##   ##
		// ##     ## ##    ##  ##
		//  #######  ##     ## ########

		// Modified URL handling
		updateURL() {
			const params = new URLSearchParams();

			if (this.filters.selected.functions.length > 0) {
				params.set('functions', this.filters.selected.functions.join('~'));
			}

			if (this.prompts.selected.length > 0) {
				params.set('prompts', this.prompts.selected.join('~'));
			}

			if (this.filters.selected.models.length > 0) {
				params.set('models', this.filters.selected.models.join('~'));
			}

			if (this.filters.selected.models.length > 0) {
				for (const model of Object.keys(this.head_selections_str)) {
					params.set(
						`${CONFIG.data.urlHeadPrefix}${model}`,
						this.head_selections_str[model].replaceAll("*", "x").replaceAll(" ", "").split(',').join('~')
					);
				}
			}

			const newURL = `${window.location.pathname}?${params.toString()}`;
			history.replaceState(null, '', newURL);
		},

		readURL() {
			const params = new URLSearchParams(window.location.search);

			this.filters.selected.functions = params.get('functions')?.split('~') || [];

			this.prompts.selected = params.get('prompts')?.split('~') || [];

			this.filters.selected.models = params.get('models')?.split('~') || [];

			try {
				this.head_selections_str = {};
				for (const [key, value] of params) {
					if (key.startsWith(CONFIG.data.urlHeadPrefix)) {
						const model = key.substring(CONFIG.data.urlHeadPrefix.length);
						this.head_selections_str[model] = value.split('~').join(', ');
					}
				}
			} catch (e) {
				console.error('Error parsing head selections from URL:', e);
			}
		},
		selectPromptsFromURL() {
			if (!this.isGridReady || this.prompts.selected.length === 0) return;

			const promptSet = new Set(this.prompts.selected);
			this.prompts.grid.api.forEachNode((node) => {
				if (promptSet.has(node.data.hash)) {
					node.setSelected(true);
				}
			});
		},
		getImageUrl(image) {
			return this.getFilterUrl('all', [image.model], [image.promptHash], [image.layer], [image.head], [image.function]);
		},

		openSingleView(promptHash, model, layer, head) {
			const singlePath = CONFIG.data.singleViewerPath;
			const params = new URLSearchParams({
				prompt: promptHash,
				head: `${model}.L${layer}.H${head}`
			});
			window.open(`${singlePath}?${params.toString()}`, '_blank');
		},

		getSinglePropertyFilterUrl(type, value) {
			const params = new URLSearchParams(window.location.search);
			params.set(type, value); // This preserves other params while updating just this one
			return `${window.location.pathname}?${params.toString()}`;
		},

		getFilterUrl(type, ...values) {
			const params = new URLSearchParams(window.location.search);

			if (type === 'all') {
				params.set('models', values[0].join('~'));
				params.set('prompts', values[1].join('~'));
				params.set('layers', values[2].join('~'));
				params.set('heads', values[3].join('~'));
				params.set('functions', values[4].join('~'));
			} else {
				params.set(type, values.flat().join('~'));
			}

			return `${window.location.pathname}?${params.toString()}`;
		},

		// ##     ## ######## ##       ########  ######## ########
		// ##     ## ##       ##       ##     ## ##       ##     ##
		// ##     ## ##       ##       ##     ## ##       ##     ##
		// ######### ######   ##       ########  ######   ########
		// ##     ## ##       ##       ##        ##       ##   ##
		// ##     ## ##       ##       ##        ##       ##    ##
		// ##     ## ######## ######## ##        ######## ##     ##

		toggleDarkMode() {
			console.log('Toggling dark mode');  // Add this debug line
			this.isDarkMode = !this.isDarkMode;
			localStorage.setItem('darkMode', this.isDarkMode);
			// Force a DOM update
			this.$nextTick(() => {
				document.documentElement.classList.toggle('dark-mode', this.isDarkMode);
			});
		},
		clearAllSelections() {
			// Clear prompts selection
			if (this.prompts.grid.api) {
				this.prompts.grid.api.deselectAll();
			}

			// Clear models selection
			if (this.models.grid.api) {
				this.models.grid.api.deselectAll();
			}

			// Clear function selections
			this.filters.selected.functions = [];

			// Reset head selections
			this.head_selections_str = {};

			// Update URL to reflect cleared state
			this.updateURL();
		},
		isIndeterminate(category) {
			const items = this.filters.available[category];
			const selectedItems = this.filters.selected[category];
			return selectedItems.length > 0 && selectedItems.length < items.length;
		},
		isChecked(category) {
			const items = this.filters.available[category];
			const selectedItems = this.filters.selected[category];
			return selectedItems.length === items.length && items.length > 0;
		},
		toggleSelectAll(category, event) {
			const checked = event.target.checked;
			this.filters.selected[category] = checked ? [...this.filters.available[category]] : [];
		},
		async loadData() {
			try {
				await this.loadModels();
				await Promise.all([
					this.loadAllPrompts(),
					this.loadFunctions()
				]);

				this.updateLayersAndHeads();
			} catch (error) {
				console.error('Error loading data:', error);
			}
		},
		async loadModels() {
			this.loading = true;
			console.log('Loading models...');
			const models = await fileOps.fetchJsonL(`${CONFIG.data.basePath}/${CONFIG.data.modelsFile}`);
			this.models.configs = {};
			for (const model of models) {
				this.models.configs[model["model_name"]] = model;
			}
			this.filters.available.models = Object.keys(this.models.configs);
			console.log('Models:', this.filters.available.models);
			this.loading = false;

			// After loading models, initialize head selections
			this.filters.selected.models.forEach(model => {
				if (!this.head_selections_str[model]) {
					this.head_selections_str[model] = 'L*H*';
				}
			});
		},
		async loadFunctions() {
			const functions = await fileOps.fetchJsonL(`${CONFIG.data.basePath}/${CONFIG.data.figuresFile}`);
			console.log('Functions:', functions);
			this.filters.available.functions = functions.reduce(
				(acc, item) => {
					acc[item.name] = item;
					return acc;
				},
				{},
			);
			console.log('this.filters.available.functions:', this.filters.available.functions);
		},
		onFirstDataRendered(params) {
			this.selectPromptsFromURL();
		},
		// Handle selection change in ag-Grid
		onSelectionChanged() {
			const selectedNodes = this.prompts.grid.api.getSelectedRows();
			this.prompts.selected = selectedNodes.map(node => node.hash);
			this.updateURL();
		},
		// Update layers and heads based on selected models
		updateLayersAndHeads() {
			// get all layer and head counts
			let mdl_n_layers = [];
			let mdl_n_heads = [];
			for (const model of this.filters.selected.models) {
				const config = this.models.configs[model];
				if (config) {
					mdl_n_layers.push(config.n_layers);
					mdl_n_heads.push(config.n_heads);
				}
			}
			// get the max layer and head counts, generate lists
			this.filters.available.layers = [];
			this.filters.available.heads = [];

			for (let i = 0; i < _.max(mdl_n_layers); i++) {
				this.filters.available.layers.push(i.toString());
			}
			for (let i = 0; i < _.max(mdl_n_heads); i++) {
				this.filters.available.heads.push(i.toString());
			}
		},

		// ##     ##  #######  ########  ######## ##        ######
		// ###   ### ##     ## ##     ## ##       ##       ##    ##
		// #### #### ##     ## ##     ## ##       ##       ##
		// ## ### ## ##     ## ##     ## ######   ##        ######
		// ##     ## ##     ## ##     ## ##       ##             ##
		// ##     ## ##     ## ##     ## ##       ##       ##    ##
		// ##     ##  #######  ########  ######## ########  ######
		getHeadSelectionCount(model) {
			const parsed = this.head_selections_arr[model];
			if (!parsed) return 0;
			return parsed.reduce((acc, layer) =>
				acc + layer.reduce((sum, isSelected) => sum + (isSelected ? 1 : 0), 0), 0);
		},
		getTotalHeads(model) {
			const config = this.models.configs[model];
			return config ? config.n_layers * config.n_heads : 0;
		},
		setupModelTable() {
			const columnDefs = [
				{
					headerName: 'Model',
					field: 'model_name',
					sort: 'asc',
					width: 150
				},
				{
					headerName: 'd_model',
					field: 'd_model',
					width: 90,
					filter: 'agNumberColumnFilter'
				},
				{
					headerName: 'n_layers',
					field: 'n_layers',
					width: 90,
					filter: 'agNumberColumnFilter'
				},
				{
					headerName: 'n_heads',
					field: 'n_heads',
					width: 90,
					filter: 'agNumberColumnFilter'
				},
				{
					headerName: 'Selected',
					valueGetter: (params) => {
						return `${this.getHeadSelectionCount(params.data.model_name)} / ${this.getTotalHeads(params.data.model_name)}`;
					},
					width: 100
				},
				{
					headerName: 'Head Grid',
					field: 'head_grid',
					width: 150,
					cellRenderer: (params) => {
						const model = params.data.model_name;
						const div = document.createElement('div');
						div.className = 'head-grid';
						div.setAttribute('data-model', model); // Add data attribute for updates

						const n_heads = params.data.n_heads;
						const n_layers = params.data.n_layers;

						for (let h = 0; h < n_heads; h++) {
							const layerDiv = document.createElement('div');
							layerDiv.className = 'headsGrid-col';

							for (let l = 0; l < n_layers; l++) {
								const cell = document.createElement('div');
								cell.className = `headsGrid-cell ${this.isHeadSelected(model, l, h) ? 'headsGrid-cell-selected' : 'headsGrid-cell-empty'}`;
								cell.setAttribute('data-layer', l);
								cell.setAttribute('data-head', h);
								layerDiv.appendChild(cell);
							}

							div.appendChild(layerDiv);
						}

						return div;
					}
				},
				{
					headerName: 'Head Selection',
					field: 'head_selection',
					editable: true,
					width: 200,
					cellEditor: 'agTextCellEditor',
					cellEditorParams: {
						maxLength: 50
					},
					valueSetter: params => {
						const newValue = params.newValue;
						const model = params.data.model_name;

						// Update the head selection in Vue's data
						params.context.componentParent.head_selections_str[model] = newValue;

						// Update the cell class for validation styling
						const isValid = params.context.componentParent.isValidHeadSelection(model);
						const cell = params.api.getCellRendererInstances({
							rowNodes: [params.node],
							columns: [params.column]
						})[0];

						if (cell) {
							const element = cell.getGui();
							if (isValid) {
								element.classList.remove('invalid-selection');
							} else {
								element.classList.add('invalid-selection');
							}
						}

						// Force refresh of the head grid cell
						const gridCol = params.api.getColumnDef('head_grid');
						if (gridCol) {
							params.api.refreshCells({
								rowNodes: [params.node],
								columns: ['head_grid'],
								force: true
							});
						}

						return true;
					},
					valueGetter: params => {
						return params.context.componentParent.head_selections_str[params.data.model_name] || 'L*H*';
					},
					cellClass: params => {
						const isValid = params.context.componentParent.isValidHeadSelection(params.data.model_name);
						return isValid ? '' : 'invalid-selection';
					}
				},
			];

			const modelGrid_options = {
				columnDefs: columnDefs,
				rowData: Object.values(this.models.configs),
				selection: {
					headerCheckbox: true,
					selectAll: 'filtered',
					checkboxes: true,
					mode: 'multiRow',
					enableClickSelection: true,
				},
				defaultColDef: {
					sortable: true,
					filter: true,
					resizable: true,
					floatingFilter: true,
					suppressKeyboardEvent: params => {
						// Allow all keyboard events in edit mode
						if (params.editing) {
							return false;
						}
						// Prevent default grid behavior for typing when not in edit mode
						if (params.event.key.length === 1 && !params.event.ctrlKey && !params.event.metaKey) {
							return false;
						}
						return true;
					},
				},
				context: {
					componentParent: this
				},
				onSelectionChanged: (event) => {
					const selectedRows = event.api.getSelectedRows();
					this.filters.selected.models = selectedRows.map(row => row.model_name);
				},
				onGridReady: (params) => {
					this.models.grid.api = params.api;
					// Select models from URL
					if (this.filters.selected.models.length > 0) {
						params.api.forEachNode(node => {
							if (this.filters.selected.models.includes(node.data.model_name)) {
								node.setSelected(true);
							}
						});
					}
				},
			};

			const modelGrid_div = document.querySelector('#modelGrid');
			this.models.grid.api = agGrid.createGrid(modelGrid_div, modelGrid_options);
		},
		refreshHeadGrids() {
			if (this.models.grid.api) {
				this.models.grid.api.refreshCells({
					columns: ['head_grid'],
					force: true
				});
			}
		},
		// ########  ########   #######  ##     ## ########  ########
		// ##     ## ##     ## ##     ## ###   ### ##     ##    ##   
		// ##     ## ##     ## ##     ## #### #### ##     ##    ##   
		// ########  ########  ##     ## ## ### ## ########     ##   
		// ##        ##   ##   ##     ## ##     ## ##           ##    
		// ##        ##    ##  ##     ## ##     ## ##           ##    
		// ##        ##     ##  #######  ##     ## ##           ##    

		async loadAllPrompts() {
			this.loading = true;
			console.log('Loading prompts...');
			this.prompts.all = {};

			for (const model of this.filters.available.models) {
				try {
					const modelPrompts = await fileOps.fetchJsonL(`${CONFIG.data.basePath}/${model}/${CONFIG.data.promptsFile}`);
					for (const prompt of modelPrompts) {
						if (prompt.hash in this.prompts.all) {
							this.prompts.all[prompt.hash].models.push(model);
						} else {
							this.prompts.all[prompt.hash] = { ...prompt, models: [model] };
						}
					}
				} catch (error) {
					console.error(`Error loading prompts for model ${model}:`, error);
				}
			}
			console.log('loaded number of prompts:', Object.keys(this.prompts.all).length);
			this.loading = false;
		},
		// Initialize the ag-Grid table
		setupPromptTable() {
			const columnDefs = [
				{
					headerName: 'Prompt Text',
					field: 'text',
					sortable: true,
					filter: true,
					flex: 2,
					cellRenderer: (params) => {
						const eGui = document.createElement('div');
						// Replace tabs and newlines with spaces for display
						eGui.innerText = params.value.replace(/\s+/g, ' ');
						eGui.classList.add('prompt-text-cell');
						eGui.addEventListener('click', () => {
							navigator.clipboard.writeText(params.value);
						});

						eGui.addEventListener('contextmenu', (event) => {
							event.preventDefault();
							const newWindow = window.open();
							newWindow.document.write(`<pre>${params.value}</pre>`);
							newWindow.document.close();
							newWindow.document.title = `Prompt '${params.data.hash}'`;
						});

						return eGui;
					},
				},
				{
					headerName: 'Models', field: 'models', sortable: true, filter: true, width: 150,
					valueFormatter: (params) => params.value.join(', '),
				},
				{ headerName: 'Hash', field: 'hash', sortable: true, filter: true, width: 100 },
				{ headerName: 'Tokens', field: 'n_tokens', sortable: true, filter: 'agNumberColumnFilter', width: 80 },
				{ headerName: 'Dataset', field: 'meta.pile_set_name', sortable: true, filter: true, width: 150 },
			];

			// Grid options
			const promptGrid_options = {
				columnDefs: columnDefs,
				rowData: Object.values(this.prompts.all),
				pagination: true,
				enableCellTextSelection: true,
				paginationPageSize: 20,
				paginationPageSizeSelector: [5, 10, 20, 50, 100, 500],
				selection: {
					headerCheckbox: true,
					selectAll: 'filtered',
					checkboxes: true,
					mode: 'multiRow',
					enableClickSelection: true,
				},

				defaultColDef: {
					sortable: true,
					filter: true,
					resizable: true,
					floatingFilter: true
				},
				onSelectionChanged: this.onSelectionChanged.bind(this),
				onFirstDataRendered: this.onFirstDataRendered.bind(this),
				onGridReady: (params) => {
					this.prompts.grid.api = params.api;
					this.isGridReady = true;
					this.selectPromptsFromURL();
				},
			};

			const promptGrid_div = document.querySelector('#promptGrid');
			this.prompts.grid.api = agGrid.createGrid(promptGrid_div, promptGrid_options);
		},

		// ########  ####  ######  ########  ##          ###    ##    ##
		// ##     ##  ##  ##    ## ##     ## ##         ## ##    ##  ##
		// ##     ##  ##  ##       ##     ## ##        ##   ##    ####
		// ##     ##  ##   ######  ########  ##       ##     ##    ##
		// ##     ##  ##        ## ##        ##       #########    ##
		// ##     ##  ##  ##    ## ##        ##       ##     ##    ##
		// ########  ####  ######  ##        ######## ##     ##    ##

		// Display images based on selected criteria
		async displayImages() {
			this.loading = true;
			this.images.requested = true;
			this.images.visible = [];

			// Calculate total images based on parsed head selections
			let totalImages = 0;
			for (const model of this.filters.selected.models) {
				totalImages += this.getHeadSelectionCount(model) * this.prompts.selected.length * this.filters.selected.functions.length;
			}
			this.images.expected = totalImages;

			// Load images based on parsed head selections
			for (const model of this.filters.selected.models) {
				const config = this.models.configs[model];
				const rawString = this.head_selections_str[model] || 'L*H*';
				const parsedHeads = this.parseHeadString(rawString, config.n_layers, config.n_heads);
				if (!parsedHeads) {
					console.warn(`Invalid head selection for ${model}: "${rawString}"`);
					continue;
				}

				// Iterate over all layers and heads
				for (let layer = 0; layer < config.n_layers; layer++) {
					for (let head = 0; head < config.n_heads; head++) {
						if (!parsedHeads[layer][head]) {
							continue;
						}
						// Now for each selected prompt and function:
						for (const promptHash of this.prompts.selected) {
							for (
								const func_name of
								this.filters.selected.functions
							) {
								let func = this.filters.available.functions[func_name];
								if (!func) {
									console.warn(`Function not found ${func_name}`, typeof func_name, JSON.stringify(func_name), func_name, this.filters.available.functions);
								}
								const basePath = `${CONFIG.data.basePath}/${model}/prompts/${promptHash}/L${layer}/H${head}`;

								// get the figure format from metadata
								let figure_format = func.figure_save_fmt;
								if (!figure_format) {
									// as a fallback, look for all valid formats
									figure_format = await fileOps.figureExists(`${basePath}/${func_name}`);
									console.log('could not find figure format for func name', func_name, 'found', figure_format);
								}

								if (figure_format) {
									// Create figure entry
									const figure_meta = {
										name: `${model} - Prompt ${promptHash} - L${layer}H${head} - ${func_name}`,
										model: model,
										promptHash: promptHash,
										layer: layer,
										head: head,
										function: func_name,
										figure_format: figure_format,
									};

									if (figure_format === 'svgz') {
										const svgText = await fileOps.fetchAndDecompressSvgz(`${basePath}/${func_name}.svgz`);
										if (svgText) {
											this.images.visible.push({
												content: svgText,
												...figure_meta,
											});
										}
									} else {
										const imglink = `<img src="${basePath}/${func_name}.${figure_format}" alt="${figure_meta.name}">`;
										this.images.visible.push({
											content: imglink,
											...figure_meta,
										});
									}
								}
							}
						}
					}
				}
			}

			this.images.upToDate = true;
			this.loading = false;
		},
		openMetadata(func) {
			const newWindow = window.open('', '_blank');
			let content = `<div style="font-family: sans-serif; line-height:1.4;">`;
			if (func.doc) {
				content += `<p><strong>Description:</strong> ${func.doc}</p>`;
			}
			if (func.figure_save_fmt) {
				content += `<p><strong>Format:</strong> ${func.figure_save_fmt}</p>`;
			}
			if (func.source) {
				content += `<p><strong>Source:</strong> ${func.source}</p>`;
			}
			content += `</div>`;
			newWindow.document.write(content);
			newWindow.document.close();
			newWindow.document.title = `Metadata for ${func.name}`;
		},

		regenerateColors() {
			if (!this.visualization.colorBy) return;

			// Get unique values for the selected property
			const uniqueValues = [...new Set(this.images.visible.map(img => img[this.visualization.colorBy]))];

			// Generate new random colors
			this.visualization.colorMap = {};
			uniqueValues.forEach(value => {
				this.visualization.colorMap[value] = colorUtils.getRandomColor();
			});
		},


		getBorderColor(image) {
			if (!this.visualization.colorBy || !image) return 'transparent';
			const value = image[this.visualization.colorBy];
			return this.visualization.colorMap[value] || 'transparent';
		},
	},



	//  ######   #######  ##     ## ########  ##     ## ######## ######## ########
	// ##    ## ##     ## ###   ### ##     ## ##     ##    ##    ##       ##     ##
	// ##       ##     ## #### #### ##     ## ##     ##    ##    ##       ##     ##
	// ##       ##     ## ## ### ## ########  ##     ##    ##    ######   ##     ##
	// ##       ##     ## ##     ## ##        ##     ##    ##    ##       ##     ##
	// ##    ## ##     ## ##     ## ##        ##     ##    ##    ##       ##     ##
	//  ######   #######  ##     ## ##         #######     ##    ######## ########

	computed: {
		uniqueDatasets() {
			return [
				...new Set(
					Object.values(this.prompts.all).map(prompt => prompt.meta.pile_set_name).filter(Boolean)
				)
			];
		},
		head_selections_arr() {
			// model -> boolean[][] mapping for efficient lookup
			let parsed = {};

			for (const model in this.head_selections_str) {
				const config = this.models.configs[model];
				if (!config) {
					console.warn(`No config found for model: ${model}`);
					parsed[model] = null;
					continue;
				}

				const parsedHeads = this.parseHeadString(
					this.head_selections_str[model] || 'L*H*',
					config.n_layers,
					config.n_heads
				);

				if (!parsedHeads) {
					console.warn(
						`Invalid head selection for ${model}: "${this.head_selections_str[model]}"`
					);
				}

				parsed[model] = parsedHeads;
			}

			return parsed;
		},
		sortedImages() {
			if (!this.visualization.sortBy) return this.images.visible;

			return [...this.images.visible].sort((a, b) => {
				const valueA = a[this.visualization.sortBy];
				const valueB = b[this.visualization.sortBy];

				// Handle numeric values for layer and head
				if (['layer', 'head'].includes(this.visualization.sortBy)) {
					const numA = Number(valueA);
					const numB = Number(valueB);
					return this.visualization.sortOrder === 'asc'
						? numA - numB
						: numB - numA;
				}

				// Handle string values
				const comparison = String(valueA).localeCompare(String(valueB));
				return this.visualization.sortOrder === 'asc' ? comparison : -comparison;
			});
		},
	},


	// ##      ##    ###    ########  ######  ##     ##
	// ##  ##  ##   ## ##      ##    ##    ## ##     ##
	// ##  ##  ##  ##   ##     ##    ##       ##     ##
	// ##  ##  ## ##     ##    ##    ##       #########
	// ##  ##  ## #########    ##    ##       ##     ##
	// ##  ##  ## ##     ##    ##    ##    ## ##     ##
	//  ###  ###  ##     ##    ##     ######  ##     ##

	// Watch for changes in selected models to load prompts and update layers and heads
	watch: {
		'filters.selected': {
			deep: true,
			handler() {
				this.images.upToDate = false;
				this.updateURL();
			}
		},
		'prompts.selected': {
			handler() {
				this.images.upToDate = false;
			}
		},
		'head_selections_str': {
			deep: true,
			handler(newValue) {
				Object.keys(newValue).forEach(model => {
					if (!this.models.configs[model]) {
						console.warn(`Attempting to update head selections for unknown model: ${model}`);
						return;
					}
				});
				this.images.upToDate = false;
				this.updateURL();
				this.refreshHeadGrids();
			}
		},
		'filters.selected.models': {
			deep: true,
			handler(newModels) {
				// Initialize head selections for new models
				newModels.forEach(model => {
					if (!this.head_selections_str[model]) {
						this.head_selections_str[model] = 'L*H*';
					}
				});
				this.updateURL();
			}
		},
		'visualization.colorBy': {
			handler(newValue) {
				if (newValue) {
					this.regenerateColors();
				}
			}
		},
	},

	// Lifecycle hook when component is mounted
	async mounted() {
		console.log('Mounting app:', this);

		// Apply config values to data (config is already initialized by main script)
		this.images.perRow = CONFIG.ui.imagesPerRow;
		this.isDarkMode = CONFIG.ui.darkModeDefault;

		const savedDarkMode = localStorage.getItem('darkMode');
		if (savedDarkMode !== null) {
			this.isDarkMode = savedDarkMode === 'true';
		}
		if (this.isDarkMode) {
			document.documentElement.classList.add('dark-mode');
		}
		this.readURL(); // Read filters from URL first
		await this.loadData(); // Load models, prompts, and functions
		this.setupModelTable(); // Initialize the model grid
		this.setupPromptTable(); // Initialize the prompts grid
		console.log('Mounted app:', this);
	}
});
