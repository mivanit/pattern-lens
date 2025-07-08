# Pattern Lens Frontend Refactoring Plan

## Overview
This document outlines the plan to streamline the pattern_lens frontend build process and configuration system with minimal changes to the existing code.

## Current State Analysis

### Build Process
- Currently, HTML bundling is done in `pattern_lens/indexes.py` using a custom `inline_assets()` function
- The `write_html_index()` function in indexes.py:
  - Reads `frontend/patternlens/index.html`
  - Inlines CSS and JS files (style.css, util.js, app.js)
  - Replaces version placeholder
  - Writes to `{data_path}/index.html`
- No build process exists for the single/ view - it's served as separate files

### Configuration System
- `single/` already has a robust config system in `config.js` with:
  - Default configuration
  - External config.json loading
  - URL parameter overrides
  - Inline config support
- `patternlens/` has the same config.js file (appears to be copied)
- Both use the same configuration approach but with different defaults

### Integration Status
- Currently no integration between patternlens and single views
- Each view operates independently

## Proposed Changes

### 1. Build Process Migration

#### Replace inline_assets() with muutils.web.bundle_html
- Remove the custom `inline_assets()` function from indexes.py
- Use muutils.web.bundle_html for bundling instead
- Benefits:
  - More robust (handles edge cases, proper regex/BS4 parsing)
  - Supports additional asset types (SVG, PNG)
  - Better error handling

#### Add Makefile Targets
Add these targets to the end of the makefile:

```makefile
# Frontend build targets
.PHONY: build-patternlens
build-patternlens:
	@echo "Building patternlens frontend"
	$(PYTHON) -m muutils.web.bundle_html pattern_lens/frontend/patternlens/index.html \
		--output pattern_lens/frontend/patternlens.html

.PHONY: build-single
build-single:
	@echo "Building single pattern viewer frontend"
	$(PYTHON) -m muutils.web.bundle_html pattern_lens/frontend/single/index.html \
		--output pattern_lens/frontend/single.html

.PHONY: build-frontend
build-frontend: build-patternlens build-single
	@echo "Built all frontend components"
```

#### Update indexes.py
- Modify `write_html_index()` to:
  - Check if pre-built `frontend/patternlens.html` exists
  - If yes, read it directly
  - If no, use `muutils.web.bundle_html`
  - Still handle version replacement



```python
# this function exists in `muutils.web.bundle_html`, we can import it from there
def inline_html_file(
    html_path: Path,
    output_path: Path,
    base_path: Path | None = None,
    config: InlineConfig | None = None,
    prettify: bool = False,
) -> Path:
    """Read *html_path*, inline its assets, and write the result.

    # Parameters
    - `html_path : Path`
        Source HTML file.
    - `output_path : Path`
        Destination path to write the modified HTML.
    - `base_path : Path | None`
        Directory used to resolve relative asset paths (defaults to the HTML file's directory).
        If `None`, uses the directory of *html_path*.
        (default: `None` -> use `html_path.parent`)
    - `config : InlineConfig | None`
        Inlining options.
        If `None`, uses default configuration.
        (default: `None` -> use `InlineConfig()`)
    - `prettify : bool`
        Pretty-print when `use_bs4=True`.
        (default: `False`)

    # Returns
    - `Path`
        Path actually written.
    """
    ...
```

### 2. Configuration System Enhancement

#### Minimal Changes to patternlens/app.js
Since both config.js files are identical, we only need to:

1. Update the default configuration in patternlens/config.js to include:
   ```javascript
   // In getDefaultConfig()
   data: {
       basePath: "./",  // Will be overridden by config.json
       singleViewerPath: "single.html"  // Path to single viewer
   }
   ```

2. Add hardcoded values that should be configurable:
   - Search for literal strings/numbers in app.js
   - Move them to config defaults
   - Access via `getConfigValue()` or `CONFIG.path.to.value`

### 3. Integration Between Views

#### Click Handler in patternlens
Add a method to handle pattern clicks in app.js:

```javascript
// In methods section
openSingleView(promptHash, model, layer, head) {
    const singlePath = getConfigValue('data.singleViewerPath', 'single.html');
    const params = new URLSearchParams({
        prompt: promptHash,
        head: `${model}.L${layer}.H${head}`
    });
    window.open(`${singlePath}?${params.toString()}`, '_blank');
}
```

#### Wire up the click handler
- Find where pattern images are displayed
- Add click handler that calls `openSingleView()` with appropriate parameters
- The single view already accepts these URL parameters

## Implementation Order

1. **Update Makefile** - Add the build targets
2. **Test muutils bundling** - Verify it works with both frontends
3. **Update indexes.py** - Modify to use pre-built files
4. **Config enhancement** - Move hardcoded values to config
5. **Add integration** - Implement click handlers

## Minimal Impact Design

### What We're NOT Changing
- Core functionality of either viewer
- The config.js system (already good)
- File structure
- Most of the existing code

### What We ARE Changing
- Build process (indexes.py and makefile)
- A few hardcoded values → config
- Adding click handlers for integration
- Output location of built files

## Testing Plan

1. Build both frontends with new makefile targets
2. Verify bundled HTML files work correctly
3. Test configuration loading and overrides
4. Test navigation from patternlens to single view
5. Ensure backwards compatibility

## Notes

- The inline config feature (`INLINE_CONFIG`) can be used for deployment-specific settings
- URL parameters will continue to work as before
- Config.json files can be placed alongside the HTML for easy customization
- The build process is optional - the old way still works