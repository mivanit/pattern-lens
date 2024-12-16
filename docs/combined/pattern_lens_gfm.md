> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## Submodules

- [`activations`](#activations)
- [`attn_figure_funcs`](#attn_figure_funcs)
- [`consts`](#consts)
- [`figures`](#figures)
- [`indexes`](#indexes)
- [`load_activations`](#load_activations)
- [`prompts`](#prompts)
- [`server`](#server)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/__init__.py)

# `pattern_lens`

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`compute_activations`](#compute_activations)
- [`get_activations`](#get_activations)
- [`activations_main`](#activations_main)
- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/activations.py)

# `pattern_lens.activations`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/activations.py#L0-L336)

### `def compute_activations`

``` python
(
    prompt: dict,
    model: transformer_lens.HookedTransformer.HookedTransformer | None = None,
    save_path: pathlib.Path = WindowsPath('attn_data'),
    return_cache: bool = True,
    names_filter: Union[Callable[[str], bool], re.Pattern] = re.compile('blocks\\.(\\d+)\\.attn\\.hook_pattern')
) -> tuple[pathlib.Path, dict[str, jaxtyping.Float[ndarray, 'n_ctx n_ctx']] | None]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/activations.py#L34-L112)

get activations for a given model and prompt, possibly from a cache

if from a cache, prompt_meta must be passed and contain the prompt hash

### Parameters:

- `prompt : dict | None` (defaults to `None`)
- `model : HookedTransformer`
- `save_path : Path` (defaults to `Path(DATA_DIR)`)
- `return_cache : bool` will return `None` as the second element if
  `False` (defaults to `True`)
- `names_filter : Callable[[str], bool]|re.Pattern` a filter for the
  names of the activations to return. if an `re.Pattern`, will use
  `lambda key: names_filter.match(key) is not None` (defaults to
  `ATTN_PATTERN_REGEX`)

### Returns:

- `tuple[Path, AttentionCache|None]`

### `def get_activations`

``` python
(
    prompt: dict,
    model: transformer_lens.HookedTransformer.HookedTransformer | str,
    save_path: pathlib.Path = WindowsPath('attn_data'),
    allow_disk_cache: bool = True,
    return_cache: bool = True
) -> tuple[pathlib.Path, dict[str, jaxtyping.Float[ndarray, 'n_ctx n_ctx']]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/activations.py#L115-L148)

### `def activations_main`

``` python
(
    model_name: str,
    save_path: str,
    prompts_path: str,
    raw_prompts: bool,
    min_chars: int,
    max_chars: int,
    force: bool,
    n_samples: int,
    no_index_html: bool
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/activations.py#L151-L237)

### `def main`

``` python
()
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/activations.py#L240-L333)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`raw`](#raw)
- [`ATTENTION_MATRIX_FIGURE_FUNCS`](#ATTENTION_MATRIX_FIGURE_FUNCS)
- [`register_attn_figure_func`](#register_attn_figure_func)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/attn_figure_funcs.py)

# `pattern_lens.attn_figure_funcs`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/attn_figure_funcs.py#L0-L19)

### `def raw`

``` python
(
    attn_matrix: jaxtyping.Float[ndarray, 'n_ctx n_ctx'],
    ax: matplotlib.axes._axes.Axes
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/attn_figure_funcs.py#L6-L9)

- `ATTENTION_MATRIX_FIGURE_FUNCS: list[typing.Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], NoneType]] = [<function raw>]`

### `def register_attn_figure_func`

``` python
(
    func: Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], NoneType]
) -> Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], NoneType]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/attn_figure_funcs.py#L17-L20)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`AttentionMatrix`](#AttentionMatrix)
- [`AttentionMatrixFigureFunc`](#AttentionMatrixFigureFunc)
- [`AttentionCache`](#AttentionCache)
- [`FIGURE_FMT`](#FIGURE_FMT)
- [`DATA_DIR`](#DATA_DIR)
- [`ATTN_PATTERN_REGEX`](#ATTN_PATTERN_REGEX)
- [`SPINNER_KWARGS`](#SPINNER_KWARGS)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/consts.py)

# `pattern_lens.consts`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/consts.py#L0-L27)

- `AttentionMatrix = <class 'jaxtyping.Float[ndarray, 'n_ctx n_ctx']'>`

type alias for attention matrix

- `AttentionMatrixFigureFunc = typing.Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], NoneType]`

Type alias for a function that takes an attention matrix and an axes and
plots something

- `AttentionCache = dict[str, jaxtyping.Float[ndarray, 'n_ctx n_ctx']]`

type alias for a cache of attention matrices, subset of ActivationCache

- `FIGURE_FMT: str = 'svgz'`

format for saving figures

- `DATA_DIR: str = 'attn_data'`

default directory for attention data

- `ATTN_PATTERN_REGEX: re.Pattern = re.compile('blocks\\.(\\d+)\\.attn\\.hook_pattern')`

regex for finding attention patterns in model state dicts

- `SPINNER_KWARGS: dict = {'config': {'success': '✔️ '}}`

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`HTConfigMock`](#HTConfigMock)
- [`process_single_head`](#process_single_head)
- [`compute_and_save_figures`](#compute_and_save_figures)
- [`process_prompt`](#process_prompt)
- [`figures_main`](#figures_main)
- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py)

# `pattern_lens.figures`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L0-L218)

### `class HTConfigMock:`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L28-L40)

### `HTConfigMock`

``` python
(**kwargs)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L29-L33)

- `n_layers: int`

- `n_heads: int`

- `model_name: str`

### `def serialize`

``` python
(self)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L35-L36)

### `def load`

``` python
(cls, data: dict)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L38-L40)

### `def process_single_head`

``` python
(
    layer_idx: int,
    head_idx: int,
    attn_pattern: jaxtyping.Float[ndarray, 'n_ctx n_ctx'],
    save_dir: pathlib.Path,
    force_overwrite: bool = False
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L43-L67)

### `def compute_and_save_figures`

``` python
(
    model_cfg: 'HookedTransformerConfig|HTConfigMock',
    activations_path: pathlib.Path,
    cache: dict,
    save_path: pathlib.Path = WindowsPath('attn_data'),
    force_overwrite: bool = False
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L70-L96)

### `def process_prompt`

``` python
(
    prompt: dict,
    model_cfg: 'HookedTransformerConfig|HTConfigMock',
    save_path: pathlib.Path,
    force_overwrite: bool = False
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L99-L117)

### `def figures_main`

``` python
(
    model_name: str,
    save_path: str,
    n_samples: int,
    force: bool,
    parallel: bool | int = True
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L121-L165)

### `def main`

``` python
()
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/figures.py#L168-L215)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`generate_prompts_jsonl`](#generate_prompts_jsonl)
- [`generate_models_jsonl`](#generate_models_jsonl)
- [`generate_functions_jsonl`](#generate_functions_jsonl)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/indexes.py)

# `pattern_lens.indexes`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/indexes.py#L0-L60)

### `def generate_prompts_jsonl`

``` python
(model_dir: pathlib.Path)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/indexes.py#L7-L23)

creates a `prompts.jsonl` file with all the prompts in the model
directory

looks in all directories in `{model_dir}/prompts` for a `prompt.json`
file

### `def generate_models_jsonl`

``` python
(path: pathlib.Path)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/indexes.py#L26-L39)

creates a `models.jsonl` file with all the models

### `def generate_functions_jsonl`

``` python
(path: pathlib.Path)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/indexes.py#L42-L61)

unions all functions from file and current
`ATTENTION_MATRIX_FIGURE_FUNCS` into a `functions.jsonl` file

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`GetActivationsError`](#GetActivationsError)
- [`ActivationsMissingError`](#ActivationsMissingError)
- [`ActivationsMismatchError`](#ActivationsMismatchError)
- [`compare_prompt_to_loaded`](#compare_prompt_to_loaded)
- [`augment_prompt_with_hash`](#augment_prompt_with_hash)
- [`load_activations`](#load_activations)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py)

# `pattern_lens.load_activations`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py#L0-L63)

### `class GetActivationsError(builtins.ValueError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py#L12-L13)

Inappropriate argument value (of correct type).

### Inherited Members

- [`ValueError`](#GetActivationsError.__init__)

- [`with_traceback`](#GetActivationsError.with_traceback)

- [`add_note`](#GetActivationsError.add_note)

- [`args`](#GetActivationsError.args)

### `class ActivationsMissingError(GetActivationsError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py#L16-L17)

Inappropriate argument value (of correct type).

### Inherited Members

- [`ValueError`](#ActivationsMissingError.__init__)

- [`with_traceback`](#ActivationsMissingError.with_traceback)

- [`add_note`](#ActivationsMissingError.add_note)

- [`args`](#ActivationsMissingError.args)

### `class ActivationsMismatchError(GetActivationsError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py#L20-L21)

Inappropriate argument value (of correct type).

### Inherited Members

- [`ValueError`](#ActivationsMismatchError.__init__)

- [`with_traceback`](#ActivationsMismatchError.with_traceback)

- [`add_note`](#ActivationsMismatchError.add_note)

- [`args`](#ActivationsMismatchError.args)

### `def compare_prompt_to_loaded`

``` python
(prompt: dict, prompt_loaded: dict) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py#L24-L29)

### `def augment_prompt_with_hash`

``` python
(prompt: dict) -> dict
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py#L32-L41)

### `def load_activations`

``` python
(
    model_name: str,
    prompt: dict,
    save_path: pathlib.Path
) -> tuple[pathlib.Path, dict[str, jaxtyping.Float[ndarray, 'n_ctx n_ctx']]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/load_activations.py#L44-L64)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`load_text_data`](#load_text_data)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/prompts.py)

# `pattern_lens.prompts`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/prompts.py#L0-L62)

### `def load_text_data`

``` python
(
    fname: pathlib.Path,
    min_chars: int | None = None,
    max_chars: int | None = None,
    shuffle: bool = False
) -> list[dict]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/prompts.py#L6-L63)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.1.0

## API Documentation

- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/server.py)

# `pattern_lens.server`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/server.py#L0-L33)

### `def main`

``` python
(path: str, port: int = 8000)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.1.0/server.py#L7-L13)
