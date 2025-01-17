> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

[![PyPI](https://img.shields.io/pypi/v/pattern-lens)](https://pypi.org/project/pattern-lens/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pattern-lens)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://miv.name/pattern-lens)
[![Checks](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml/badge.svg)](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml)

[![Coverage](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4NCjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iOTkiIGhlaWdodD0iMjAiPg0KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iYiIgeDI9IjAiIHkyPSIxMDAlIj4NCiAgICAgICAgPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjYmJiIiBzdG9wLW9wYWNpdHk9Ii4xIi8+DQogICAgICAgIDxzdG9wIG9mZnNldD0iMSIgc3RvcC1vcGFjaXR5PSIuMSIvPg0KICAgIDwvbGluZWFyR3JhZGllbnQ+DQogICAgPG1hc2sgaWQ9ImEiPg0KICAgICAgICA8cmVjdCB3aWR0aD0iOTkiIGhlaWdodD0iMjAiIHJ4PSIzIiBmaWxsPSIjZmZmIi8+DQogICAgPC9tYXNrPg0KICAgIDxnIG1hc2s9InVybCgjYSkiPg0KICAgICAgICA8cGF0aCBmaWxsPSIjNTU1IiBkPSJNMCAwaDYzdjIwSDB6Ii8+DQogICAgICAgIDxwYXRoIGZpbGw9IiNhNGE2MWQiIGQ9Ik02MyAwaDM2djIwSDYzeiIvPg0KICAgICAgICA8cGF0aCBmaWxsPSJ1cmwoI2IpIiBkPSJNMCAwaDk5djIwSDB6Ii8+DQogICAgPC9nPg0KICAgIDxnIGZpbGw9IiNmZmYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJEZWphVnUgU2FucyxWZXJkYW5hLEdlbmV2YSxzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIj4NCiAgICAgICAgPHRleHQgeD0iMzEuNSIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPmNvdmVyYWdlPC90ZXh0Pg0KICAgICAgICA8dGV4dCB4PSIzMS41IiB5PSIxNCI+Y292ZXJhZ2U8L3RleHQ+DQogICAgICAgIDx0ZXh0IHg9IjgwIiB5PSIxNSIgZmlsbD0iIzAxMDEwMSIgZmlsbC1vcGFjaXR5PSIuMyI+ODIlPC90ZXh0Pg0KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTQiPjgyJTwvdGV4dD4NCiAgICA8L2c+DQo8L3N2Zz4NCg==)](docs/coverage/html/)
![GitHub
commits](https://img.shields.io/github/commit-activity/t/mivanit/pattern-lens)
![GitHub commit
activity](https://img.shields.io/github/commit-activity/m/mivanit/pattern-lens)
![GitHub closed pull
requests](https://img.shields.io/github/issues-pr-closed/mivanit/pattern-lens)
![code size,
bytes](https://img.shields.io/github/languages/code-size/mivanit/pattern-lens)

# pattern-lens

visualization of LLM attention patterns and things computed about them

`pattern-lens` makes it easy to:

- Generate visualizations of attention patterns, or figures computed
  from attention patterns, from models supported by
  [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Compare generated figures across models, layers, and heads in an
  [interactive web interface](https://miv.name/pattern-lens/demo/)

# Installation

``` bash
pip install pattern-lens
```

# Usage

The pipeline is as follows:

- Generate attention patterns using
  `pattern_lens.activations.acitvations_main()`, saving them in `npz`
  files
- Generate visualizations using `pattern_lens.figures.figures_main()` –
  read the `npz` files, pass each attention pattern to each
  visualization function, and save the resulting figures
- Serve the web interface using `pattern_lens.server` – web interface
  reads metadata in json/jsonl files, then lets the user select figures
  to show

## Basic CLI

Generate attention patterns and default visualizations:

``` bash
# generate activations
python -m pattern_lens.activations --model gpt2 --prompts data/pile_1k.jsonl --save-path attn_data
# create visualizations
python -m pattern_lens.figures --model gpt2 --save-path attn_data
```

serve the web UI:

``` bash
python -m pattern_lens.server --path attn_data
```

## Web UI

View a demo of the web UI at
[miv.name/pattern-lens/demo](https://miv.name/pattern-lens/demo/).

## Custom Figures

Add custom visualization functions by decorating them with
`@register_attn_figure_func`. You should still generate the activations
first:

    python -m pattern_lens.activations --model gpt2 --prompts data/pile_1k.jsonl --save-path attn_data

and then write+run a script/notebook that looks something like this:

``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

# these functions simplify writing a function which saves a figure
from pattern_lens.figure_util import matplotlib_figure_saver, save_matrix_wrapper
# decorator to register your function, such that it will be run by `figures_main`
from pattern_lens.attn_figure_funcs import register_attn_figure_func
# runs the actual figure generation pipeline
from pattern_lens.figures import figures_main

# define your own functions
# this one uses `matplotlib_figure_saver` -- define a function that takes matrix and `plt.Axes`, modify the axes
@register_attn_figure_func
@matplotlib_figure_saver(fmt="svgz")
def svd_spectra(attn_matrix: np.ndarray, ax: plt.Axes) -> None:
    # Perform SVD
    U, s, Vh = svd(attn_matrix)

    # Plot singular values
    ax.plot(s, "o-")
    ax.set_yscale("log")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("Singular Value Spectrum of Attention Matrix")


# run the figures pipelne
# run the pipeline
figures_main(
    model_name="pythia-14m",
    save_path=Path("docs/demo/"),
    n_samples=5,
    force=False,
)
```

see `demo.ipynb` for a full example

## Submodules

- [`activations`](#activations)
- [`attn_figure_funcs`](#attn_figure_funcs)
- [`consts`](#consts)
- [`figure_util`](#figure_util)
- [`figures`](#figures)
- [`indexes`](#indexes)
- [`load_activations`](#load_activations)
- [`prompts`](#prompts)
- [`server`](#server)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/__init__.py)

# `pattern_lens`

[![PyPI](https://img.shields.io/pypi/v/pattern-lens)](https://pypi.org/project/pattern-lens/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pattern-lens)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://miv.name/pattern-lens)
[![Checks](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml/badge.svg)](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml)

[![Coverage](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4NCjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iOTkiIGhlaWdodD0iMjAiPg0KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0iYiIgeDI9IjAiIHkyPSIxMDAlIj4NCiAgICAgICAgPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjYmJiIiBzdG9wLW9wYWNpdHk9Ii4xIi8+DQogICAgICAgIDxzdG9wIG9mZnNldD0iMSIgc3RvcC1vcGFjaXR5PSIuMSIvPg0KICAgIDwvbGluZWFyR3JhZGllbnQ+DQogICAgPG1hc2sgaWQ9ImEiPg0KICAgICAgICA8cmVjdCB3aWR0aD0iOTkiIGhlaWdodD0iMjAiIHJ4PSIzIiBmaWxsPSIjZmZmIi8+DQogICAgPC9tYXNrPg0KICAgIDxnIG1hc2s9InVybCgjYSkiPg0KICAgICAgICA8cGF0aCBmaWxsPSIjNTU1IiBkPSJNMCAwaDYzdjIwSDB6Ii8+DQogICAgICAgIDxwYXRoIGZpbGw9IiNhNGE2MWQiIGQ9Ik02MyAwaDM2djIwSDYzeiIvPg0KICAgICAgICA8cGF0aCBmaWxsPSJ1cmwoI2IpIiBkPSJNMCAwaDk5djIwSDB6Ii8+DQogICAgPC9nPg0KICAgIDxnIGZpbGw9IiNmZmYiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGZvbnQtZmFtaWx5PSJEZWphVnUgU2FucyxWZXJkYW5hLEdlbmV2YSxzYW5zLXNlcmlmIiBmb250LXNpemU9IjExIj4NCiAgICAgICAgPHRleHQgeD0iMzEuNSIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPmNvdmVyYWdlPC90ZXh0Pg0KICAgICAgICA8dGV4dCB4PSIzMS41IiB5PSIxNCI+Y292ZXJhZ2U8L3RleHQ+DQogICAgICAgIDx0ZXh0IHg9IjgwIiB5PSIxNSIgZmlsbD0iIzAxMDEwMSIgZmlsbC1vcGFjaXR5PSIuMyI+ODIlPC90ZXh0Pg0KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTQiPjgyJTwvdGV4dD4NCiAgICA8L2c+DQo8L3N2Zz4NCg==)](docs/coverage/html/)
![GitHub
commits](https://img.shields.io/github/commit-activity/t/mivanit/pattern-lens)
![GitHub commit
activity](https://img.shields.io/github/commit-activity/m/mivanit/pattern-lens)
![GitHub closed pull
requests](https://img.shields.io/github/issues-pr-closed/mivanit/pattern-lens)
![code size,
bytes](https://img.shields.io/github/languages/code-size/mivanit/pattern-lens)

### pattern-lens

visualization of LLM attention patterns and things computed about them

`pattern-lens` makes it easy to:

- Generate visualizations of attention patterns, or figures computed
  from attention patterns, from models supported by
  [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Compare generated figures across models, layers, and heads in an
  [interactive web interface](https://miv.name/pattern-lens/demo/)

### Installation

``` bash
pip install pattern-lens
```

### Usage

The pipeline is as follows:

- Generate attention patterns using
  `pattern_lens.activations.acitvations_main()`, saving them in `npz`
  files
- Generate visualizations using
  `<a href="pattern_lens/figures.html#figures_main">pattern_lens.figures.figures_main()</a>`
  – read the `npz` files, pass each attention pattern to each
  visualization function, and save the resulting figures
- Serve the web interface using
  `<a href="pattern_lens/server.html">pattern_lens.server</a>` – web
  interface reads metadata in json/jsonl files, then lets the user
  select figures to show

#### Basic CLI

Generate attention patterns and default visualizations:

``` bash
### generate activations
python -m <a href="pattern_lens/activations.html">pattern_lens.activations</a> --model gpt2 --prompts data/pile_1k.jsonl --save-path attn_data
### create visualizations
python -m <a href="pattern_lens/figures.html">pattern_lens.figures</a> --model gpt2 --save-path attn_data
```

serve the web UI:

``` bash
python -m <a href="pattern_lens/server.html">pattern_lens.server</a> --path attn_data
```

#### Web UI

View a demo of the web UI at
[miv.name/pattern-lens/demo](https://miv.name/pattern-lens/demo/).

#### Custom Figures

Add custom visualization functions by decorating them with
`@register_attn_figure_func`. You should still generate the activations
first:

    python -m <a href="pattern_lens/activations.html">pattern_lens.activations</a> --model gpt2 --prompts data/pile_1k.jsonl --save-path attn_data

and then write+run a script/notebook that looks something like this:

``` python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd

### these functions simplify writing a function which saves a figure
from <a href="pattern_lens/figure_util.html">pattern_lens.figure_util</a> import matplotlib_figure_saver, save_matrix_wrapper
### decorator to register your function, such that it will be run by `figures_main`
from <a href="pattern_lens/attn_figure_funcs.html">pattern_lens.attn_figure_funcs</a> import register_attn_figure_func
### runs the actual figure generation pipeline
from <a href="pattern_lens/figures.html">pattern_lens.figures</a> import figures_main

### define your own functions
### this one uses `matplotlib_figure_saver` -- define a function that takes matrix and `plt.Axes`, modify the axes
@register_attn_figure_func
@matplotlib_figure_saver(fmt="svgz")
def svd_spectra(attn_matrix: np.ndarray, ax: plt.Axes) -> None:
    # Perform SVD
    U, s, Vh = svd(attn_matrix)

    # Plot singular values
    ax.plot(s, "o-")
    ax.set_yscale("log")
    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value")
    ax.set_title("Singular Value Spectrum of Attention Matrix")


### run the figures pipelne
### run the pipeline
figures_main(
    model_name="pythia-14m",
    save_path=Path("docs/demo/"),
    n_samples=5,
    force=False,
)
```

see `demo.ipynb` for a full example

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/__init__.py#L0-L2)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

computing and saving activations given a model and prompts

# Usage:

from the command line:

``` bash
python -m pattern_lens.activations --model <model_name> --prompts <prompts_path> --save-path <save_path> --min-chars <min_chars> --max-chars <max_chars> --n-samples <n_samples>
```

from a script:

``` python
from pattern_lens.activations import activations_main
activations_main(
    model_name="gpt2",
    save_path="demo/"
    prompts_path="data/pile_1k.jsonl",
)
```

## API Documentation

- [`compute_activations`](#compute_activations)
- [`get_activations`](#get_activations)
- [`activations_main`](#activations_main)
- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/activations.py)

# `pattern_lens.activations`

computing and saving activations given a model and prompts

### Usage:

from the command line:

``` bash
python -m <a href="">pattern_lens.activations</a> --model <model_name> --prompts <prompts_path> --save-path <save_path> --min-chars <min_chars> --max-chars <max_chars> --n-samples <n_samples>
```

from a script:

``` python
from <a href="">pattern_lens.activations</a> import activations_main
activations_main(
    model_name="gpt2",
    save_path="demo/"
    prompts_path="data/pile_1k.jsonl",
)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/activations.py#L0-L482)

### `def compute_activations`

``` python
(
    prompt: dict,
    model: transformer_lens.HookedTransformer.HookedTransformer | None = None,
    save_path: pathlib.Path = WindowsPath('attn_data'),
    return_cache: bool = True,
    names_filter: Union[Callable[[str], bool], re.Pattern] = re.compile('blocks\\.(\\d+)\\.attn\\.hook_pattern')
) -> tuple[pathlib.Path, dict[str, numpy.ndarray] | None]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/activations.py#L62-L143)

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

- `tuple[Path, ActivationCacheNp|None]`

### `def get_activations`

``` python
(
    prompt: dict,
    model: transformer_lens.HookedTransformer.HookedTransformer | str,
    save_path: pathlib.Path = WindowsPath('attn_data'),
    allow_disk_cache: bool = True,
    return_cache: bool = True
) -> tuple[pathlib.Path, dict[str, numpy.ndarray] | None]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/activations.py#L162-L223)

given a prompt and a model, save or load activations

### Parameters:

- `prompt : dict` expected to contain the ‘text’ key
- `model : HookedTransformer | str` either a `HookedTransformer` or a
  string model name, to be loaded with
  `HookedTransformer.from_pretrained`
- `save_path : Path` path to save the activations to (and load from)
  (defaults to `Path(DATA_DIR)`)
- `allow_disk_cache : bool` whether to allow loading from disk cache
  (defaults to `True`)
- `return_cache : bool` whether to return the cache. if `False`, will
  return `None` as the second element (defaults to `True`)

### Returns:

- `tuple[Path, ActivationCacheNp | None]` the path to the activations
  and the cache if `return_cache` is `True`

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
    no_index_html: bool,
    shuffle: bool = False,
    device: str | torch.device = 'cuda'
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/activations.py#L226-L350)

main function for computing activations

### Parameters:

- `model_name : str` name of a model to load with
  `HookedTransformer.from_pretrained`
- `save_path : str` path to save the activations to
- `prompts_path : str` path to the prompts file
- `raw_prompts : bool` whether the prompts are raw, not filtered by
  length. `load_text_data` will be called if `True`, otherwise just load
  the “text” field from each line in `prompts_path`
- `min_chars : int` minimum number of characters for a prompt
- `max_chars : int` maximum number of characters for a prompt
- `force : bool` whether to overwrite existing files
- `n_samples : int` maximum number of samples to process
- `no_index_html : bool` whether to write an index.html file
- `shuffle : bool` whether to shuffle the prompts (defaults to `False`)
- `device : str | torch.device` the device to use. if a string, will be
  passed to `torch.device`

### `def main`

``` python
()
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/activations.py#L353-L479)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

default figure functions

- If you are making a PR, add your new figure function here.
- if you are using this as a library, then you can see examples here

note that for `pattern_lens.figures` to recognize your function, you
need to use the `register_attn_figure_func` decorator which adds your
function to `ATTENTION_MATRIX_FIGURE_FUNCS`

## API Documentation

- [`ATTENTION_MATRIX_FIGURE_FUNCS`](#ATTENTION_MATRIX_FIGURE_FUNCS)
- [`register_attn_figure_func`](#register_attn_figure_func)
- [`raw`](#raw)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/attn_figure_funcs.py)

# `pattern_lens.attn_figure_funcs`

default figure functions

- If you are making a PR, add your new figure function here.
- if you are using this as a library, then you can see examples here

note that for `<a href="figures.html">pattern_lens.figures</a>` to
recognize your function, you need to use the `register_attn_figure_func`
decorator which adds your function to `ATTENTION_MATRIX_FIGURE_FUNCS`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/attn_figure_funcs.py#L0-L98)

- `ATTENTION_MATRIX_FIGURE_FUNCS: list[typing.Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType]] = [<function raw>]`

### `def register_attn_figure_func`

``` python
(
    func: Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType]
) -> Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/attn_figure_funcs.py#L24-L56)

decorator for registering attention matrix figure function

if you want to add a new figure function, you should use this decorator

    # Parameters:
     - `func : AttentionMatrixFigureFunc`
       your function, which should take an attention matrix and path

    # Returns:
     - `AttentionMatrixFigureFunc`
       your function, after we add it to `ATTENTION_MATRIX_FIGURE_FUNCS`

### Usage:

``` python
@register_attn_figure_func
def my_new_figure_func(attn_matrix: AttentionMatrix, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(attn_matrix, cmap="viridis")
    ax.set_title("My New Figure Function")
    ax.axis("off")
    plt.savefig(path / "my_new_figure_func", format="svgz")
    plt.close(fig)
```

### `def raw`

``` python
(
    attn_matrix: jaxtyping.Float[ndarray, 'n_ctx n_ctx']
) -> jaxtyping.Float[ndarray, 'n m']
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/attn_figure_funcs.py#L75-L79)

raw attention matrix

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

implements some constants and types

## API Documentation

- [`AttentionMatrix`](#AttentionMatrix)
- [`ActivationCacheNp`](#ActivationCacheNp)
- [`DATA_DIR`](#DATA_DIR)
- [`ATTN_PATTERN_REGEX`](#ATTN_PATTERN_REGEX)
- [`SPINNER_KWARGS`](#SPINNER_KWARGS)
- [`DIVIDER_S1`](#DIVIDER_S1)
- [`DIVIDER_S2`](#DIVIDER_S2)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/consts.py)

# `pattern_lens.consts`

implements some constants and types

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/consts.py#L0-L28)

- `AttentionMatrix = <class 'jaxtyping.Float[ndarray, 'n_ctx n_ctx']'>`

type alias for attention matrix

- `ActivationCacheNp = dict[str, numpy.ndarray]`

type alias for a cache of attention matrices, subset of ActivationCache

- `DATA_DIR: str = 'attn_data'`

default directory for attention data

- `ATTN_PATTERN_REGEX: re.Pattern = re.compile('blocks\\.(\\d+)\\.attn\\.hook_pattern')`

regex for finding attention patterns in model state dicts

- `SPINNER_KWARGS: dict = {'config': {'success': '✔️ '}}`

default kwargs for `muutils.spinner.Spinner`

- `DIVIDER_S1: str = '======================================================================'`

divider string for separating sections

- `DIVIDER_S2: str = '--------------------------------------------------'`

divider string for separating subsections

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

implements a bunch of types, default values, and templates which are
useful for figure functions

notably, you can use the decorators `matplotlib_figure_saver`,
`save_matrix_wrapper` to make your functions save figures

## API Documentation

- [`AttentionMatrixFigureFunc`](#AttentionMatrixFigureFunc)
- [`Matrix2D`](#Matrix2D)
- [`Matrix2Drgb`](#Matrix2Drgb)
- [`AttentionMatrixToMatrixFunc`](#AttentionMatrixToMatrixFunc)
- [`MATPLOTLIB_FIGURE_FMT`](#MATPLOTLIB_FIGURE_FMT)
- [`MatrixSaveFormat`](#MatrixSaveFormat)
- [`MATRIX_SAVE_NORMALIZE`](#MATRIX_SAVE_NORMALIZE)
- [`MATRIX_SAVE_CMAP`](#MATRIX_SAVE_CMAP)
- [`MATRIX_SAVE_FMT`](#MATRIX_SAVE_FMT)
- [`MATRIX_SAVE_SVG_TEMPLATE`](#MATRIX_SAVE_SVG_TEMPLATE)
- [`matplotlib_figure_saver`](#matplotlib_figure_saver)
- [`matrix_to_image_preprocess`](#matrix_to_image_preprocess)
- [`matrix2drgb_to_png_bytes`](#matrix2drgb_to_png_bytes)
- [`matrix_as_svg`](#matrix_as_svg)
- [`save_matrix_wrapper`](#save_matrix_wrapper)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figure_util.py)

# `pattern_lens.figure_util`

implements a bunch of types, default values, and templates which are
useful for figure functions

notably, you can use the decorators `matplotlib_figure_saver`,
`save_matrix_wrapper` to make your functions save figures

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figure_util.py#L0-L447)

- `AttentionMatrixFigureFunc = typing.Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType]`

Type alias for a function that, given an attention matrix, saves a
figure

- `Matrix2D = <class 'jaxtyping.Float[ndarray, 'n m']'>`

Type alias for a 2D matrix (plottable)

- `Matrix2Drgb = <class 'jaxtyping.UInt8[ndarray, 'n m rgb=3']'>`

Type alias for a 2D matrix with 3 channels (RGB)

- `AttentionMatrixToMatrixFunc = typing.Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx']], jaxtyping.Float[ndarray, 'n m']]`

Type alias for a function that, given an attention matrix, returns a 2D
matrix

- `MATPLOTLIB_FIGURE_FMT: str = 'svgz'`

format for saving matplotlib figures

- `MatrixSaveFormat = typing.Literal['png', 'svg', 'svgz']`

Type alias for the format to save a matrix as when saving raw matrix,
not matplotlib figure

- `MATRIX_SAVE_NORMALIZE: bool = False`

default for whether to normalize the matrix to range \[0, 1\]

- `MATRIX_SAVE_CMAP: str = 'viridis'`

default colormap for saving matrices

- `MATRIX_SAVE_FMT: Literal['png', 'svg', 'svgz'] = 'svgz'`

default format for saving matrices

- `MATRIX_SAVE_SVG_TEMPLATE: str = '<svg xmlns="http://www.w3.org/2000/svg" width="{m}" height="{n}" viewBox="0 0 {m} {n}" image-rendering="pixelated"> <image href="data:image/png;base64,{png_base64}" width="{m}" height="{n}" /> </svg>'`

template for saving an `n` by `m` matrix as an svg/svgz

### `def matplotlib_figure_saver`

``` python
(
    func: Optional[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], NoneType]] = None,
    *args,
    fmt: str = 'svgz'
) -> Union[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType], Callable[[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], NoneType], str], Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType]]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figure_util.py#L67-L127)

decorator for functions which take an attention matrix and predefined
`ax` object, making it save a figure

### Parameters:

- `func : Callable[[AttentionMatrix, plt.Axes], None]` your function,
  which should take an attention matrix and predefined `ax` object
- `fmt : str` format for saving matplotlib figures (defaults to
  `MATPLOTLIB_FIGURE_FMT`)

### Returns:

- `AttentionMatrixFigureFunc` your function, after we wrap it to save a
  figure

### Usage:

``` python
@register_attn_figure_func
@matplotlib_figure_saver
def raw(attn_matrix: AttentionMatrix, ax: plt.Axes) -> None:
    ax.matshow(attn_matrix, cmap="viridis")
    ax.set_title("Raw Attention Pattern")
    ax.axis("off")
```

### `def matrix_to_image_preprocess`

``` python
(
    matrix: jaxtyping.Float[ndarray, 'n m'],
    normalize: bool = False,
    cmap: str | matplotlib.colors.Colormap = 'viridis',
    diverging_colormap: bool = False,
    normalize_min: float | None = None
) -> jaxtyping.UInt8[ndarray, 'n m rgb=3']
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figure_util.py#L130-L226)

preprocess a 2D matrix into a plottable heatmap image

### Parameters:

- `matrix : Matrix2D` input matrix
- `normalize : bool` whether to normalize the matrix to range \[0, 1\]
  (defaults to `MATRIX_SAVE_NORMALIZE`)
- `cmap : str|Colormap` the colormap to use for the matrix (defaults to
  `MATRIX_SAVE_CMAP`)
- `diverging_colormap : bool` if True and using a diverging colormap,
  ensures 0 values map to the center of the colormap (defaults to False)
- `normalize_min : float|None` if a float, then for `normalize=True` and
  `diverging_colormap=False`, the minimum value to normalize to
  (generally set this to zero?). if `None`, then the minimum value of
  the matrix is used. if `diverging_colormap=True` OR `normalize=False`,
  this **must** be `None`. (defaults to `None`)

### Returns:

- `Matrix2Drgb`

### `def matrix2drgb_to_png_bytes`

``` python
(
    matrix: jaxtyping.UInt8[ndarray, 'n m rgb=3'],
    buffer: _io.BytesIO | None = None
) -> bytes | None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figure_util.py#L233-L259)

Convert a `Matrix2Drgb` to valid PNG bytes via PIL

- if `buffer` is provided, it will write the PNG bytes to the buffer and
  return `None`
- if `buffer` is not provided, it will return the PNG bytes

### Parameters:

- `matrix : Matrix2Drgb`
- `buffer : io.BytesIO | None` (defaults to `None`, in which case it
  will return the PNG bytes)

### Returns:

- `bytes|None` `bytes` if `buffer` is `None`, otherwise `None`

### `def matrix_as_svg`

``` python
(
    matrix: jaxtyping.Float[ndarray, 'n m'],
    normalize: bool = False,
    cmap: str | matplotlib.colors.Colormap = 'viridis',
    diverging_colormap: bool = False,
    normalize_min: float | None = None
) -> str
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figure_util.py#L262-L315)

quickly convert a 2D matrix to an SVG image, without matplotlib

### Parameters:

- `matrix : Float[np.ndarray, 'n m']` a 2D matrix to convert to an SVG
  image
- `normalize : bool` whether to normalize the matrix to range \[0, 1\].
  if it’s not in the range \[0, 1\], this must be `True` or it will
  raise an `AssertionError` (defaults to `False`)
- `cmap : str` the colormap to use for the matrix – will look up in
  `matplotlib.colormaps` if it’s a string (defaults to `"viridis"`)
- `diverging_colormap : bool` if True and using a diverging colormap,
  ensures 0 values map to the center of the colormap (defaults to False)
- `normalize_min : float|None` if a float, then for `normalize=True` and
  `diverging_colormap=False`, the minimum value to normalize to
  (generally set this to zero?) if `None`, then the minimum value of the
  matrix is used if `diverging_colormap=True` OR `normalize=False`, this
  **must** be `None` (defaults to `None`)

### Returns:

- `str` the SVG content for the matrix

### `def save_matrix_wrapper`

``` python
(
    func: Optional[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx']], jaxtyping.Float[ndarray, 'n m']]] = None,
    *args,
    fmt: Literal['png', 'svg', 'svgz'] = 'svgz',
    normalize: bool = False,
    cmap: str | matplotlib.colors.Colormap = 'viridis',
    diverging_colormap: bool = False,
    normalize_min: float | None = None
) -> Union[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType], Callable[[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx']], jaxtyping.Float[ndarray, 'n m']]], Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], NoneType]]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figure_util.py#L338-L448)

Decorator for functions that process an attention matrix and save it as
an SVGZ image. Can handle both argumentless usage and with arguments.

### Parameters:

- `func : AttentionMatrixToMatrixFunc|None` Either the function to
  decorate (in the no-arguments case) or `None` when used with
  arguments.
- `fmt : MatrixSaveFormat, keyword-only` The format to save the matrix
  as. Defaults to `MATRIX_SAVE_FMT`.
- `normalize : bool, keyword-only` Whether to normalize the matrix to
  range \[0, 1\]. Defaults to `False`.
- `cmap : str, keyword-only` The colormap to use for the matrix.
  Defaults to `MATRIX_SVG_CMAP`.
- `diverging_colormap : bool` if True and using a diverging colormap,
  ensures 0 values map to the center of the colormap (defaults to False)
- `normalize_min : float|None` if a float, then for `normalize=True` and
  `diverging_colormap=False`, the minimum value to normalize to
  (generally set this to zero?) if `None`, then the minimum value of the
  matrix is used if `diverging_colormap=True` OR `normalize=False`, this
  **must** be `None` (defaults to `None`)

### Returns:

`AttentionMatrixFigureFunc|Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]`

- `AttentionMatrixFigureFunc` if `func` is `AttentionMatrixToMatrixFunc`
  (no arguments case)
- `Callable[[AttentionMatrixToMatrixFunc], AttentionMatrixFigureFunc]`
  if `func` is `None` – returns the decorator which will then be applied
  to the (with arguments case)

### Usage:

``` python
@save_matrix_wrapper
def identity_matrix(matrix):
    return matrix

@save_matrix_wrapper(normalize=True, fmt="png")
def scale_matrix(matrix):
    return matrix * 2

@save_matrix_wrapper(normalize=True, cmap="plasma")
def scale_matrix(matrix):
    return matrix * 2
```

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

code for generating figures from attention patterns, using the functions
decorated with `register_attn_figure_func`

## API Documentation

- [`HTConfigMock`](#HTConfigMock)
- [`process_single_head`](#process_single_head)
- [`compute_and_save_figures`](#compute_and_save_figures)
- [`process_prompt`](#process_prompt)
- [`figures_main`](#figures_main)
- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py)

# `pattern_lens.figures`

code for generating figures from attention patterns, using the functions
decorated with `register_attn_figure_func`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L0-L329)

### `class HTConfigMock:`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L32-L54)

Mock of `transformer_lens.HookedTransformerConfig` for type hinting and
loading config json

can be initialized with any kwargs, and will update its `__dict__` with
them. does, however, require the following attributes: -
`n_layers: int` - `n_heads: int` - `model_name: str`

### `HTConfigMock`

``` python
(**kwargs)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L41-L45)

- `n_layers: int`

- `n_heads: int`

- `model_name: str`

### `def serialize`

``` python
(self)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L47-L49)

serialize the config to json. values which aren’t serializable will be
converted via `muutils.json_serialize.json_serialize`

### `def load`

``` python
(cls, data: dict)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L51-L54)

try to load a config from a dict, using the `__init__` method

### `def process_single_head`

``` python
(
    layer_idx: int,
    head_idx: int,
    attn_pattern: jaxtyping.Float[ndarray, 'n_ctx n_ctx'],
    save_dir: pathlib.Path,
    force_overwrite: bool = False
) -> dict[str, bool | Exception]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L57-L103)

process a single head’s attention pattern, running all the functions in
`ATTENTION_MATRIX_FIGURE_FUNCS` on the attention pattern

### Parameters:

- `layer_idx : int`
- `head_idx : int`
- `attn_pattern : AttentionMatrix` attention pattern for the head
- `save_dir : Path` directory to save the figures to
- `force_overwrite : bool` whether to overwrite existing figures. if
  `False`, will skip any functions which have already saved a figure
  (defaults to `False`)

### Returns:

- `dict[str, bool | Exception]` a dictionary of the status of each
  function, with the function name as the key and the status as the
  value

### `def compute_and_save_figures`

``` python
(
    model_cfg: 'HookedTransformerConfig|HTConfigMock',
    activations_path: pathlib.Path,
    cache: dict[str, numpy.ndarray],
    save_path: pathlib.Path = WindowsPath('attn_data'),
    force_overwrite: bool = False,
    track_results: bool = False
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L106-L162)

compute and save figures for all heads in the model, using the functions
in `ATTENTION_MATRIX_FIGURE_FUNCS`

### Parameters:

- `model_cfg : HookedTransformerConfig|HTConfigMock`
- `cache : ActivationCacheNp`
- `save_path : Path` (defaults to `Path(DATA_DIR)`)
- `force_overwrite : bool` force overwrite of existing figures. if
  `False`, will skip any functions which have already saved a figure
  (defaults to `False`)
- `track_results : bool` whether to track the results of each function
  for each head. Isn’t used for anything yet, but this is a TODO
  (defaults to `False`)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L165-L196)

process a single prompt, loading the activations and computing and
saving the figures

basically just calls `load_activations` and then
`compute_and_save_figures`

### Parameters:

- `prompt : dict`
- `model_cfg : HookedTransformerConfig|HTConfigMock`
- `force_overwrite : bool` (defaults to `False`)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L199-L262)

main function for generating figures from attention patterns, using the
functions in `ATTENTION_MATRIX_FIGURE_FUNCS`

### Parameters:

- `model_name : str` model name to use, used for loading the model
  config, prompts, activations, and saving the figures
- `save_path : str` base path to look in
- `n_samples : int` max number of samples to process
- `force : bool` force overwrite of existing figures. if `False`, will
  skip any functions which have already saved a figure
- `parallel : bool | int` whether to run in parallel. if `True`, will
  use all available cores. if `False`, will run in serial. if an int,
  will try to use that many cores (defaults to `True`)

### `def main`

``` python
()
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/figures.py#L265-L326)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

writes indexes to the model directory for the frontend to use or for
record keeping

## API Documentation

- [`generate_prompts_jsonl`](#generate_prompts_jsonl)
- [`generate_models_jsonl`](#generate_models_jsonl)
- [`get_func_metadata`](#get_func_metadata)
- [`generate_functions_jsonl`](#generate_functions_jsonl)
- [`write_html_index`](#write_html_index)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/indexes.py)

# `pattern_lens.indexes`

writes indexes to the model directory for the frontend to use or for
record keeping

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/indexes.py#L0-L121)

### `def generate_prompts_jsonl`

``` python
(model_dir: pathlib.Path)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/indexes.py#L14-L30)

creates a `prompts.jsonl` file with all the prompts in the model
directory

looks in all directories in `{model_dir}/prompts` for a `prompt.json`
file

### `def generate_models_jsonl`

``` python
(path: pathlib.Path)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/indexes.py#L33-L46)

creates a `models.jsonl` file with all the models

### `def get_func_metadata`

``` python
(func: Callable) -> dict[str, str | None]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/indexes.py#L49-L80)

get metadata for a function

### Parameters:

- `func : Callable`

### Returns:

`dict[str, str | None]` dictionary:

- `name : str` : the name of the function
- `doc : str` : the docstring of the function
- `figure_save_fmt : str | None` : the format of the figure that the
  function saves, using the `figure_save_fmt` attribute of the function.
  `None` if the attribute does not exist
- `source : str | None` : the source file of the function
- `code : str | None` : the source code of the function, split by line.
  `None` if the source file cannot be read

### `def generate_functions_jsonl`

``` python
(path: pathlib.Path)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/indexes.py#L83-L109)

unions all functions from file and current
`ATTENTION_MATRIX_FIGURE_FUNCS` into a `functions.jsonl` file

### `def write_html_index`

``` python
(path: pathlib.Path)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/indexes.py#L112-L122)

writes an index.html file to the path

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

loading activations from .npz on disk. implements some custom Exception
classes

## API Documentation

- [`GetActivationsError`](#GetActivationsError)
- [`ActivationsMissingError`](#ActivationsMissingError)
- [`ActivationsMismatchError`](#ActivationsMismatchError)
- [`InvalidPromptError`](#InvalidPromptError)
- [`compare_prompt_to_loaded`](#compare_prompt_to_loaded)
- [`augment_prompt_with_hash`](#augment_prompt_with_hash)
- [`load_activations`](#load_activations)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py)

# `pattern_lens.load_activations`

loading activations from .npz on disk. implements some custom Exception
classes

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L0-L156)

### `class GetActivationsError(builtins.ValueError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L12-L15)

base class for errors in getting activations

### Inherited Members

- [`ValueError`](#GetActivationsError.__init__)

- [`with_traceback`](#GetActivationsError.with_traceback)

- [`add_note`](#GetActivationsError.add_note)

- [`args`](#GetActivationsError.args)

### `class ActivationsMissingError(GetActivationsError, builtins.FileNotFoundError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L18-L21)

error for missing activations – can’t find the activations file

### Inherited Members

- [`ValueError`](#ActivationsMissingError.__init__)

- [`errno`](#ActivationsMissingError.errno)

- [`strerror`](#ActivationsMissingError.strerror)

- [`filename`](#ActivationsMissingError.filename)

- [`filename2`](#ActivationsMissingError.filename2)

- [`winerror`](#ActivationsMissingError.winerror)

- [`characters_written`](#ActivationsMissingError.characters_written)

- [`with_traceback`](#ActivationsMissingError.with_traceback)

- [`add_note`](#ActivationsMissingError.add_note)

- [`args`](#ActivationsMissingError.args)

### `class ActivationsMismatchError(GetActivationsError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L24-L30)

error for mismatched activations – the prompt text or hash do not match

raised by `compare_prompt_to_loaded`

### Inherited Members

- [`ValueError`](#ActivationsMismatchError.__init__)

- [`with_traceback`](#ActivationsMismatchError.with_traceback)

- [`add_note`](#ActivationsMismatchError.add_note)

- [`args`](#ActivationsMismatchError.args)

### `class InvalidPromptError(GetActivationsError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L33-L39)

error for invalid prompt – the prompt does not have fields “hash” or
“text”

raised by `augment_prompt_with_hash`

### Inherited Members

- [`ValueError`](#InvalidPromptError.__init__)

- [`with_traceback`](#InvalidPromptError.with_traceback)

- [`add_note`](#InvalidPromptError.add_note)

- [`args`](#InvalidPromptError.args)

### `def compare_prompt_to_loaded`

``` python
(prompt: dict, prompt_loaded: dict) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L42-L59)

compare a prompt to a loaded prompt, raise an error if they do not match

### Parameters:

- `prompt : dict`
- `prompt_loaded : dict`

### Returns:

- `None`

### Raises:

- `ActivationsMismatchError` : if the prompt text or hash do not match

### `def augment_prompt_with_hash`

``` python
(prompt: dict) -> dict
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L62-L88)

if a prompt does not have a hash, add one

not having a “text” field is allowed, but only if “hash” is present

### Parameters:

- `prompt : dict`

### Returns:

- `dict`

### Modifies:

the input `prompt` dictionary, if it does not have a `"hash"` key

### `def load_activations`

``` python
(
    model_name: str,
    prompt: dict,
    save_path: pathlib.Path,
    return_fmt: Literal['torch', 'numpy'] = 'torch'
) -> tuple[pathlib.Path, dict[str, torch.Tensor] | dict[str, numpy.ndarray]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/load_activations.py#L106-L157)

load activations for a prompt and model, from an npz file

### Parameters:

- `model_name : str`
- `prompt : dict`
- `save_path : Path`
- `return_fmt : Literal["torch", "numpy"]` (defaults to `"torch"`)

### Returns:

- `tuple[Path, dict[str, torch.Tensor]|dict[str, np.ndarray]]` the path
  to the activations file and the activations as a dictionary of numpy
  arrays or torch tensors, depending on `return_fmt`

### Raises:

- `ActivationsMissingError` : if the activations file is missing
- `ValueError` : if `return_fmt` is not `"torch"` or `"numpy"`

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

implements `load_text_data` for loading prompts

## API Documentation

- [`load_text_data`](#load_text_data)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/prompts.py)

# `pattern_lens.prompts`

implements `load_text_data` for loading prompts

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/prompts.py#L0-L80)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/prompts.py#L8-L81)

given `fname`, the path to a jsonl file, split prompts up into more
reasonable sizes

### Parameters:

- `fname : Path` jsonl file with prompts. Expects a list of dicts with a
  “text” key
- `min_chars : int | None` (defaults to `None`)
- `max_chars : int | None` (defaults to `None`)
- `shuffle : bool` (defaults to `False`)

### Returns:

- `list[dict]` new, processed list of prompts. Each prompt has a “text”
  key with a string value, and some metadata. this is not guaranteed to
  be the same length as the input list!

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.2.0

## Contents

cli for starting the server to show the web ui.

can also run with –rewrite-index to update the index.html file. this is
useful for working on the ui.

## API Documentation

- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/server.py)

# `pattern_lens.server`

cli for starting the server to show the web ui.

can also run with –rewrite-index to update the index.html file. this is
useful for working on the ui.

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/server.py#L0-L54)

### `def main`

``` python
(path: str, port: int = 8000)
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.2.0/server.py#L16-L26)
