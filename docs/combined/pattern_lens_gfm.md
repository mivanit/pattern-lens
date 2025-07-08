> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

## Contents

[![PyPI](https://img.shields.io/pypi/v/pattern-lens)](https://pypi.org/project/pattern-lens/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pattern-lens)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://miv.name/pattern-lens)
[![Checks](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml/badge.svg)](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml)

[![Coverage](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCI+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImIiIHgyPSIwIiB5Mj0iMTAwJSI+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjYmJiIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPG1hc2sgaWQ9ImEiPgogICAgICAgIDxyZWN0IHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCIgcng9IjMiIGZpbGw9IiNmZmYiLz4KICAgIDwvbWFzaz4KICAgIDxnIG1hc2s9InVybCgjYSkiPgogICAgICAgIDxwYXRoIGZpbGw9IiM1NTUiIGQ9Ik0wIDBoNjN2MjBIMHoiLz4KICAgICAgICA8cGF0aCBmaWxsPSIjOTdDQTAwIiBkPSJNNjMgMGgzNnYyMEg2M3oiLz4KICAgICAgICA8cGF0aCBmaWxsPSJ1cmwoI2IpIiBkPSJNMCAwaDk5djIwSDB6Ii8+CiAgICA8L2c+CiAgICA8ZyBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iRGVqYVZ1IFNhbnMsVmVyZGFuYSxHZW5ldmEsc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSI+CiAgICAgICAgPHRleHQgeD0iMzEuNSIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPmNvdmVyYWdlPC90ZXh0PgogICAgICAgIDx0ZXh0IHg9IjMxLjUiIHk9IjE0Ij5jb3ZlcmFnZTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPjkzJTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTQiPjkzJTwvdGV4dD4KICAgIDwvZz4KPC9zdmc+Cg==)](docs/coverage/html/)
![GitHub
commits](https://img.shields.io/github/commit-activity/t/mivanit/pattern-lens)
![GitHub commit
activity](https://img.shields.io/github/commit-activity/m/mivanit/pattern-lens)
![GitHub closed pull
requests](https://img.shields.io/github/issues-pr-closed/mivanit/pattern-lens)
![code size,
bytes](https://img.shields.io/github/languages/code-size/mivanit/pattern-lens)

|                                                                                                            |                                                                                                            |
|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
| [Inspect patterns across models, heads, prompts, etc.](https://miv.name/pattern-lens/assets/pl-demo.html)  |               [Inspect a single pattern](https://miv.name/pattern-lens/assets/sg-demo.html)                |
| [![](https://miv.name/pattern-lens/assets/pl-demo.png)](https://miv.name/pattern-lens/assets/pl-demo.html) | [![](https://miv.name/pattern-lens/assets/sg-demo.png)](https://miv.name/pattern-lens/assets/sg-demo.html) |

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

pattern-lens provides two complementary web interfaces for exploring
attention patterns:

- The main interface for comparing attention patterns across models,
  layers, and heads
  - Filter and select patterns by model, layer, head, prompt, etc.
  - View multiple patterns simultaneously in a grid layout
  - Click patterns to open detailed single-pattern view
- A focused interface for detailed examination of individual attention
  patterns
  - Interactive heatmap with hover highlights and keyboard navigation
  - Token-by-token analysis with Q/K axis highlighting

View a demo of the web UI at
[miv.name/pattern-lens/demo](https://miv.name/pattern-lens/demo/).

Much of this web UI is inspired by
[`CircuitsVis`](https://github.com/TransformerLensOrg/CircuitsVis), but
with a focus on just attention patterns and figures computed from them.
I have also tried to make the interface a bit simpler, more flexible,
and faster.

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

See `demo.ipynb` for a full example.

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0__init__.py)

# `pattern_lens`

[![PyPI](https://img.shields.io/pypi/v/pattern-lens)](https://pypi.org/project/pattern-lens/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pattern-lens)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://miv.name/pattern-lens)
[![Checks](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml/badge.svg)](https://github.com/mivanit/pattern-lens/actions/workflows/checks.yml)

[![Coverage](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCI+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImIiIHgyPSIwIiB5Mj0iMTAwJSI+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjYmJiIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPG1hc2sgaWQ9ImEiPgogICAgICAgIDxyZWN0IHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCIgcng9IjMiIGZpbGw9IiNmZmYiLz4KICAgIDwvbWFzaz4KICAgIDxnIG1hc2s9InVybCgjYSkiPgogICAgICAgIDxwYXRoIGZpbGw9IiM1NTUiIGQ9Ik0wIDBoNjN2MjBIMHoiLz4KICAgICAgICA8cGF0aCBmaWxsPSIjOTdDQTAwIiBkPSJNNjMgMGgzNnYyMEg2M3oiLz4KICAgICAgICA8cGF0aCBmaWxsPSJ1cmwoI2IpIiBkPSJNMCAwaDk5djIwSDB6Ii8+CiAgICA8L2c+CiAgICA8ZyBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iRGVqYVZ1IFNhbnMsVmVyZGFuYSxHZW5ldmEsc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSI+CiAgICAgICAgPHRleHQgeD0iMzEuNSIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPmNvdmVyYWdlPC90ZXh0PgogICAgICAgIDx0ZXh0IHg9IjMxLjUiIHk9IjE0Ij5jb3ZlcmFnZTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPjkzJTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTQiPjkzJTwvdGV4dD4KICAgIDwvZz4KPC9zdmc+Cg==)](docs/coverage/html/)
![GitHub
commits](https://img.shields.io/github/commit-activity/t/mivanit/pattern-lens)
![GitHub commit
activity](https://img.shields.io/github/commit-activity/m/mivanit/pattern-lens)
![GitHub closed pull
requests](https://img.shields.io/github/issues-pr-closed/mivanit/pattern-lens)
![code size,
bytes](https://img.shields.io/github/languages/code-size/mivanit/pattern-lens)

|                                                                                                            |                                                                                                            |
|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
| [Inspect patterns across models, heads, prompts, etc.](https://miv.name/pattern-lens/assets/pl-demo.html)  |               [Inspect a single pattern](https://miv.name/pattern-lens/assets/sg-demo.html)                |
| [![](https://miv.name/pattern-lens/assets/pl-demo.png)](https://miv.name/pattern-lens/assets/pl-demo.html) | [![](https://miv.name/pattern-lens/assets/sg-demo.png)](https://miv.name/pattern-lens/assets/sg-demo.html) |

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

pattern-lens provides two complementary web interfaces for exploring
attention patterns:

- The main interface for comparing attention patterns across models,
  layers, and heads
  - Filter and select patterns by model, layer, head, prompt, etc.
  - View multiple patterns simultaneously in a grid layout
  - Click patterns to open detailed single-pattern view
- A focused interface for detailed examination of individual attention
  patterns
  - Interactive heatmap with hover highlights and keyboard navigation
  - Token-by-token analysis with Q/K axis highlighting

View a demo of the web UI at
[miv.name/pattern-lens/demo](https://miv.name/pattern-lens/demo/).

Much of this web UI is inspired by
[`CircuitsVis`](https://github.com/TransformerLensOrg/CircuitsVis), but
with a focus on just attention patterns and figures computed from them.
I have also tried to make the interface a bit simpler, more flexible,
and faster.

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

See `demo.ipynb` for a full example.

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0__init__.py#L0-L12)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

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
- [`DEFAULT_DEVICE`](#DEFAULT_DEVICE)
- [`activations_main`](#activations_main)
- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0activations.py)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0activations.py#L0-L656)

### `def compute_activations`

``` python
(
    prompt: dict,
    model: transformer_lens.HookedTransformer.HookedTransformer | None = None,
    save_path: pathlib.Path = PosixPath('attn_data'),
    names_filter: Callable[[str], bool] | re.Pattern = re.compile('blocks\\.(\\d+)\\.attn\\.hook_pattern'),
    return_cache: Literal[None, 'numpy', 'torch'] = 'torch',
    stack_heads: bool = False
) -> tuple[pathlib.Path, dict[str, numpy.ndarray] | transformer_lens.ActivationCache.ActivationCache | jaxtyping.Float[ndarray, 'n_layers n_heads n_ctx n_ctx'] | jaxtyping.Float[Tensor, 'n_layers n_heads n_ctx n_ctx'] | None]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0activations.py#L120-L278)

get activations for a given model and prompt, possibly from a cache

if from a cache, prompt_meta must be passed and contain the prompt hash

### Parameters:

- `prompt : dict | None` (defaults to `None`)
- `model : HookedTransformer`
- `save_path : Path` (defaults to `Path(DATA_DIR)`)
- `names_filter : Callable[[str], bool]|re.Pattern` a filter for the
  names of the activations to return. if an `re.Pattern`, will use
  `lambda key: names_filter.match(key) is not None` (defaults to
  `ATTN_PATTERN_REGEX`)
- `return_cache : Literal[None, "numpy", "torch"]` will return `None` as
  the second element if `None`, otherwise will return the cache in the
  specified tensor format. `stack_heads` still affects whether it will
  be a dict (False) or a single tensor (True) (defaults to `None`)
- `stack_heads : bool` whether the heads should be stacked in the
  output. this causes a number of changes:
- `npy` file with a single `(n_layers, n_heads, n_ctx, n_ctx)` tensor
  saved for each prompt instead of `npz` file with dict by layer
- `cache` will be a single `(n_layers, n_heads, n_ctx, n_ctx)` tensor
  instead of a dict by layer if `return_cache` is `True` will assert
  that everything in the activation cache is only attention patterns,
  and is all of the attention patterns. raises an exception if not.

### Returns:

    tuple[
            Path,
            Union[
                    None,
                    ActivationCacheNp, ActivationCache,
                    Float[np.ndarray, "n_layers n_heads n_ctx n_ctx"], Float[torch.Tensor, "n_layers n_heads n_ctx n_ctx"],
            ]
    ]

### `def get_activations`

``` python
(
    prompt: dict,
    model: transformer_lens.HookedTransformer.HookedTransformer | str,
    save_path: pathlib.Path = PosixPath('attn_data'),
    allow_disk_cache: bool = True,
    return_cache: Literal[None, 'numpy', 'torch'] = 'numpy'
) -> tuple[pathlib.Path, dict[str, numpy.ndarray] | transformer_lens.ActivationCache.ActivationCache | None]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0activations.py#L305-L369)

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
- `return_cache : Literal[None, "numpy", "torch"]` whether to return the
  cache, and in what format (defaults to `"numpy"`)

### Returns:

- `tuple[Path, ActivationCacheNp | ActivationCache | None]` the path to
  the activations and the cache if `return_cache is not None`

- `DEFAULT_DEVICE: torch.device = device(type='cuda')`

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
    stacked_heads: bool = False,
    device: str | torch.device = device(type='cuda')
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0activations.py#L377-L514)

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
- `stacked_heads : bool` whether to stack the heads in the output
  tensor. will save as `.npy` instead of `.npz` if `True` (defaults to
  `False`)
- `device : str | torch.device` the device to use. if a string, will be
  passed to `torch.device`

### `def main`

``` python
() -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0activations.py#L517-L653)

generate attention pattern activations for a model and prompts

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

## Contents

default figure functions

- If you are making a PR, add your new figure function here.
- if you are using this as a library, then you can see examples here

note that for `pattern_lens.figures` to recognize your function, you
need to use the `register_attn_figure_func` decorator which adds your
function to `ATTENTION_MATRIX_FIGURE_FUNCS`

## API Documentation

- [`ATTENTION_MATRIX_FIGURE_FUNCS`](#ATTENTION_MATRIX_FIGURE_FUNCS)
- [`get_all_figure_names`](#get_all_figure_names)
- [`register_attn_figure_func`](#register_attn_figure_func)
- [`register_attn_figure_multifunc`](#register_attn_figure_multifunc)
- [`raw`](#raw)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0attn_figure_funcs.py)

# `pattern_lens.attn_figure_funcs`

default figure functions

- If you are making a PR, add your new figure function here.
- if you are using this as a library, then you can see examples here

note that for `<a href="figures.html">pattern_lens.figures</a>` to
recognize your function, you need to use the `register_attn_figure_func`
decorator which adds your function to `ATTENTION_MATRIX_FIGURE_FUNCS`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0attn_figure_funcs.py#L0-L117)

- `ATTENTION_MATRIX_FIGURE_FUNCS: list[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]] = [<function raw>]`

### `def get_all_figure_names`

``` python
() -> list[str]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0attn_figure_funcs.py#L27-L38)

get all figure names

### `def register_attn_figure_func`

``` python
(
    func: Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]
) -> Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0attn_figure_funcs.py#L41-L73)

decorator for registering attention matrix figure function

if you want to add a new figure function, you should use this decorator

### Parameters:

- `func : AttentionMatrixFigureFunc` your function, which should take an
  attention matrix and path

### Returns:

- `AttentionMatrixFigureFunc` your function, after we add it to
  `ATTENTION_MATRIX_FIGURE_FUNCS`

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

### `def register_attn_figure_multifunc`

``` python
(
    names: Sequence[str]
) -> Callable[[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]], Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0attn_figure_funcs.py#L76-L91)

decorator which registers a function as a multi-figure function

### `def raw`

``` python
(
    attn_matrix: jaxtyping.Float[ndarray, 'n_ctx n_ctx']
) -> jaxtyping.Float[ndarray, 'n m']
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0attn_figure_funcs.py#L94-L98)

raw attention matrix

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

## Contents

implements some constants and types

## API Documentation

- [`AttentionMatrix`](#AttentionMatrix)
- [`ActivationCacheNp`](#ActivationCacheNp)
- [`ActivationCacheTorch`](#ActivationCacheTorch)
- [`DATA_DIR`](#DATA_DIR)
- [`ATTN_PATTERN_REGEX`](#ATTN_PATTERN_REGEX)
- [`SPINNER_KWARGS`](#SPINNER_KWARGS)
- [`DIVIDER_S1`](#DIVIDER_S1)
- [`DIVIDER_S2`](#DIVIDER_S2)
- [`ReturnCache`](#ReturnCache)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0consts.py)

# `pattern_lens.consts`

implements some constants and types

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0consts.py#L0-L36)

- `AttentionMatrix = <class 'jaxtyping.Float[ndarray, 'n_ctx n_ctx']'>`

type alias for attention matrix

- `ActivationCacheNp = dict[str, numpy.ndarray]`

type alias for a cache of activations, like a
transformer_lens.ActivationCache

- `ActivationCacheTorch = dict[str, torch.Tensor]`

type alias for a cache of activations, like a
transformer_lens.ActivationCache but without the extras. useful for when
loading from an npz file

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

- `ReturnCache = typing.Literal[None, 'numpy', 'torch']`

return type for a cache of activations

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

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
- [`matplotlib_multifigure_saver`](#matplotlib_multifigure_saver)
- [`matrix_to_image_preprocess`](#matrix_to_image_preprocess)
- [`matrix2drgb_to_png_bytes`](#matrix2drgb_to_png_bytes)
- [`matrix_as_svg`](#matrix_as_svg)
- [`save_matrix_wrapper`](#save_matrix_wrapper)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py)

# `pattern_lens.figure_util`

implements a bunch of types, default values, and templates which are
useful for figure functions

notably, you can use the decorators `matplotlib_figure_saver`,
`save_matrix_wrapper` to make your functions save figures

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py#L0-L514)

- `AttentionMatrixFigureFunc = collections.abc.Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]`

Type alias for a function that, given an attention matrix, saves one or
more figures

- `Matrix2D = <class 'jaxtyping.Float[ndarray, 'n m']'>`

Type alias for a 2D matrix (plottable)

- `Matrix2Drgb = <class 'jaxtyping.UInt8[ndarray, 'n m rgb=3']'>`

Type alias for a 2D matrix with 3 channels (RGB)

- `AttentionMatrixToMatrixFunc = collections.abc.Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx']], jaxtyping.Float[ndarray, 'n m']]`

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
    func: Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], None] | None = None,
    fmt: str = 'svgz'
) -> Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None] | Callable[[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], matplotlib.axes._axes.Axes], None], str], Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py#L67-L125)

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

### `def matplotlib_multifigure_saver`

``` python
(
    names: Sequence[str],
    fmt: str = 'svgz'
) -> Callable[[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], dict[str, matplotlib.axes._axes.Axes]], None]], Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py#L128-L193)

decorate a function such that it saves multiple figures, one for each
name in `names`

### Parameters:

- `names : Sequence[str]` the names of the figures to save
- `fmt : str` format for saving matplotlib figures (defaults to
  `MATPLOTLIB_FIGURE_FMT`)

### Returns:

- `Callable[[Callable[[AttentionMatrix, dict[str, plt.Axes]], None], AttentionMatrixFigureFunc]`
  the decorator, which will then be applied to the function we expect
  the decorated function to take an attention pattern, and a dict of
  axes corresponding to the names

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py#L196-L293)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py#L300-L326)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py#L329-L383)

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
    func: Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx']], jaxtyping.Float[ndarray, 'n m']] | None = None,
    *args,
    fmt: Literal['png', 'svg', 'svgz'] = 'svgz',
    normalize: bool = False,
    cmap: str | matplotlib.colors.Colormap = 'viridis',
    diverging_colormap: bool = False,
    normalize_min: float | None = None
) -> Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None] | Callable[[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx']], jaxtyping.Float[ndarray, 'n m']]], Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figure_util.py#L406-L515)

Decorator for functions that process an attention matrix and save it as
an SVGZ image.

Can handle both argumentless usage and with arguments.

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
> v0.5.0

## Contents

code for generating figures from attention patterns, using the functions
decorated with `register_attn_figure_func`

## API Documentation

- [`HTConfigMock`](#HTConfigMock)
- [`process_single_head`](#process_single_head)
- [`compute_and_save_figures`](#compute_and_save_figures)
- [`process_prompt`](#process_prompt)
- [`select_attn_figure_funcs`](#select_attn_figure_funcs)
- [`figures_main`](#figures_main)
- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py)

# `pattern_lens.figures`

code for generating figures from attention patterns, using the functions
decorated with `register_attn_figure_func`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L0-L471)

### `class HTConfigMock:`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L41-L67)

Mock of `transformer_lens.HookedTransformerConfig` for type hinting and
loading config json

can be initialized with any kwargs, and will update its `__dict__` with
them. does, however, require the following attributes: -
`n_layers: int` - `n_heads: int` - `model_name: str`

we do this to avoid having to import `torch` and `transformer_lens`,
since this would have to be done for each process in the parallelization
and probably slows things down significantly

### `HTConfigMock`

``` python
(**kwargs: dict[str, str | int])
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L52-L57)

will pass all kwargs to `__dict__`

- `n_layers: int`

- `n_heads: int`

- `model_name: str`

### `def serialize`

``` python
(self) -> dict
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L59-L62)

serialize the config to json. values which aren’t serializable will be
converted via `muutils.json_serialize.json_serialize`

### `def load`

``` python
(cls, data: dict) -> pattern_lens.figures.HTConfigMock
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L64-L67)

try to load a config from a dict, using the `__init__` method

### `def process_single_head`

``` python
(
    layer_idx: int,
    head_idx: int,
    attn_pattern: jaxtyping.Float[ndarray, 'n_ctx n_ctx'],
    save_dir: pathlib.Path,
    figure_funcs: list[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]],
    force_overwrite: bool = False
) -> dict[str, bool | Exception]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L70-L123)

process a single head’s attention pattern, running all the functions in
`figure_funcs` on the attention pattern

> \[gotcha:\] if `force_overwrite` is `False`, and we used a
> multi-figure function, it will skip all figures for that function if
> any are already saved and it assumes a format of
> `{func_name}.{figure_name}.{fmt}` for the saved figures

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
    cache: dict[str, numpy.ndarray] | jaxtyping.Float[ndarray, 'n_layers n_heads n_ctx n_ctx'],
    figure_funcs: list[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]],
    save_path: pathlib.Path = PosixPath('attn_data'),
    force_overwrite: bool = False,
    track_results: bool = False
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L126-L199)

compute and save figures for all heads in the model, using the functions
in `ATTENTION_MATRIX_FIGURE_FUNCS`

### Parameters:

- `model_cfg : HookedTransformerConfig|HTConfigMock` configuration of
  the model, used for loading the activations
- `cache : ActivationCacheNp | Float[np.ndarray, &quot;n_layers n_heads n_ctx n_ctx&quot;]`
  activation cache containing actual patterns for the prompt we are
  processing
- `figure_funcs : list[AttentionMatrixFigureFunc]` list of functions to
  run
- `save_path : Path` directory to save the figures to (defaults to
  `Path(DATA_DIR)`)
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
    figure_funcs: list[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]],
    force_overwrite: bool = False
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L202-L245)

process a single prompt, loading the activations and computing and
saving the figures

basically just calls `load_activations` and then
`compute_and_save_figures`

### Parameters:

- `prompt : dict` prompt to process, should be a dict with the following
  keys: - `"text"`: the prompt string - `"hash"`: the hash of the prompt
- `model_cfg : HookedTransformerConfig|HTConfigMock` configuration of
  the model, used for figuring out where to save
- `save_path : Path` directory to save the figures to
- `figure_funcs : list[AttentionMatrixFigureFunc]` list of functions to
  run
- `force_overwrite : bool` (defaults to `False`)

### `def select_attn_figure_funcs`

``` python
(
    figure_funcs_select: set[str] | str | None = None
) -> list[Callable[[jaxtyping.Float[ndarray, 'n_ctx n_ctx'], pathlib.Path], None]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L248-L284)

given a selector, figure out which functions from
`ATTENTION_MATRIX_FIGURE_FUNCS` to use

- if arg is `None`, will use all functions
- if a string, will use the function names which match the string
  (glob/fnmatch syntax)
- if a set, will use functions whose names are in the set

### `def figures_main`

``` python
(
    model_name: str,
    save_path: str,
    n_samples: int,
    force: bool,
    figure_funcs_select: set[str] | str | None = None,
    parallel: bool | int = True
) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L287-L368)

main function for generating figures from attention patterns, using the
functions in `ATTENTION_MATRIX_FIGURE_FUNCS`

### Parameters:

- `model_name : str` model name to use, used for loading the model
  config, prompts, activations, and saving the figures
- `save_path : str` base path to look in
- `n_samples : int` max number of samples to process
- `force : bool` force overwrite of existing figures. if `False`, will
  skip any functions which have already saved a figure
- `figure_funcs_select : set[str]|str|None` figure functions to use. if
  `None`, will use all functions. if a string, will use the function
  names which match the string. if a set, will use the function names in
  the set (defaults to `None`)
- `parallel : bool | int` whether to run in parallel. if `True`, will
  use all available cores. if `False`, will run in serial. if an int,
  will try to use that many cores (defaults to `True`)

### `def main`

``` python
() -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0figures.py#L441-L468)

generates figures from the activations using the functions decorated
with `register_attn_figure_func`

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0indexes.py)

# `pattern_lens.indexes`

writes indexes to the model directory for the frontend to use or for
record keeping

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0indexes.py#L0-L150)

### `def generate_prompts_jsonl`

``` python
(model_dir: pathlib.Path) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0indexes.py#L17-L33)

creates a `prompts.jsonl` file with all the prompts in the model
directory

looks in all directories in `{model_dir}/prompts` for a `prompt.json`
file

### `def generate_models_jsonl`

``` python
(path: pathlib.Path) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0indexes.py#L36-L49)

creates a `models.jsonl` file with all the models

### `def get_func_metadata`

``` python
(func: Callable) -> list[dict[str, str | None]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0indexes.py#L52-L101)

get metadata for a function

### Parameters:

- `func : Callable` which has a `_FIGURE_NAMES_KEY` (by default
  `_figure_names`) attribute

### Returns:

`list[dict[str, str | None]]` each dictionary is for a function,
containing:

- `name : str` : the name of the figure
- `func_name : str` the name of the function. if not a multi-figure
  function, this is identical to `name` if it is a multi-figure
  function, then `name` is `{func_name}.{figure_name}`
- `doc : str` : the docstring of the function
- `figure_save_fmt : str | None` : the format of the figure that the
  function saves, using the `figure_save_fmt` attribute of the function.
  `None` if the attribute does not exist
- `source : str | None` : the source file of the function
- `code : str | None` : the source code of the function, split by line.
  `None` if the source file cannot be read

### `def generate_functions_jsonl`

``` python
(path: pathlib.Path) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0indexes.py#L104-L133)

unions all functions from `figures.jsonl` and
`ATTENTION_MATRIX_FIGURE_FUNCS` into the file

### `def write_html_index`

``` python
(path: pathlib.Path) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0indexes.py#L136-L151)

writes index.html and single.html files to the path (version replacement
handled by makefile)

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py)

# `pattern_lens.load_activations`

loading activations from .npz on disk. implements some custom Exception
classes

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L0-L166)

### `class GetActivationsError(builtins.ValueError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L14-L17)

base class for errors in getting activations

### Inherited Members

- [`ValueError`](#GetActivationsError.__init__)

- [`with_traceback`](#GetActivationsError.with_traceback)

- [`add_note`](#GetActivationsError.add_note)

- [`args`](#GetActivationsError.args)

### `class ActivationsMissingError(GetActivationsError, builtins.FileNotFoundError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L20-L23)

error for missing activations – can’t find the activations file

### Inherited Members

- [`ValueError`](#ActivationsMissingError.__init__)

- [`errno`](#ActivationsMissingError.errno)

- [`strerror`](#ActivationsMissingError.strerror)

- [`filename`](#ActivationsMissingError.filename)

- [`filename2`](#ActivationsMissingError.filename2)

- [`characters_written`](#ActivationsMissingError.characters_written)

- [`with_traceback`](#ActivationsMissingError.with_traceback)

- [`add_note`](#ActivationsMissingError.add_note)

- [`args`](#ActivationsMissingError.args)

### `class ActivationsMismatchError(GetActivationsError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L26-L32)

error for mismatched activations – the prompt text or hash do not match

raised by `compare_prompt_to_loaded`

### Inherited Members

- [`ValueError`](#ActivationsMismatchError.__init__)

- [`with_traceback`](#ActivationsMismatchError.with_traceback)

- [`add_note`](#ActivationsMismatchError.add_note)

- [`args`](#ActivationsMismatchError.args)

### `class InvalidPromptError(GetActivationsError):`

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L35-L41)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L44-L62)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L65-L93)

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
    return_fmt: Literal[None, 'numpy', 'torch'] = 'torch'
) -> tuple[pathlib.Path, dict[str, torch.Tensor] | dict[str, numpy.ndarray]]
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0load_activations.py#L111-L164)

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
> v0.5.0

## Contents

implements `load_text_data` for loading prompts

## API Documentation

- [`load_text_data`](#load_text_data)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0prompts.py)

# `pattern_lens.prompts`

implements `load_text_data` for loading prompts

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0prompts.py#L0-L81)

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
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0prompts.py#L8-L82)

given `fname`, the path to a jsonl file, split prompts up into more
reasonable sizes

### Parameters:

- `fname : Path` jsonl file with prompts. Expects a list of dicts with a
  “text” key
- `min_chars : int | None` (defaults to `None`)
- `max_chars : int | None` (defaults to `None`)
- `shuffle : bool` (defaults to `False`)

### Returns:

- `list[dict]` processed list of prompts. Each prompt has a “text” key
  w/ a string value and some metadata. this is not guaranteed to be the
  same length as the input list!

> docs for [`pattern_lens`](https://github.com/mivanit/pattern-lens)
> v0.5.0

## Contents

cli for starting the server to show the web ui.

can also run with –rewrite-index to update the index.html file. this is
useful for working on the ui.

## API Documentation

- [`main`](#main)

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0server.py)

# `pattern_lens.server`

cli for starting the server to show the web ui.

can also run with –rewrite-index to update the index.html file. this is
useful for working on the ui.

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0server.py#L0-L58)

### `def main`

``` python
(path: str | None = None, port: int = 8000) -> None
```

[View Source on
GitHub](https://github.com/mivanit/pattern-lens/blob/0.5.0server.py#L17-L30)

move to the given path and start the server
