# pattern-lens
visualization of LLM attention patterns and things computed about them

`pattern-lens` makes it easy to:

- Generate visualizations of attention patterns, or figures computed from attention patterns, from models supported by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- Compare generated figures across models, layers, and heads in an [interactive web interface](https://miv.name/pattern-lens/demo/)

# Installation

```bash
pip install pattern-lens
```


# Usage

The pipeline is as follows:

- Generate attention patterns using `pattern_lens.activations.acitvations_main()`, saving them in `npz` files
- Generate visualizations using `pattern_lens.figures.figures_main()` -- read the `npz` files, pass each attention pattern to each visualization function, and save the resulting figures
- Serve the web interface using `pattern_lens.server` -- web interface reads metadata in json/jsonl files, then lets the user select figures to show


## Basic CLI

Generate attention patterns and default visualizations:

```bash
# generate activations
python -m pattern_lens.activations --model gpt2 --prompts data/pile_1k.jsonl --save-path attn_data
# create visualizations
python -m pattern_lens.figures --model gpt2 --save-path attn_data
```

serve the web UI:

```bash
python -m pattern_lens.server --path attn_data
```


## Web UI

View a demo of the web UI at [miv.name/pattern-lens/demo](https://miv.name/pattern-lens/demo/).

## Custom Visualizations

Add custom visualization functions by decorating them with `@register_attn_figure_func`. You should generate the activations first:
```
python -m pattern_lens.activations --model gpt2 --prompts data/pile_1k.jsonl --save-path attn_data
```


and then write+run a script/notebook that looks something like this:
```python
# imports
import matplotlib.pyplot as plt
from pattern_lens.figure_util import save_matrix_wrapper, matplotlib_figure_saver
from pattern_lens.attn_figure_funcs import register_attn_figure_func
from pattern_lens.figures import figures_main

# define your own functions


# run the figures pipelne


```