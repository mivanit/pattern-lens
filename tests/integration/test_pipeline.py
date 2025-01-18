import pytest

from pattern_lens.activations import activations_main
from pattern_lens.figures import figures_main

SAVE_PATH_BASE: str = "tests/_temp/pipeline"
PROMPTS_PATH: str = "data/pile_100.jsonl"
N_SAMPLES: int = 3
MIN_CHARS: int = 32
MAX_CHARS: int = 128
FORCE: bool = True
NO_INDEX_HTML: bool = False
FIGURES_PARALLEL: bool = False


@pytest.mark.parametrize(
	"model_name",
	["pythia-14m", "tiny-stories-1M"],
)
def test_pipeline(model_name: str):
	activations_main(
		model_name=model_name,
		save_path=SAVE_PATH_BASE,
		prompts_path=PROMPTS_PATH,
		raw_prompts=True,
		min_chars=MIN_CHARS,
		max_chars=MAX_CHARS,
		force=FORCE,
		n_samples=N_SAMPLES,
		no_index_html=NO_INDEX_HTML,
	)
	figures_main(
		model_name=model_name,
		save_path=SAVE_PATH_BASE,
		n_samples=N_SAMPLES,
		force=FORCE,
		parallel=FIGURES_PARALLEL,
	)
