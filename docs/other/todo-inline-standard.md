 # Inline TODOs


# TODO

## [`pattern_lens/activations.py`](/pattern_lens/activations.py)

- batching?  
  local link: [`/pattern_lens/activations.py:203`](/pattern_lens/activations.py#L203) 
  | view on GitHub: [pattern_lens/activations.py#L203](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/activations.py#L203)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=batching%3F&body=%23%20source%0A%0A%5B%60pattern_lens%2Factivations.py%23L203%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Factivations.py%23L203%29%0A%0A%23%20context%0A%60%60%60python%0A%09with%20torch.no_grad%28%29%3A%0A%09%09model.eval%28%29%0A%09%09%23%20TODO%3A%20batching%3F%0A%09%09_%2C%20cache_torch%20%3D%20model.run_with_cache%28%0A%09%09%09prompt_str%2C%0A%60%60%60&labels=enhancement)

  ```python
  with torch.no_grad():
  	model.eval()
  	# TODO: batching?
  	_, cache_torch = model.run_with_cache(
  		prompt_str,
  ```


- this basically does nothing, since we load the activations and then immediately get rid of them.  
  local link: [`/pattern_lens/activations.py:353`](/pattern_lens/activations.py#L353) 
  | view on GitHub: [pattern_lens/activations.py#L353](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/activations.py#L353)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=this%20basically%20does%20nothing%2C%20since%20we%20load%20the%20activations%20and%20then%20immediately%20get%20rid%20of%20them.&body=%23%20source%0A%0A%5B%60pattern_lens%2Factivations.py%23L353%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Factivations.py%23L353%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09return%20path%2C%20cache%0A%09%09%09else%3A%0A%09%09%09%09%23%20TODO%3A%20this%20basically%20does%20nothing%2C%20since%20we%20load%20the%20activations%20and%20then%20immediately%20get%20rid%20of%20them.%0A%09%09%09%09%23%20maybe%20refactor%20this%20so%20that%20load_activations%20can%20take%20a%20parameter%20to%20simply%20assert%20that%20the%20cache%20exists%3F%0A%09%09%09%09%23%20this%20will%20let%20us%20avoid%20loading%20it%2C%20which%20slows%20things%20down%0A%60%60%60&labels=enhancement)

  ```python
  	return path, cache
  else:
  	# TODO: this basically does nothing, since we load the activations and then immediately get rid of them.
  	# maybe refactor this so that load_activations can take a parameter to simply assert that the cache exists?
  	# this will let us avoid loading it, which slows things down
  ```


- not implemented yet  
  local link: [`/pattern_lens/activations.py:485`](/pattern_lens/activations.py#L485) 
  | view on GitHub: [pattern_lens/activations.py#L485](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/activations.py#L485)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=not%20implemented%20yet&body=%23%20source%0A%0A%5B%60pattern_lens%2Factivations.py%23L485%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Factivations.py%23L485%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09write_html_index%28save_path_p%29%0A%0A%09%23%20TODO%3A%20not%20implemented%20yet%0A%09if%20stacked_heads%3A%0A%09%09raise%20NotImplementedError%28%22stacked_heads%20not%20implemented%20yet%22%29%0A%60%60%60&labels=enhancement)

  ```python
  		write_html_index(save_path_p)

  # TODO: not implemented yet
  if stacked_heads:
  	raise NotImplementedError("stacked_heads not implemented yet")
  ```




## [`pattern_lens/figures.py`](/pattern_lens/figures.py)

- do something with results  
  local link: [`/pattern_lens/figures.py:185`](/pattern_lens/figures.py#L185) 
  | view on GitHub: [pattern_lens/figures.py#L185](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/figures.py#L185)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=do%20something%20with%20results&body=%23%20source%0A%0A%5B%60pattern_lens%2Ffigures.py%23L185%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Ffigures.py%23L185%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09results%5Bfunc_name%5D%5B%28layer_idx%2C%20head_idx%29%5D%20%3D%20status%0A%0A%09%23%20TODO%3A%20do%20something%20with%20results%0A%0A%09generate_prompts_jsonl%28save_path%20%2F%20model_cfg.model_name%29%0A%60%60%60&labels=enhancement)

  ```python
  			results[func_name][(layer_idx, head_idx)] = status

  # TODO: do something with results

  generate_prompts_jsonl(save_path / model_cfg.model_name)
  ```




## [`tests/integration/test_clis.py`](/tests/integration/test_clis.py)

- make these mock checks work -- I have no idea how to use mock properly  
  local link: [`/tests/integration/test_clis.py:147`](/tests/integration/test_clis.py#L147) 
  | view on GitHub: [tests/integration/test_clis.py#L147](https://github.com/mivanit/pattern-lens/blob/main/tests/integration/test_clis.py#L147)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=make%20these%20mock%20checks%20work%20--%20I%20have%20no%20idea%20how%20to%20use%20mock%20properly&body=%23%20source%0A%0A%5B%60tests%2Fintegration%2Ftest_clis.py%23L147%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Ftests%2Fintegration%2Ftest_clis.py%23L147%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09server_main%28%29%0A%0A%09%09%23%20TODO%3A%20make%20these%20mock%20checks%20work%20--%20I%20have%20no%20idea%20how%20to%20use%20mock%20properly%0A%09%09%23%20%23%20Check%20that%20write_html_index%20was%20called%0A%09%09%23%20mock_write_html.assert_called_once%28%29%0A%60%60%60&labels=enhancement)

  ```python
  	server_main()

  # TODO: make these mock checks work -- I have no idea how to use mock properly
  # # Check that write_html_index was called
  # mock_write_html.assert_called_once()
  ```





# TYPING

## [`pattern_lens/figure_util.py`](/pattern_lens/figure_util.py)

- error: Item "SubFigure" of "Figure | SubFigure" has no attribute "tight_layout"  [union-attr]  
  local link: [`/pattern_lens/figure_util.py:178`](/pattern_lens/figure_util.py#L178) 
  | view on GitHub: [pattern_lens/figure_util.py#L178](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/figure_util.py#L178)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=error%3A%20Item%20%22SubFigure%22%20of%20%22Figure%20%7C%20SubFigure%22%20has%20no%20attribute%20%22tight_layout%22%20%20%5Bunion-attr%5D&body=%23%20source%0A%0A%5B%60pattern_lens%2Ffigure_util.py%23L178%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Ffigure_util.py%23L178%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09for%20name%2C%20fig_%20in%20figs_dict.items%28%29%3A%0A%09%09%09%09%09fig_path%3A%20Path%20%3D%20save_dir%20%2F%20f%22%7Bfunc_name%7D.%7Bname%7D.%7Bfmt%7D%22%0A%09%09%09%09%09%23%20TYPING%3A%20error%3A%20Item%20%22SubFigure%22%20of%20%22Figure%20%7C%20SubFigure%22%20has%20no%20attribute%20%22tight_layout%22%20%20%5Bunion-attr%5D%0A%09%09%09%09%09fig_.tight_layout%28%29%20%20%23%20type%3A%20ignore%5Bunion-attr%5D%0A%09%09%09%09%09%23%20TYPING%3A%20error%3A%20Item%20%22SubFigure%22%20of%20%22Figure%20%7C%20SubFigure%22%20has%20no%20attribute%20%22savefig%22%20%20%5Bunion-attr%5D%0A%60%60%60&labels=TYPING)

  ```python
  for name, fig_ in figs_dict.items():
  	fig_path: Path = save_dir / f"{func_name}.{name}.{fmt}"
  	# TYPING: error: Item "SubFigure" of "Figure | SubFigure" has no attribute "tight_layout"  [union-attr]
  	fig_.tight_layout()  # type: ignore[union-attr]
  	# TYPING: error: Item "SubFigure" of "Figure | SubFigure" has no attribute "savefig"  [union-attr]
  ```


- error: Item "SubFigure" of "Figure | SubFigure" has no attribute "savefig"  [union-attr]  
  local link: [`/pattern_lens/figure_util.py:180`](/pattern_lens/figure_util.py#L180) 
  | view on GitHub: [pattern_lens/figure_util.py#L180](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/figure_util.py#L180)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=error%3A%20Item%20%22SubFigure%22%20of%20%22Figure%20%7C%20SubFigure%22%20has%20no%20attribute%20%22savefig%22%20%20%5Bunion-attr%5D&body=%23%20source%0A%0A%5B%60pattern_lens%2Ffigure_util.py%23L180%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Ffigure_util.py%23L180%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09%09%23%20TYPING%3A%20error%3A%20Item%20%22SubFigure%22%20of%20%22Figure%20%7C%20SubFigure%22%20has%20no%20attribute%20%22tight_layout%22%20%20%5Bunion-attr%5D%0A%09%09%09%09%09fig_.tight_layout%28%29%20%20%23%20type%3A%20ignore%5Bunion-attr%5D%0A%09%09%09%09%09%23%20TYPING%3A%20error%3A%20Item%20%22SubFigure%22%20of%20%22Figure%20%7C%20SubFigure%22%20has%20no%20attribute%20%22savefig%22%20%20%5Bunion-attr%5D%0A%09%09%09%09%09fig_.savefig%28fig_path%29%20%20%23%20type%3A%20ignore%5Bunion-attr%5D%0A%09%09%09finally%3A%0A%60%60%60&labels=TYPING)

  ```python
  		# TYPING: error: Item "SubFigure" of "Figure | SubFigure" has no attribute "tight_layout"  [union-attr]
  		fig_.tight_layout()  # type: ignore[union-attr]
  		# TYPING: error: Item "SubFigure" of "Figure | SubFigure" has no attribute "savefig"  [union-attr]
  		fig_.savefig(fig_path)  # type: ignore[union-attr]
  finally:
  ```


- error: Argument 1 to "close" has incompatible type "Figure | SubFigure"; expected "int | str | Figure | Literal['all'] | None"  [arg-type]  
  local link: [`/pattern_lens/figure_util.py:185`](/pattern_lens/figure_util.py#L185) 
  | view on GitHub: [pattern_lens/figure_util.py#L185](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/figure_util.py#L185)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=error%3A%20Argument%201%20to%20%22close%22%20has%20incompatible%20type%20%22Figure%20%7C%20SubFigure%22%3B%20expected%20%22int%20%7C%20str%20%7C%20Figure%20%7C%20Literal%5B%27all%27%5D%20%7C%20None%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60pattern_lens%2Ffigure_util.py%23L185%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Ffigure_util.py%23L185%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09%23%20Always%20clean%20up%20figures%2C%20even%20if%20an%20error%20occurred%0A%09%09%09%09for%20fig%20in%20figs_dict.values%28%29%3A%0A%09%09%09%09%09%23%20TYPING%3A%20error%3A%20Argument%201%20to%20%22close%22%20has%20incompatible%20type%20%22Figure%20%7C%20SubFigure%22%3B%20expected%20%22int%20%7C%20str%20%7C%20Figure%20%7C%20Literal%5B%27all%27%5D%20%7C%20None%22%20%20%5Barg-type%5D%0A%09%09%09%09%09plt.close%28fig%29%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%60%60%60&labels=TYPING)

  ```python
  # Always clean up figures, even if an error occurred
  for fig in figs_dict.values():
  	# TYPING: error: Argument 1 to "close" has incompatible type "Figure | SubFigure"; expected "int | str | Figure | Literal['all'] | None"  [arg-type]
  	plt.close(fig)  # type: ignore[arg-type]
  ```




