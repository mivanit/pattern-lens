 # Inline TODOs


# TODO

## [`pattern_lens/activations.py`](/pattern_lens/activations.py)

- batching?  
  local link: [`/pattern_lens/activations.py#203`](/pattern_lens/activations.py#203) 
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
  local link: [`/pattern_lens/activations.py#350`](/pattern_lens/activations.py#350) 
  | view on GitHub: [pattern_lens/activations.py#L350](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/activations.py#L350)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=this%20basically%20does%20nothing%2C%20since%20we%20load%20the%20activations%20and%20then%20immediately%20get%20rid%20of%20them.&body=%23%20source%0A%0A%5B%60pattern_lens%2Factivations.py%23L350%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Factivations.py%23L350%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09return%20path%2C%20cache%0A%09%09%09else%3A%0A%09%09%09%09%23%20TODO%3A%20this%20basically%20does%20nothing%2C%20since%20we%20load%20the%20activations%20and%20then%20immediately%20get%20rid%20of%20them.%0A%09%09%09%09%23%20maybe%20refactor%20this%20so%20that%20load_activations%20can%20take%20a%20parameter%20to%20simply%20assert%20that%20the%20cache%20exists%3F%0A%09%09%09%09%23%20this%20will%20let%20us%20avoid%20loading%20it%2C%20which%20slows%20things%20down%0A%60%60%60&labels=enhancement)

  ```python
return path, cache
			else:
				# TODO: this basically does nothing, since we load the activations and then immediately get rid of them.
				# maybe refactor this so that load_activations can take a parameter to simply assert that the cache exists?
				# this will let us avoid loading it, which slows things down
  ```


- not implemented yet  
  local link: [`/pattern_lens/activations.py#474`](/pattern_lens/activations.py#474) 
  | view on GitHub: [pattern_lens/activations.py#L474](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/activations.py#L474)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=not%20implemented%20yet&body=%23%20source%0A%0A%5B%60pattern_lens%2Factivations.py%23L474%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Factivations.py%23L474%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09write_html_index%28save_path_p%29%0A%0A%09%23%20TODO%3A%20not%20implemented%20yet%0A%09if%20stacked_heads%3A%0A%09%09raise%20NotImplementedError%28%22stacked_heads%20not%20implemented%20yet%22%29%0A%60%60%60&labels=enhancement)

  ```python
write_html_index(save_path_p)

	# TODO: not implemented yet
	if stacked_heads:
		raise NotImplementedError("stacked_heads not implemented yet")
  ```




## [`pattern_lens/figures.py`](/pattern_lens/figures.py)

-   
  local link: [`/pattern_lens/figures.py#136`](/pattern_lens/figures.py#136) 
  | view on GitHub: [pattern_lens/figures.py#L136](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/figures.py#L136)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=Issue%20from%20inline%20todo&body=%23%20source%0A%0A%5B%60pattern_lens%2Ffigures.py%23L136%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Ffigures.py%23L136%29%0A%0A%23%20context%0A%60%60%60python%0A%09%20%20%20%28defaults%20to%20%60False%60%29%0A%09%20-%20%60track_results%20%3A%20bool%60%0A%09%20%20%20%20whether%20to%20track%20the%20results%20of%20each%20function%20for%20each%20head.%20Isn%27t%20used%20for%20anything%20yet%2C%20but%20this%20is%20a%20TODO%0A%09%20%20%20%28defaults%20to%20%60False%60%29%0A%09%22%22%22%0A%60%60%60&labels=enhancement)

  ```python
(defaults to `False`)
	 - `track_results : bool`
	    whether to track the results of each function for each head. Isn't used for anything yet, but this is a TODO
	   (defaults to `False`)
	"""
  ```


- do something with results  
  local link: [`/pattern_lens/figures.py#178`](/pattern_lens/figures.py#178) 
  | view on GitHub: [pattern_lens/figures.py#L178](https://github.com/mivanit/pattern-lens/blob/main/pattern_lens/figures.py#L178)
  | [Make Issue](https://github.com/mivanit/pattern-lens/issues/new?title=do%20something%20with%20results&body=%23%20source%0A%0A%5B%60pattern_lens%2Ffigures.py%23L178%60%5D%28https%3A%2F%2Fgithub.com%2Fmivanit%2Fpattern-lens%2Fblob%2Fmain%2Fpattern_lens%2Ffigures.py%23L178%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09results%5Bfunc_name%5D%5B%28layer_idx%2C%20head_idx%29%5D%20%3D%20status%0A%0A%09%23%20TODO%3A%20do%20something%20with%20results%0A%0A%09generate_prompts_jsonl%28save_path%20%2F%20model_cfg.model_name%29%0A%60%60%60&labels=enhancement)

  ```python
results[func_name][(layer_idx, head_idx)] = status

	# TODO: do something with results

	generate_prompts_jsonl(save_path / model_cfg.model_name)
  ```




