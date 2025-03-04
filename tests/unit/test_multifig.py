from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pattern_lens.figure_util import matplotlib_multifigure_saver

TEMP_DIR: Path = Path("tests/_temp")


def test_matplotlib_multifigure_saver():
	"""Test the matplotlib_multifigure_saver decorator."""
	# Create a temporary directory for saving figures
	temp_dir = TEMP_DIR / "test_matplotlib_multifigure_saver"
	temp_dir.mkdir(parents=True, exist_ok=True)

	# Define a test function with the decorator
	@matplotlib_multifigure_saver(names=["hist", "heatmap"])
	def multi_plot(attn_matrix, axes_dict):
		# Plot histogram
		axes_dict["hist"].hist(attn_matrix.flatten(), bins=30)
		axes_dict["hist"].set_title("Attention Values Histogram")

		# Plot heatmap
		im = axes_dict["heatmap"].matshow(attn_matrix, cmap="viridis")  # noqa: F841
		axes_dict["heatmap"].set_title("Attention Heatmap")

	# Generate a test attention matrix
	attn_matrix = np.random.rand(10, 10).astype(np.float32)

	# Call the decorated function
	multi_plot(attn_matrix, temp_dir)

	# Check that both figures were saved
	hist_file = temp_dir / "multi_plot.hist.svgz"
	heatmap_file = temp_dir / "multi_plot.heatmap.svgz"

	assert hist_file.exists(), "Histogram figure file was not saved"
	assert heatmap_file.exists(), "Heatmap figure file was not saved"

	# Check that figure_save_fmt attribute was set correctly
	assert hasattr(multi_plot, "figure_save_fmt")
	assert multi_plot.figure_save_fmt == "svgz"


def test_matplotlib_multifigure_saver_custom_format():
	"""Test the matplotlib_multifigure_saver decorator with a custom format."""
	# Create a temporary directory for saving figures
	temp_dir = TEMP_DIR / "test_matplotlib_multifigure_saver_custom_format"
	temp_dir.mkdir(parents=True, exist_ok=True)

	# Define a test function with the decorator and custom format
	@matplotlib_multifigure_saver(names=["plot1", "plot2"], fmt="png")
	def multi_plot_png(attn_matrix, axes_dict):
		# Simple plots for each axis
		axes_dict["plot1"].plot(attn_matrix.mean(axis=0))
		axes_dict["plot1"].set_title("Mean by Column")

		axes_dict["plot2"].plot(attn_matrix.mean(axis=1))
		axes_dict["plot2"].set_title("Mean by Row")

	# Generate a test attention matrix
	attn_matrix = np.random.rand(10, 10).astype(np.float32)

	# Call the decorated function
	multi_plot_png(attn_matrix, temp_dir)

	# Check that both figures were saved with the custom format
	plot1_file = temp_dir / "multi_plot_png.plot1.png"
	plot2_file = temp_dir / "multi_plot_png.plot2.png"

	assert plot1_file.exists(), "Plot1 figure file was not saved"
	assert plot2_file.exists(), "Plot2 figure file was not saved"

	# Check that figure_save_fmt attribute was set correctly
	assert hasattr(multi_plot_png, "figure_save_fmt")
	assert multi_plot_png.figure_save_fmt == "png"


def test_matplotlib_multifigure_saver_error_handling():
	"""Test error handling in the matplotlib_multifigure_saver decorator."""
	# Create a temporary directory for saving figures
	temp_dir = TEMP_DIR / "test_matplotlib_multifigure_saver_error_handling"
	temp_dir.mkdir(parents=True, exist_ok=True)

	# Define a test function that raises an error for one of the plots
	@matplotlib_multifigure_saver(names=["good_plot", "error_plot"])
	def plot_with_error(attn_matrix, axes_dict):
		# This plot should work fine
		axes_dict["good_plot"].matshow(attn_matrix)

		# This plot will raise an error
		axes_dict["error_plot"].plot(1 / 0)  # Division by zero

	# Generate a test attention matrix
	attn_matrix = np.random.rand(10, 10).astype(np.float32)

	# Call the decorated function and expect an error
	try:
		plot_with_error(attn_matrix, temp_dir)
		raise AssertionError("Expected ZeroDivisionError but no exception was raised")
	except ZeroDivisionError:
		# Check that the first figure was saved before the error
		good_plot_file = temp_dir / "plot_with_error.good_plot.svgz"
		assert not good_plot_file.exists(), "Good plot file was saved before the error"


def test_matplotlib_multifigure_saver_cleanup():
	"""Test that matplotlib_multifigure_saver properly closes figures."""
	# Create a temporary directory for saving figures
	temp_dir = TEMP_DIR / "test_matplotlib_multifigure_saver_cleanup"
	temp_dir.mkdir(parents=True, exist_ok=True)

	# Count the number of open figures before the test
	initial_figures = len(plt.get_fignums())

	# Define a test function with the decorator
	@matplotlib_multifigure_saver(names=["plot1", "plot2"])
	def multi_plot_cleanup(attn_matrix, axes_dict):
		axes_dict["plot1"].plot(attn_matrix[0])
		axes_dict["plot2"].plot(attn_matrix[1])

	# Generate a test attention matrix
	attn_matrix = np.random.rand(10, 10).astype(np.float32)

	# Call the decorated function
	multi_plot_cleanup(attn_matrix, temp_dir)

	# Check that no new figures remain open
	assert len(plt.get_fignums()) == initial_figures, "Figures were not properly closed"
