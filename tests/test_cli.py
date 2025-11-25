"""
Tests for the Command-Line Interface (cli/cli.py).
"""

from click.testing import CliRunner
from PIL import Image
from cli.cli import cli

def test_predict_command(tmp_path):
    """
    Test the 'predict' command.
    
    It checks if the command runs successfully and prints the expected
    "Prediction:" output.
    """
    fake_image_path = tmp_path / "fake_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(fake_image_path)

    runner = CliRunner()
    result = runner.invoke(cli, [
        'predict',
        str(fake_image_path)
    ])

    assert result.exit_code == 0
    assert "Prediction:" in result.output

def test_resize_command(tmp_path):
    """
    Test the 'resize' command.

    It checks if the command runs, creates an output file, and
    verifies that the new file has the correct dimensions.
    """
    fake_image_path = tmp_path / "input.jpg"
    img = Image.new('RGB', (200, 200), color='blue')
    img.save(fake_image_path)

    output_image_path = tmp_path / "output.png"
    target_width = 50
    target_height = 50

    runner = CliRunner()
    result = runner.invoke(cli, [
        'resize',
        str(fake_image_path),
        '--width', str(target_width),
        '--height', str(target_height),
        str(output_image_path)
    ])

    assert result.exit_code == 0
    assert "Image resized and saved to:" in result.output

    assert output_image_path.exists()

    with Image.open(output_image_path) as resized_img:
        assert resized_img.size == (target_width, target_height)

def test_grayscale_command(tmp_path):
    """
    Test the 'grayscale' command.
    """
    fake_image_path = tmp_path / "input_color.jpg"
    img = Image.new('RGB', (200, 200), color='blue')
    img.save(fake_image_path)

    output_image_path = tmp_path / "output_gray.png"

    runner = CliRunner()
    result = runner.invoke(cli, [
        'grayscale',
        str(fake_image_path),
        str(output_image_path)
    ])

    assert result.exit_code == 0
    assert "Grayscale image saved to:" in result.output
    assert output_image_path.exists()

    with Image.open(output_image_path) as gray_img:
        assert gray_img.mode == 'L'

def test_cli_help():
    """Test that the main --help command works."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])

    assert result.exit_code == 0
    assert "Usage: cli [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "predict" in result.output
    assert "resize" in result.output
    assert "grayscale" in result.output
