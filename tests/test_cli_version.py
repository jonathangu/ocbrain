import pytest

from ocbrain.cli import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert capsys.readouterr().out == "ocbrain 1.1.0\n"
