import pytest

from ocbrain.cli import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])
    assert capsys.readouterr().out == "ocbrain 0.4.0\n"
