from tests.helpers import ROOT


def test_pythonhashseed_documented_in_readme() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "PYTHONHASHSEED=42" in readme, (
        "README deve documentar PYTHONHASHSEED=42 na se\u00e7\u00e3o de Reprodutibilidade"
    )


def test_hash_strategies_do_not_depend_on_python_hash() -> None:
    """
    Garante que nenhuma HashStrategy usa hash() nativo do Python
    (que depende de PYTHONHASHSEED e n\u00e3o \u00e9 est\u00e1vel entre processos).
    """
    import inspect
    from src.cache import hash_strategy as hs_module

    source = inspect.getsource(hs_module)
    # builtin hash() n\u00e3o deve aparecer fora de coment\u00e1rios/docstrings \u00f3bvios
    assert "= hash(" not in source.replace(" ", "")
    assert "return hash(" not in source.replace(" ", "")
