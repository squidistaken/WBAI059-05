# Natural Language Processing

This project tackles 4-way news topic classification on the [AGNews](https://huggingface.co/datasets/sh0416/ag_news) on the classes World, Sports, Business, Sci/Tech. It (will) contain the three deliverables required for Natural Language Processing (WBAI059-05).

## Team
* Teun Boersma (s5195179)
* Julian Sprietsma (s5096219)
* Marcus Harald Olof Persson (s5343798)

## Developing
This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

1. Clone the project.
2. Create a copy of [example.config.yaml](example.config.yaml) and rename it to `config.yaml`.
3. Sychronise the project.
```bash
uv sync
```
4. Run the project.
```bash
uv run main.py [--assignment] [--functionality]
```
 * `--assignment`: The assignment to run. Dependent on `functionality`.
 * `--functionality`: The functionality to run. Dependent on `asssignment`.

## Command Line Interface

This project uses a CLI built with [rich](https://rich.readthedocs.io/en/stable/introduction.html) for a clean interface and to provide a fresh-out-of-the-box project immediately after cloning. Using the CLI, you can interact with all the required deliverables per assignment efficiently.
