# Natural Language Processing

This project tackles 4-way news topic classification on the [AGNews](https://huggingface.co/datasets/sh0416/ag_news) on the classes World, Sports, Business, Sci/Tech. It contains the three deliverables required for Natural Language Processing (WBAI059-05).

## Team
* Teun Boersma (s5195179)
* Julian Sprietsma (s5096219)
* Marcus Harald Olof Persson (s5343798)

## Developing
This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

1. Clone the project.
2. Create a copy of [example.config.yaml](example.config.yaml) and rename it to `config.yaml`.
   * `hf_token` is not required.
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

## Deliverables

You can run the project as is by following the above instructions. Below, you can run specific deliverables for quick reference.

### Assignment 1
 * **Train and Evaluate Base Models:** 

```bash
uv run main.py --assignment 1 --functionality 1
```
 * **Perform SVM Grid Search:**

```bash
uv run main.py --assignment 1 --functionality 2
```
 * **Analyze Errors on Models:**

```bash
uv run main.py --assignment 1 --functionality 3
```

### Assignment 2
 * **Examine Word Similarity:** 

```bash
uv run main.py --assignment 2 --functionality 1
```
 * **Train and Evaluate CNN Model:**

```bash
uv run main.py --assignment 2 --functionality 2
```
 * **Train and Evaluate LSTM Model:**
 
```bash
uv run main.py --assignment 2 --functionality 3
```

 * **Analyze Errors:**

```bash
uv run main.py --assignment 2 --functionality 4
```

 * **Ablation Study on Sequence Length:**

```bash
uv run main.py --assignment 2 --functionality 5
```

### Assignment 3

 * **Finetune and Evaluate DistilBERT:**

```bash
uv run main.py --assignment 3 --functionality 1
```

 * **Robustness Evaluation:**

```bash
uv run main.py --assignment 3 --functionality 2
```

 * **Analyze Errors:**

```bash
uv run main.py --assignment 3 --functionality 3
```
