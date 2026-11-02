# ParallelExpectationMaximizationClustering

High Performance Computing (HPC) project (University of Trento): parallel Expectation-Maximization (EM) clustering.

**Authors:** 
- Michael Bernasconi
- Gabriele Fronzoni

## Repository structure

```
.
├── src/
│   ├── PEMC-baseline-sequential-diagonal.c    # Sequential version (correctness reference and serial time)
│   ├── PEMC-baseline-MPI-OpenMP-diagonal.c    # Hybrid MPI + OpenMP baseline
│   ├── PEMC-opt1-MPI-OpenMP-diagonal.c        # Optimization 1 of the baseline
│   ├── PEMC-opt2-MPI-OpenMP-diagonal.c        # Optimization 2 of the baseline
│   ├── PEMC-opt3-MPI-OpenMP-diagonal.c        # Optimization 3 of the baseline
│   ├── PEMC-baseline-MPI-OpenMP-full.c        # Baseline variant with full covariance (extension)
│   ├── generate-synthetic-dataset.py          # Generator of the synthetic datasets (3 profiles, one file per number of processes)
│   └── common/
│       └── my-time-lib.c                      # Shared code: statistics for the benchmarks (mean, median, standard deviation)
├── include/
│   └── my-time-lib.h                          # Timer macros and statistics prototypes (compile with -Iinclude and -lm)
├── data/            # NOT versioned (listed in .gitignore): it is filled locally
│   ├── real/        # Real (medical) datasets, downloaded locally
│   └── synthetic/   # Synthetic datasets, one folder per profile: created by running src/generate-synthetic-dataset.py
│       ├── synth_clean/     # K=4, D=10: files synth_clean_p01.bin ... p16.bin, labels, .json
│       ├── synth_overlap/   # K=5, D=20: overlapping clusters, correlated variables
│       └── synth_rare/      # K=6, D=50: rare group (1%), 2% outliers
├── results/
│   ├── logs/        # Raw execution logs
│   ├── tables/      # Benchmark tables
│   ├── plots/       # Generated plots
│   ├── plots-performance.py  # Performance plots (time, speedup, efficiency, scaling, ...)
│   └── plots-cluster.py      # Clustering plots (PCA, centroids, cluster sizes, ...)
├── doc/
│   ├── latex/
│   │   ├── Group6-Report.tex         # IEEE-style report
│   │   └── theory/em-theory.tex      # EM theory notes (Gaussian Mixture Model, algorithm, sequential cost)
│   ├── em-explained.md   # Study guide on the EM algorithm
│   ├── papers.md         # Free papers for the state of the art (theory, single node, multiple nodes, 5 candidate optimizations)
│   ├── papers/           # Local copies of the papers, numbered like papers.md (not versioned, listed in .gitignore)
│   ├── project-guide.md  # Project requirements, report outline, evaluation criteria, data tips
│   ├── synthetic-datasets.md  # Characteristics of the synthetic datasets (tables, sizes, format), ready for the report
│   ├── question.md       # Open questions for the instructors
│   └── todo.md           # Task list and workflow
├── makefile         # Build rules
└── README.md
```

The `-diagonal` files use a Gaussian mixture with diagonal covariance (the main line of the project: sequential reference, baseline and 3 optimizations). The `-full` file is a variant with full covariance: it changes the model, so its results are compared only with runs of the same model.

## Documentation

- [doc/todo.md](doc/todo.md): task list and workflow
- [doc/question.md](doc/question.md): open questions for the instructors
- [doc/project-guide.md](doc/project-guide.md): project requirements, report outline, evaluation criteria, data tips
- [doc/em-explained.md](doc/em-explained.md): study guide on the EM algorithm (formulas, notation, data, parallelism)
- [doc/synthetic-datasets.md](doc/synthetic-datasets.md): characteristics of the three synthetic datasets (profiles, variables, weak scaling files and sizes, file format), ready to be reused in the report
- [doc/papers.md](doc/papers.md): free papers for the state of the art (theory, parallel EM on one and several nodes, candidate optimizations); the PDFs are in `doc/papers/`, kept only locally
- [doc/latex/theory/em-theory.tex](doc/latex/theory/em-theory.tex): LaTeX notes on the EM theory
- [doc/latex/Group6-Report.tex](doc/latex/Group6-Report.tex): IEEE-style report
- [Presentation](https://docs.google.com/presentation/d/1nnzwo61ZBHoQlmlYfGVbIb39rVTOexQlnH01uqBfev0/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45): project presentation slides (Google Slides)

## Data (required step before running the experiments)

The whole `data/` folder is **not versioned** (it is listed in `.gitignore`), because the datasets are large. After cloning the repository the folder is empty, so **it is necessary to run the generator of the synthetic datasets** before any experiment:

```
python3 src/generate-synthetic-dataset.py
```

It needs Python 3 with NumPy and writes about 5 GB in `data/synthetic/`, one folder per profile (`synth_clean/`, `synth_overlap/`, `synth_rare/`), with one file for each number of processes of the weak scaling. The generation is deterministic (default seed 42), so everyone obtains identical files. Options (for example a smaller size) are listed with `python3 src/generate-synthetic-dataset.py --help`.

The real datasets also have to be downloaded locally into `data/real/`. See [doc/synthetic-datasets.md](doc/synthetic-datasets.md) for the characteristics of the synthetic datasets and the file format.

## Build

TODO

## Usage

TODO

## Results

TODO
