# ParallelExpectationMaximizationClustering

High Performance Computing (HPC) project (University of Trento): parallel Expectation-Maximization (EM) clustering.

**Authors:** 
- Michael Bernasconi
- Gabriele Fronzoni

## Repository structure

```
.
├── src/
│   ├── PEMC-baseline-sequential.c    # Sequential version (correctness reference and serial time)
│   ├── PEMC-baseline-MPI-OpenMP.c    # Hybrid MPI + OpenMP baseline
│   ├── PEMC-opt1-MPI-OpenMP.c        # Optimization 1 of the baseline
│   ├── PEMC-opt2-MPI-OpenMP.c        # Optimization 2 of the baseline
│   └── PEMC-opt3-MPI-OpenMP.c        # Optimization 3 of the baseline
├── include/         # Header files
├── data/
│   ├── real/        # Real (medical) datasets
│   └── synthetic/   # Synthetically generated datasets
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
│   ├── project-guide.md  # Project requirements, report outline, evaluation criteria, data tips
│   ├── question.md       # Open questions for the instructors
│   └── todo.md           # Task list and workflow
├── makefile         # Build rules
└── README.md
```

## Documentation

- [doc/todo.md](doc/todo.md): task list and workflow
- [doc/question.md](doc/question.md): open questions for the instructors
- [doc/project-guide.md](doc/project-guide.md): project requirements, report outline, evaluation criteria, data tips
- [doc/em-explained.md](doc/em-explained.md): study guide on the EM algorithm (formulas, notation, data, parallelism)
- [doc/latex/theory/em-theory.tex](doc/latex/theory/em-theory.tex): LaTeX notes on the EM theory
- [doc/latex/Group6-Report.tex](doc/latex/Group6-Report.tex): IEEE-style report
- [Presentation](https://docs.google.com/presentation/d/1nnzwo61ZBHoQlmlYfGVbIb39rVTOexQlnH01uqBfev0/edit?slide=id.g4dfce81f19_0_45#slide=id.g4dfce81f19_0_45): project presentation slides (Google Slides)

## Build

TODO

## Usage

TODO

## Results

TODO
