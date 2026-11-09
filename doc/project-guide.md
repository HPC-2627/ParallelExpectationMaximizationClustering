# Project Guide
Everything needed for the project, based on the course slides (project template, evaluation criteria, tips).

## 1. Hard requirements
- [X] Parallel application using **MPI and OpenMP** (hybrid recommended)
- [X] Developed **in team**; **all members present** at the oral exam and presentation.
- [X] Implementation in **C** (C++ allowed), **properly commented**.
- [X] **Link to the repo** in the report: https://github.com/HPC-2627/ParallelExpectationMaximizationClustering
- [ ] Report **>= 10 pages**.
- [ ] **AI statement** in the report (mandatory).
- [ ] Cite papers/material that inspired the work (state of the art is part of the report).
- [ ] Cite the data sources.

## 2. Report structure (project template)

### 2.1 Introduction
- Accurate description of the assigned problem (EM clustering).
- Analysis of the **sequential algorithm** (complexity, cost per phase), for both models (diagonal and full covariance, 4 versions each: sequential, baseline, 2 optimizations).
- Pseudo-code, examples, graphs, figures, application instances.

### 2.2 Parallel design
- Preliminary study of the opportunities for parallelism in the sequential algorithm.
- **State of the art** on parallel design strategies.
- **Alternative designs** (different parallelization strategies), discussed and motivated:
  - why some operations parallelize well and others do not;
  - why some data structures minimize communication/synchronization burden and others do not.
- Reference to prior work and link with related work, stated clearly.
- **Hybrid parallelization** and **data dependencies** must be discussed.

### 2.3 Implementation
- C implementation, code snippets of the most critical parts.
- If multiple strategies: several implementations, with pros and cons of each.
- Link to the repo.

### 2.4 Performance and scalability analysis
- Execution time, **speedup**, **efficiency**.
- **Strong scaling** and **weak scaling** where possible.
- **Single-node and multi-node** benchmarks are expected.
- Critical discussion of the results.

## 3. How it will be evaluated
- Clarity and effectiveness of presentation.
- Depth, correctness and originality of the theoretical analysis.
- Technical skills and documentation of the implementation.
- Number and quality of multiple parallel strategies.
- Critical thinking in performance evaluation.
- **Significance of results**: a good project must show a significant speedup.

## 4. Data tips from the slides
- Hard to find input data? Write a simple **random generator** for synthetic data.
- Use **small, medium and large** datasets and compare performance on the three.
- If replicating an existing work:
  - reproduce and compare results, comment differences;
  - start from the paper pseudocode, write your own C implementation, compare with the authors' findings;
  - or take existing code and add your own delta, then compare;
  - or port two existing implementations on the cluster and compare them (plus the original work).

## 5. Final checklist before submission
- [ ] Report >= 10 pages, IEEE format ([latex/Group6-Report.tex](latex/Group6-Report.tex)).
- [ ] AI statement included.
- [ ] Repo link and data sources cited.
- [ ] Code commented, makefile builds from scratch, README has build/usage instructions.
- [ ] Strong + weak scaling, single-node + multi-node results.
- [ ] Both team members ready for the oral exam.
