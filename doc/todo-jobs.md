# Jobs on the cluster: checklist

Submission commands of [exec-cluster.md](exec-cluster.md#phase-4-experiments-pbs), one stage at a time. Tick a command (`[X]`) when it has been submitted.

---

## Batch A: `pack`, `scatter`

```bash
export PLACES_LIST="pack scatter"
```

- [X] `src/script/pemc-submit.sh diagonal sequential` (1 job)
- [X] `src/script/pemc-submit.sh diagonal baseline` (8 jobs)
- [X] `src/script/pemc-submit.sh diagonal opt1` (8 jobs)
- [ ] `src/script/pemc-submit.sh diagonal opt2` (8 jobs)
- [X] `src/script/pemc-submit.sh full sequential` (1 job)
- [X] `src/script/pemc-submit.sh full baseline` (8 jobs)
- [ ] `src/script/pemc-submit.sh full opt1` (8 jobs, when opt1 full is written)
- [ ] `src/script/pemc-submit.sh full opt2` (8 jobs, when opt2 full is written)

---

## Batch B: `pack:excl`, `scatter:excl` (no sequential: it is already exclusive)

```bash
export PLACES_LIST="pack:excl scatter:excl"
```

- [X] `src/script/pemc-submit.sh diagonal baseline` (8 jobs)
- [X] `src/script/pemc-submit.sh diagonal opt1` (8 jobs)
- [ ] `src/script/pemc-submit.sh diagonal opt2` (8 jobs)
- [X] `src/script/pemc-submit.sh full baseline` (8 jobs)
- [ ] `src/script/pemc-submit.sh full opt1` (8 jobs, when opt1 full is written)
- [ ] `src/script/pemc-submit.sh full opt2` (8 jobs, when opt2 full is written)
