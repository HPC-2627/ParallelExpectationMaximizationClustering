# Color palette for plots
## Versions (lines: speedup, scaling, time per iteration, comparison with the baseline)
```
Version     Hex      Name        Marker
----------  -------  ----------  ------
sequential  #6B7280  gray        s
baseline    #2579ff  vivid blue  o
opt1        #ff2e2e  red         ^
opt2        #38ffca  aquamarine  D
```
The model is never a color: diagonal and full are two panels (a)/(b). If they must share a panel, diagonal = solid line, full = dashed line.

## Placement (bars of P6, and the 2x2 grid)
Hue = pack (blue) / scatter (orange), lightness = shared (light) / excl (dark). The excl bars also get a hatch (`//`). Bars have a thin dark edge.
```
Placement     Hex      Name
------------  -------  ---------------
pack          #627fff  periwinkle blue
pack:excl     #0600be  dark indigo
scatter       #ffab76  light orange
scatter:excl  #af3200  dark rust
```

## Time breakdown (P8, stacked bars)
```
Component                           Hex      Name
----------------------------------  -------  -------------
E-step                              #6fff2c  mint green
M-step                              #3415ff  indigo
t_comm (communication + imbalance)  #ff0000  coral red
I/O (read)                          #efc000  golden yellow
```

## Clusters (C3, C4, C5): categorical, up to 12 classes
In this order. Every class also has its own marker in the PCA plot. 
Classes 10-12 are only needed by the datasets with more than 9 classes (brainweb, K = 12).
```
Class  Hex      Name          Marker
-----  -------  ------------  ------
1      #ff0000  red           o
2      #22C55E  green         s
3      #69d2ff  sky blue      ^
4      #dca500  amber         D
5      #912603  rust brown    v
6      #006326  dark green    P
7      #002fb0  navy          X
8      #7904d3  dark purple   <
9      #502B00  dark brown    >
10     #e44aff  neon magenta  h
11     #ff8d35  orange        *
12     #2c79ff  bright blue   p
```

## Datasets and models in the clustering plots

- **C1 (one line per dataset):** the first colors and markers of the cluster list above, in order. 
  In C1 there are no cluster classes, so there is no ambiguity.
- **C2 (BIC vs AIC):** BIC = black solid line with circles, AIC = gray dashed line with squares (neutrals, no new color). 
  The minimum is a star, the true K a dotted gray line.
- **C6 (BIC full - diagonal):** light gray fill with a dark edge. 
  The sign is read from the axis, not from the hue.
- **C3 (PCA):** points colored with the cluster list. 
  The outliers of the synthetic data (label -1) are slate gray with an `x` marker.

## Neutrals
```
Use                                                 Hex      Name
--------------------------------------------------  -------  ----------
text, axes                                          #171e26  black
ideal line (speedup, efficiency, roofline), dashed  #6B7280  gray
grid                                                #E5E7EB  light gray
missing / not available                             #656B75  slate gray
```

## Colormaps (heatmaps)
Continuous scales: they are not part of the hex list above.
| Plot | Colormap | Why |
|---|---|---|
| P7 (time, ranks x threads) | `viridis` | sequential, perceptually uniform, readable in grayscale |
| C4 (standardized centroids, have a sign) | `RdBu_r`, limits symmetric around 0 | diverging: 0 must be the neutral color |
