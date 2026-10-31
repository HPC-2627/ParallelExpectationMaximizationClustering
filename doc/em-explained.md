# EM clustering spiegato da zero

Guida di studio, senza prerequisiti di machine learning. Le formule si leggono bene con l'anteprima Markdown di VS Code.

## 1. L'idea in breve

Hai una tabella di dati (per esempio pazienti, ognuno descritto da età, pressione, glicemia...). **Nessuno ti ha detto a quale gruppo appartiene ogni paziente**, ma sospetti che esistano dei gruppi.

L'ipotesi dell'EM: ogni gruppo ha la "forma a campana" (una **gaussiana**) intorno a un centro, e i dati sono una **mescolanza** di queste campane.

Il problema è un cane che si morde la coda:
- se conoscessi i **gruppi** (centro, forma, peso), potresti calcolare per ogni paziente **quanto è probabile che appartenga a ciascuno**;
- se conoscessi l'**appartenenza** di ogni paziente, potresti calcolare **centro e forma** di ogni gruppo.

Non conosci né l'uno né l'altro. L'EM parte da una stima a caso e alterna due passi finché non cambia più:

1. **E-step (Expectation):** con i gruppi attuali, calcola per ogni punto la probabilità di appartenere a ciascun gruppo.
2. **M-step (Maximization):** con quelle probabilità, ricalcola centro, forma e peso di ogni gruppo.

È un k-means "morbido": ogni punto ha una **probabilità** per ogni gruppo ("70% A, 25% B, 5% C") invece di un'etichetta secca.

## 2. Che dati servono

Una **matrice** $X$ di numeri reali, con $N$ righe e $D$ colonne:

```
              età   BMI   pressione  glicemia   ...   (D colonne = variabili)
paziente 1    54    27.1    130       95
paziente 2    61    31.4    145      130
paziente 3    33    22.0    115       88
...
paziente N    ...                                 (N righe = punti)
```

- Ogni **riga** è un punto (un paziente) e ogni **colonna** è una variabile.
- Serve solo dati **numerici**. Una variabile categorica (sesso, fumatore sì/no) va trasformata in numeri (0/1 o one-hot).
- **Non servono etichette**: è apprendimento non supervisionato. Se il dataset ha una diagnosi, la usi solo alla fine per valutare, non per allenare.
- **Normalizzazione (z-score):** si sottrae la media e si divide per la deviazione standard di ogni colonna. Senza, una variabile con numeri grandi (colesterolo in mg/dL) pesa più di una con numeri piccoli (BMI) solo per la scala.
- **Valori mancanti:** vanno gestiti prima (rimuovere la riga o sostituire con la media), perché l'algoritmo non li accetta.
- **Quello che scegli tu:** il numero di gruppi $K$.

## 3. Tutte le lettere

| Simbolo | Cosa è | Dimensione |
|---|---|---|
| $N$ | numero di punti (pazienti) | numero |
| $D$ | numero di variabili (colonne) | numero |
| $K$ | numero di gruppi (cluster) che cerchiamo | numero |
| $i = 1 \dots N$ | indice del punto | |
| $k = 1 \dots K$ | indice del gruppo | |
| $d = 1 \dots D$ | indice della variabile | |
| $x_i$ | il punto $i$, cioè una riga di $X$ | vettore di $D$ numeri |
| $\mu_k$ | **media** (centro) del gruppo $k$ | vettore di $D$ numeri |
| $\Sigma_k$ | **matrice di covarianza** del gruppo $k$ (forma) | matrice $D \times D$ |
| $\sigma^2_{kd}$ | varianza della variabile $d$ nel gruppo $k$ (versione diagonale) | numero |
| $\pi_k$ | **peso** del gruppo $k$: quota di punti che gli appartiene | numero tra 0 e 1, somma 1 |
| $\theta$ | tutti i parametri insieme: $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^{K}$ | |
| $z_i$ | gruppo "vero" (nascosto) del punto $i$: la **variabile latente** | uno tra $1 \dots K$ |
| $r_{ik}$ | **responsabilità**: probabilità che il punto $i$ appartenga al gruppo $k$ | numero tra 0 e 1, per ogni $i$ somma 1 |
| $N_k$ | "numero effettivo" di punti del gruppo $k$ | numero |
| $\mathcal{N}(x \mid \mu, \Sigma)$ | densità della gaussiana | numero $\ge 0$ |
| $\mathcal{L}$ | **log-verosimiglianza**: quanto bene i parametri spiegano i dati | numero |

## 4. Concetti di statistica necessari

### Media
Il valore centrale. Per la variabile $d$: $\bar{x}_d = \frac{1}{N}\sum_{i=1}^{N} x_{id}$.

### Varianza e deviazione standard
Misura **quanto i valori si disperdono** attorno alla media:
$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2, \qquad \sigma = \sqrt{\sigma^2}$$
Varianza piccola = valori vicini alla media (campana stretta). Grande = molto dispersi (campana larga).

### Covarianza (la domanda: cosa fa?)
La varianza guarda **una** variabile. La covarianza guarda **due** variabili insieme e dice se **crescono insieme**:
$$\mathrm{cov}(a, b) = \frac{1}{N}\sum_{i=1}^{N}(a_i - \bar{a})(b_i - \bar{b})$$

- **Positiva:** quando $a$ è sopra la media, anche $b$ tende a esserlo (età e pressione).
- **Negativa:** quando una sale, l'altra scende.
- **Vicina a zero:** nessuna relazione lineare.

Il valore dipende dalle unità di misura, per questo si usa spesso la **correlazione**, cioè la covarianza divisa per le due deviazioni standard (sempre tra -1 e +1).

### Matrice di covarianza $\Sigma$ ($D \times D$)
Raccoglie **tutte** le varianze e covarianze. Per due variabili, età e BMI:

$$\Sigma = \begin{pmatrix} \mathrm{var}(\text{età}) & \mathrm{cov}(\text{età},\text{BMI}) \\ \mathrm{cov}(\text{BMI},\text{età}) & \mathrm{var}(\text{BMI}) \end{pmatrix}$$

- **Diagonale:** le varianze di ciascuna variabile.
- **Fuori diagonale:** le covarianze tra coppie. È **simmetrica**.
- **Cosa fa nella gaussiana:** decide la **forma e l'orientamento** della campana. Con covarianza zero la campana è un cerchio (o un'ellisse allineata agli assi). Con covarianza positiva è un'ellisse **inclinata**, perché le due variabili vanno insieme.

### Covarianza piena e covarianza diagonale
| | Piena | Diagonale |
|---|---|---|
| Matrice | $D \times D$ completa | solo la diagonale (vettore di $D$ varianze) |
| Forma del gruppo | ellisse inclinata | ellisse allineata agli assi |
| Parametri per gruppo | circa $D^2/2$ | $D$ |
| Costo per iterazione | $O(NKD^2)$ + inversioni $O(KD^3)$ | $O(NKD)$ |
| Rischio | instabile con $D$ alto e pochi dati | assume variabili indipendenti dentro il gruppo |

**Per il progetto conviene partire dalla diagonale**: formule più semplici e senza inverse di matrici.

### La gaussiana
**In una dimensione**, la campana con media $\mu$ e varianza $\sigma^2$:
$$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$
- $(x-\mu)^2$: distanza al quadrato dal centro. Più il punto è lontano, più l'esponente è negativo e il valore scende.
- $\sigma^2$ a denominatore: se la campana è larga, la distanza "pesa meno".
- Il fattore davanti serve solo perché l'area totale sia 1.

**In $D$ dimensioni**, con matrice di covarianza $\Sigma$:
$$\mathcal{N}(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{D/2}\sqrt{\det\Sigma}} \exp\left(-\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu)\right)$$
- $(x-\mu)^T \Sigma^{-1} (x-\mu)$ è la **distanza di Mahalanobis** al quadrato: una distanza dal centro che tiene conto della forma del gruppo (lontano lungo l'asse corto conta di più che lungo l'asse lungo).
- $\det\Sigma$ è il "volume" del gruppo e $\Sigma^{-1}$ l'inversa: sono le parti costose da calcolare.

**Caso diagonale:** la gaussiana diventa il prodotto di $D$ gaussiane 1D, una per variabile, e il logaritmo è una semplice somma:
$$\log\mathcal{N}(x \mid \mu_k, \sigma_k^2) = -\frac{1}{2}\sum_{d=1}^{D}\left[\log(2\pi\sigma_{kd}^2) + \frac{(x_d-\mu_{kd})^2}{\sigma_{kd}^2}\right]$$

## 5. Il modello: mescolanza di gaussiane (GMM)

La densità di un punto qualsiasi è una **somma pesata** di $K$ gaussiane:
$$p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \sum_{k}\pi_k = 1$$

La "storia" con cui si immagina che i dati siano stati generati:
1. Si sceglie un gruppo $z$ a caso: il gruppo $k$ con probabilità $\pi_k$ (il peso).
2. Si estrae il punto da una gaussiana con media $\mu_k$ e covarianza $\Sigma_k$.

Noi vediamo solo il punto $x_i$, mai il gruppo $z_i$ (ecco perché è una **variabile latente**, cioè nascosta). L'EM serve a stimare $\theta$ senza conoscere $z_i$.

## 6. Cosa vogliamo massimizzare: la verosimiglianza

Vogliamo i parametri $\theta$ per cui i dati osservati sono **più probabili**. Si usa il logaritmo, perché trasforma i prodotti in somme ed evita numeri minuscoli:
$$\mathcal{L}(\theta) = \sum_{i=1}^{N}\log\sum_{k=1}^{K}\pi_k\,\mathcal{N}(x_i \mid \mu_k, \Sigma_k)$$

- Più è alta (meno negativa), meglio il modello spiega i dati.
- Non esiste una formula chiusa per i parametri migliori, perché c'è una somma dentro il logaritmo. Da qui l'EM.
- **Garanzia:** a ogni iterazione $\mathcal{L}$ non diminuisce, quindi si converge a un massimo **locale** (non necessariamente il migliore).

## 7. L'algoritmo, passo per passo

### Inizializzazione
Servono valori di partenza per $\pi_k$, $\mu_k$, $\Sigma_k$ (vedi sezione 9).

### E-step: calcolo delle responsabilità
$$r_{ik} = \frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K}\pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$

Si legge così: "la probabilità che il punto $i$ appartenga al gruppo $k$ è proporzionale a **quanto pesa il gruppo** ($\pi_k$) per **quanto bene la sua campana spiega il punto** ($\mathcal{N}$)". Il denominatore normalizza, così per ogni punto le $K$ probabilità sommano 1. È il teorema di Bayes.

### M-step: aggiornamento dei parametri
Ogni formula è la solita media o varianza, con i punti **pesati** da $r_{ik}$ invece di contati interi.

| Parametro | Formula | In parole |
|---|---|---|
| Punti effettivi | $N_k = \sum_{i=1}^{N} r_{ik}$ | somma delle probabilità: "quanti" punti ha il gruppo (anche decimale) |
| Peso | $\pi_k = \dfrac{N_k}{N}$ | quota di punti del gruppo |
| Media | $\mu_k = \dfrac{1}{N_k}\sum_{i} r_{ik}\,x_i$ | media dei punti, pesata dalla probabilità di appartenere al gruppo |
| Covarianza piena | $\Sigma_k = \dfrac{1}{N_k}\sum_{i} r_{ik}\,(x_i-\mu_k)(x_i-\mu_k)^T$ | covarianza pesata |
| Varianza (diagonale) | $\sigma_{kd}^2 = \dfrac{1}{N_k}\sum_{i} r_{ik}\,(x_{id}-\mu_{kd})^2$ | varianza pesata, variabile per variabile |

Un punto con $r_{ik}=0.9$ conta "quasi intero" nel gruppo $k$. Uno con $r_{ik}=0.01$ quasi non conta.

### Controllo di convergenza
Si calcola $\mathcal{L}$. Se $|\mathcal{L}_{\text{nuovo}}-\mathcal{L}_{\text{vecchio}}|$ è sotto una tolleranza (per esempio $10^{-6}$ relativa), oppure si raggiunge il numero massimo di iterazioni, ci si ferma. Altrimenti si torna all'E-step.

## 8. Esempio numerico a mano (1 dimensione)

Dati: $x = [1,\ 2,\ 5,\ 8,\ 9]$, quindi $N=5$, $D=1$ (la covarianza diventa una semplice varianza), $K=2$.
Partenza: $\mu_1=0$, $\mu_2=10$, $\sigma_1^2=\sigma_2^2=4$, $\pi_1=\pi_2=0.5$.

**E-step.** Con varianze e pesi uguali, la probabilità dipende solo da $\exp(-(x-\mu)^2/8)$:

| $x$ | gruppo 1 (centro 0) | gruppo 2 (centro 10) | $r_{i1}$ | $r_{i2}$ |
|---|---|---|---|---|
| 1 | $e^{-1/8}=0.8825$ | $e^{-81/8}=0.00004$ | 0.99995 | 0.00005 |
| 2 | $e^{-4/8}=0.6065$ | $e^{-64/8}=0.00034$ | 0.99945 | 0.00055 |
| 5 | $e^{-25/8}=0.0439$ | $e^{-25/8}=0.0439$ | **0.5** | **0.5** |
| 8 | 0.00034 | 0.6065 | 0.00055 | 0.99945 |
| 9 | 0.00004 | 0.8825 | 0.00005 | 0.99995 |

Il punto in mezzo ($x=5$) è esattamente a metà: l'EM gli dà il 50% e il 50%. Un k-means avrebbe dovuto scegliere.

**M-step per il gruppo 1.**
- $N_1 = 0.99995 + 0.99945 + 0.5 + 0.00055 + 0.00005 = 2.5$, quindi $\pi_1 = 2.5/5 = 0.5$.
- $\mu_1 = \dfrac{1\cdot0.99995 + 2\cdot0.99945 + 5\cdot0.5 + 8\cdot0.00055 + 9\cdot0.00005}{2.5} = \dfrac{5.5037}{2.5} \approx 2.20$
- $\sigma_1^2 = \dfrac{\sum_i r_{i1}(x_i-2.20)^2}{2.5} = \dfrac{5.42}{2.5} \approx 2.17$

Per simmetria, $\mu_2 \approx 7.80$ e $\sigma_2^2 \approx 2.17$.

**Risultato dopo un'iterazione:** i centri si sono avvicinati ai dati (da 0 e 10 a 2.2 e 7.8) e le campane si sono ristrette (varianza da 4 a 2.17). Dopo qualche iterazione i centri si fermano vicino a 1.5 e 8.5. È tutto qui.

## 9. Dettagli pratici

- **Inizializzazione:** sceglie da dove parte l'algoritmo. Opzioni: $K$ punti casuali come medie; oppure **k-means++** (più robusto). Varianze iniziali: la varianza globale dei dati. Pesi: $1/K$. Con un seed fisso i risultati sono riproducibili, e ti serve per confrontare le versioni.
- **Minimi locali:** partenze diverse possono dare risultati diversi. Si può provare più inizializzazioni e tenere la migliore.
- **Stabilità numerica (log-sum-exp):** le probabilità $\pi_k\mathcal{N}$ possono essere numeri minuscoli (underflow). Si calcolano in scala logaritmica e si normalizza sottraendo il massimo:
  $\log r_{ik} = \ell_{ik} - \left(m_i + \log\sum_j e^{\ell_{ij}-m_i}\right)$, con $\ell_{ik}=\log(\pi_k\mathcal{N}_{ik})$ e $m_i=\max_k \ell_{ik}$.
- **Regolarizzazione:** una varianza che tende a 0 fa esplodere il calcolo (un gruppo che "si aggancia" a un punto solo). Si aggiunge un piccolo $\epsilon$ (per esempio $10^{-6}$): $\sigma^2 + \epsilon$ oppure $\Sigma + \epsilon I$.
- **Scelta di $K$:** si prova più valori e si guardano **BIC** o **AIC**, che premiano l'accordo coi dati e penalizzano i modelli troppo complessi. Si sceglie il valore con BIC più basso.
- **Gruppo vuoto:** se $N_k \approx 0$, il gruppo è morto. Va gestito (reinizializzarlo o ridurre $K$).

## 10. Pseudocodice e strutture dati in C

```
leggi X (N x D), normalizza
inizializza pi[K], mu[K][D], var[K][D]            // diagonale
ripeti fino a convergenza:
    // E-step
    per ogni punto i:
        per ogni gruppo k:
            logp[k] = log(pi[k]) + log-gaussiana(x[i], mu[k], var[k])
        r[i][k] = normalizza con log-sum-exp
        ll += log-sum-exp(logp)                    // contributo alla log-verosimiglianza
    // M-step
    per ogni gruppo k:
        Nk = somma_i r[i][k]
        mu[k] = (somma_i r[i][k] * x[i]) / Nk
        var[k] = (somma_i r[i][k] * x[i]^2) / Nk - mu[k]^2  (+ epsilon)
        pi[k] = Nk / N
    controlla |ll - ll_prec| < tolleranza
```

Array da allocare (tutti di `double`):

| Array | Dimensione | Note |
|---|---|---|
| `X` | $N \times D$ | i dati, il più grosso |
| `r` | $N \times K$ | responsabilità (si può evitare fondendo E-step e M-step) |
| `mu` | $K \times D$ | medie |
| `var` | $K \times D$ (diagonale) oppure `Sigma` $K \times D \times D$ (piena) | |
| `pi` | $K$ | pesi |
| `Nk`, somme parziali | $K$, $K \times D$ | statistiche dell'M-step |

Memoria dei dati: $N \cdot D \cdot 8$ byte, per esempio 1 milione di punti con 10 variabili sono 80 MB.

**Costo per iterazione:** E-step $O(NKD)$ (diagonale) e M-step $O(NKD)$. Con covarianza piena $O(NKD^2)$ più $O(KD^3)$ per le inverse.

## 11. Dove sta il parallelismo

- **E-step:** ogni punto è **indipendente** dagli altri, quindi si divide $X$ a blocchi di righe tra i processi MPI e, dentro, tra i thread OpenMP. Nessuna comunicazione.
- **M-step:** sono **somme sui punti**. Ogni processo calcola le somme sul suo blocco:
  $$N_k^{\text{loc}},\quad \sum_i r_{ik}x_i,\quad \sum_i r_{ik}x_i^2$$
  poi una sola `MPI_Allreduce` le somma tra tutti i processi. La quantità comunicata dipende da $K$ e $D$ e **non da $N$**.
- **Parametri:** dopo l'`Allreduce` ogni processo ha le stesse somme e ricalcola da sé $\mu_k$, $\sigma_k^2$, $\pi_k$ (piccolo calcolo, nessun broadcast).
- **Log-verosimiglianza:** somma locale più `MPI_Allreduce` di un solo numero, così tutti decidono la convergenza allo stesso modo.
- **Non parallelizzabile:** le iterazioni tra loro, perché ognuna usa i parametri della precedente. Questo limita lo speedup (legge di Amdahl).

## 12. Cosa saper dire all'orale

1. **L'EM cerca gruppi non etichettati** modellando i dati come mescolanza di gaussiane.
2. **Due passi:** E (probabilità di appartenenza di ogni punto a ogni gruppo) e M (ricalcolo di centro, forma e peso con medie pesate).
3. **Perché si ripete:** conoscere i gruppi serve per le probabilità, conoscere le probabilità serve per i gruppi.
4. **Cosa fa la covarianza:** descrive come le variabili variano insieme e dà alla gaussiana forma e inclinazione. Noi usiamo la versione diagonale per semplicità e costo $O(NKD)$.
5. **Cosa si parallelizza:** i punti sono indipendenti, e l'M-step sono somme che si combinano con `MPI_Allreduce`.
6. **Limiti:** minimi locali (dipende dall'inizializzazione), scelta di $K$, iterazioni sequenziali.

## Riferimenti

- A. P. Dempster, N. M. Laird, D. B. Rubin, "Maximum Likelihood from Incomplete Data via the EM Algorithm", *Journal of the Royal Statistical Society B*, 1977 (l'articolo originale).
- C. M. Bishop, *Pattern Recognition and Machine Learning*, capitolo 9 (sezioni 9.2 e 9.3).
