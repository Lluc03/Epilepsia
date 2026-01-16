# Repte 4 — Detecció de crisis epilèptiques amb EEG (CHB-MIT)

Projecte de classificació binària de finestres EEG per a la detecció automàtica de crisis epilèptiques utilitzant models d’aprenentatge profund.

Classes:
- `0` → Estat normal
- `1` → Crisi epilèptica

---

## 1) Objectiu del projecte

L’objectiu del projecte és desenvolupar i avaluar diferents models de deep learning capaços de detectar crisis epilèptiques a partir de senyals EEG multicanal.

Es comparen dos enfocaments principals:
- **Model estàtic (CNN):** tracta cada finestra EEG de manera independent
- **Model temporal (LSTM):** incorpora informació temporal per millorar la detecció i reduir falsos positius

L’avaluació es realitza amb estratègies orientades a la generalització entre pacients.

---

## 2) Dataset i format de dades

### Dataset CHB-MIT
- 24 pacients (23 pediàtrics, 1 adult)
- Registres EEG reals en entorn clínic
- Senyals multicanal amb 21 elèctrodes
- Freqüència de mostreig: 128 Hz

### Segmentació
- Finestres d’1 segon
- Cada finestra té dimensions `(21 canals × 128 mostres)`

### Etiquetatge
- Cada finestra s’etiqueta com:
  - `0` → no crisi
  - `1` → crisi
- Es descarten finestres ±30 segons al voltant de les crisis per evitar ambigüitats en la classificació

### Format d’arxius
- `.npz` → dades EEG segmentades
- `.parquet` → metadades i etiquetes

---

## 3) Pipeline general

1. Carrega i preprocessament del senyal EEG
2. Segmentació en finestres d’1 segon
3. Balanceig de dades (segons el model)
4. Entrenament del model (CNN o LSTM)
5. Predicció probabilística
6. Aplicació de threshold
7. Càlcul de mètriques

---

## 4) Models implementats

### A) Model estàtic (CNN)

El model estàtic classifica cada finestra EEG de forma independent, sense considerar la informació temporal entre finestres consecutives.

**Entrada:**
- `(21 × 128)`

**Arquitectura general:**
- Extractor de característiques basat en CNN
- Capes convolucionals + pooling
- Capes fully-connected
- Activació sigmoide final

**Variants implementades:**
- CNN 1D separated
- CNN 2D separated amb fusió de canals

**Estratègies de fusió (feature-level):**
- Concatenació (`concat`)
- Mitjana (`average pooling`)
- Fusió ponderada (`weighted fusion`)

**Avantatges:**
- Arquitectura simple
- Entrenament ràpid
- Alta sensibilitat

**Limitacions:**
- No captura context temporal
- Elevat nombre de falsos positius aïllats

---

### B) Model temporal (LSTM)

El model temporal introdueix memòria mitjançant una xarxa LSTM per capturar dependències temporals entre finestres consecutives.

**Entrada:**
- Seqüències de finestres EEG o de característiques

**Arquitectura:**
- LSTM amb 2 capes
- Hidden size = 128
- Dropout entre capes
- Classificador sobre l’últim hidden state

**Avantatges:**
- Modela la continuïtat temporal de les crisis
- Redueix falsos positius puntuals
- Millora l’equilibri entre sensibilitat i especificitat

**Inconvenients:**
- Major cost computacional
- Entrenament més delicat

---

## 5) Estratègies d’avaluació

### Leave-One-Patient-Out (LOPO)

- En cada iteració:
  - 1 pacient per test
  - Resta per entrenament
- Simula un escenari clínic real
- Avalua la generalització a pacients no vistos

---

### K-Fold LOPO (3 folds)

- Selecció de 5 pacients per test
- Cada fold utilitza un pacient diferent com a test
- Resultats més estables que LOPO pur
- Menor variància entre experiments

---

## 6) Desbalanceig i balanceig de dades

El dataset és fortament desbalancejat:
- Gran majoria de finestres normals
- Percentatge reduït de finestres de crisi

### Estratègies aplicades

#### CNN (model estàtic)
- Undersampling de la classe majoritària
- Mescla aleatòria de finestres
- L’ordre temporal no és rellevant

#### LSTM (model temporal)
- Balanceig temporal
- Preservació de seqüències contigües
- Sense shuffle de finestres individuals
- Es mantenen transicions realistes normal → crisi

---

## 7) Mètriques, resultats i conclusions

### Mètriques utilitzades

- **Sensitivity (Recall)** → detecció de crisis
- **Specificity** → detecció correcta d’estat normal
- **F1-score** → equilibri entre precisió i sensibilitat
- **AUC (ROC)** → capacitat discriminativa global

L’accuracy no es considera representativa degut al desbalanceig.

---

### Resultats obtinguts (resum)

#### CNN estàtic (K-Fold LOPO)
- Accuracy: ~0.68 – 0.85 (poc representativa)
- Sensitivity: molt alta (fins a ~0.95)
- Specificity: mitjana (~0.64 – 0.87)
- F1-score: moderat (~0.42 – 0.56)

Interpretació:
- El model detecta moltes crisis
- Elevat nombre de falsos positius

---

#### CNN + fusió de canals
- Millora lleugera de l’F1-score
- Millor estabilitat entre folds
- Sensitivity alta
- Specificity moderada

---

#### LSTM temporal (LOPO)
- F1-score superior al model estàtic
- Millor equilibri sensibilitat / especificitat
- Reducció clara de falsos positius aïllats
- Millor generalització entre pacients

---

### Conclusions finals

- El problema és altament desbalancejat i clínicament exigent
- Els models CNN aconsegueixen alta sensibilitat però pateixen molts falsos positius
- La incorporació de temporalitat amb LSTM millora l’equilibri global del sistema
- L’F1-score i la sensibilitat són mètriques clau per avaluar el rendiment
- El model temporal és més adequat per a un escenari clínic real

