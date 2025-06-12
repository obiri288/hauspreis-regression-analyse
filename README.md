# Hauspreis-Vorhersage (Regressions-Analyse)

Dieses Repository dokumentiert ein End-to-End Machine-Learning-Projekt zur Vorhersage von Hauspreisen in Ames, Iowa, basierend auf dem bekannten Kaggle-Datensatz "House Prices: Advanced Regression Techniques".

Das Projekt umfasst eine detaillierte Datenbereinigung, umfassendes Feature Engineering und das Training eines Basismodells (Lineare Regression), das eine **Genauigkeit (R² Score) von 0.64** auf dem Testdatensatz erreichte.

---

## Inhaltsverzeichnis
1.  [Projekt-Workflow](#projekt-workflow)
2.  [Ergebnisse des Basismodells](#ergebnisse-des-basismodells)
3.  [Verwendete Technologien](#verwendete-technologien)
4.  [Setup und Ausführung](#setup-und-ausführung)
5.  [Nächste Schritte](#nächste-schritte)
6.  [Autor](#autor)
7.  [Lizenz](#lizenz)

---

## Projekt-Workflow

Das Projekt folgt einem strukturierten Workflow, der die folgenden Phasen umfasst:

**1. Explorative Datenanalyse (EDA)**
* Analyse des Datensatzes mit fast 80 Merkmalen zur Identifizierung von Datentypen, Verteilungen und potenziellen Korrelationen.
* Besondere Untersuchung der Zielvariable `SalePrice`, um deren Verteilung (leichte Rechtsschiefe) zu verstehen.

**2. Datenbereinigung (Data Cleaning)**
Eine mehrstufige Strategie wurde angewendet, um fehlende Werte (`NaN`) zu behandeln:
* **Entfernen von Spalten:** Spalten mit einem sehr hohen Anteil an fehlenden Werten (>80% wie `PoolQC`, `MiscFeature`, `Alley`, `Fence`) wurden entfernt.
* **Bedeutungsvolle Imputation:** Für kategoriale Merkmale, bei denen `NaN` eine spezifische Bedeutung hat (z.B. "Kein Kamin" bei `FireplaceQu`), wurden die fehlenden Werte mit `"None"` aufgefüllt.
* **Numerische Imputation:** Fehlende numerische Werte wurden kontextabhängig mit `0` gefüllt (z.B. `GarageYrBlt` oder `MasVnrArea` bei Abwesenheit des Merkmals).
* **Intelligente Imputation:** Fehlende `LotFrontage`-Werte wurden mit dem **Median der jeweiligen Nachbarschaft** aufgefüllt, was genauer ist als ein globaler Median.

**3. Feature Engineering**
* **One-Hot Encoding:** Alle verbleibenden kategorialen Text-Spalten (z.B. `MSZoning`, `LotShape` etc.) wurden in numerische Merkmale umgewandelt. Dies erweiterte den Datensatz auf über 200 Spalten.
* Der Datensatz wurde so in ein vollständig numerisches Format überführt, das für Machine-Learning-Modelle geeignet ist.

**4. Modelltraining & Evaluierung**
* Die aufbereiteten Daten wurden in ein Trainings- (80%) und ein Testset (20%) aufgeteilt.
* Ein **Lineares Regressionsmodell** wurde als Basismodell ("Baseline") trainiert, um eine erste Leistungsmessung zu erhalten.

---

## Ergebnisse des Basismodells

Das lineare Regressionsmodell lieferte folgende Leistung auf dem ungesehenen Testdatensatz:

* **R² Score (Bestimmtheitsmaß):** `0.6415`
  * *Das Modell kann ca. 64% der Varianz in den Hauspreisen erklären. Ein solider Startpunkt.*

* **Root Mean Squared Error (RMSE):** `$52,438.57`
  * *Die Preisvorhersagen weichen im Durchschnitt um diesen Betrag vom tatsächlichen Verkaufspreis ab. Diese Metrik bestraft größere Fehler stärker.*

* **Mean Absolute Error (MAE):** `$20,684.51`
  * *Im reinen Durchschnitt liegt der Vorhersagefehler bei ca. $20,685.*

Diese Ergebnisse dienen als Referenzwert, der mit komplexeren Modellen verbessert werden soll.

---

## Verwendete Technologien
* **Sprache:** Python 3.x
* **Bibliotheken:**
    * `Pandas` & `NumPy`: Für Datenmanipulation und numerische Berechnungen.
    * `Scikit-learn`: Für Modelltraining (`LinearRegression`), Datenaufteilung und Evaluierungsmetriken.
    * `Matplotlib` & `Seaborn`: Für die Datenvisualisierung.
* **Umgebung:** Jupyter Notebook / Google Colaboratory.

---

## Setup und Ausführung

Um die Analyse und das Modelltraining nachzuvollziehen:

1.  **Klone dieses Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)obiri288/repositories.git
    cd [DEIN_REPOSITORY_NAME]
    ```

2.  **Installiere die Abhängigkeiten:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Führe das Notebook aus:**
    Öffne die `.ipynb`-Datei in einer Jupyter-Umgebung und führe die Zellen aus. Die Daten (`train.csv`) müssen sich im selben Verzeichnis befinden oder der Pfad muss angepasst werden.

---

## Nächste Schritte

Das lineare Regressionsmodell dient als solide Baseline. Geplante nächste Schritte zur Verbesserung der Vorhersagegenauigkeit umfassen:
* Training von komplexeren, nicht-linearen Modellen wie `RandomForestRegressor` und `GradientBoostingRegressor`.
* Anwendung von Merkmalskalierung (`StandardScaler`).
* Detailliertere Analyse der Merkmalswichtigkeit (Feature Importance).

---

## Autor

* **[DEIN NAME HIER]**
* GitHub: `https://github.com/obiri288`

---

## Lizenz

Dieses Projekt ist unter der **MIT Lizenz** lizenziert.
