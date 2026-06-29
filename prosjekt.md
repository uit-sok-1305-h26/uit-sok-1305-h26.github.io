{% include navbar.html %}{% include top-box.html %}
# Prosjekt: Statistisk Læring (SOK-1305)

I dette faget skal vi ha et praktisk data science- og maskinlæringsprosjekt, hvor dere vil bli utfordret på å bygge, evaluere og tolke egne modeller. Dere skal gjennomføre arbeidet enten alene, eller i en gruppe på to.

## 🎯📊 Oppgave
Dere skal utforske og sammenligne statistiske læringsmetoder og maskinlæringsmodeller på et selvvalgt eller anbefalt datasett (gjerne med en økonomisk vinkling). Målet er å analysere hvordan ulike teknikker presterer på den valgte problemstillingen, med sterk forankring i pensum og **ISLP-boka** (*Introduction to Statistical Learning with Applications in Python*).

### 🔍 Metoder og Statistisk Læring
Prosjektet skal vise at dere behersker sentrale konsepter fra statistisk læring. Dere skal bruke **minst én to teknikker fra pensum**, som for eksempel:
- **Trebaserte metoder:** Random Forests, Bagging eller Boosting.
- **Reguleringsteknikker:** Ridge- og Lasso-regresjon.
- **Support Vector Machines (SVM).**
- **Lineær regrejson**: Enkel og multippel
- **Logitistisk regresjon**

For å etablere et referansepunkt for ytelsen, **skal** dere sammenligne den avanserte metoden med mer grunnleggende og tolkbare modeller (baseline-modeller), slik som:
- Lineær regresjon (Multiple Linear Regression)
- Logistisk regresjon
- Enkle beslutningstrær (Decision Trees)

**Husk:** Mer komplekse modeller er ikke alltid best! Ta hensyn til *Bias-Variance Trade-off*, og evaluer modellene basert på deres evne til å predikere usette data (testdata) versus hvor lett det er å tolke resultatene (*Prediction vs. Inference*).

## 📂 Krav til prosjektet

Jeg har funnet frem en del eksempel-datasett dere kan (les: *kan*, ikke *må*) bruke. Lenke til disse finner dere nederst på siden. Dere står også fritt til å velge datasett selv (f.eks. fra Yahoo Finance, Kaggle, SSB, osv.). **Uansett hvilke data dere bruker, må opphav og kilde oppgis!**

Oppgaven har ganske åpne rammer, men stiller følgende formelle krav:

### 1. Datasett og problemstilling
- Velg et datasett.
- Beskriv datasettet, kilden, og definer problemet klart. (Hva er responsvariabelen $Y$, og hva er prediktorene $X$?). Er det et *klassifiserings-* eller *regresjonsproblem*?

### 2. Dataforståelse og preprosessering (Koding)
- Utforsk datasettet gjennom deskriptiv statistikk og visualiseringer.
- Gjør nødvendige transformasjoner i Python/R: håndter manglende verdier, lag dummy-variabler for kategoriske data, og gjennomfør skalering/standardisering.

### 3. Implementasjon av modeller (Python)
- Bruk Python-biblioteker fra ISLP (som `scikit-learn`, `statsmodels` eller lignende).
- Sørg for "riktig" oppdeling av data i trenings- og testsett. Implementer gjerne teknikker som kryssvalidering (*Cross-Validation*), og forklar hvorfor du gjør som du gjør.
- **Koden skal være godt kommentert og forklart!** Forklar *hvorfor* dere gjør som dere gjør, ikke bare *hva* koden gjør.

### 4. Matematisk og statistisk beskrivelse
Bruk **LaTeX i Markdown** for å forklare den matematiske intuisjonen og konseptene bak modellene dere benytter.
- For lineær regresjon (OLS) kan estimatoren f.eks. uttrykkes slik:
  $$\hat{\beta} = (X^T X)^{-1} X^T y$$

### 5. Evaluering av modellene
- Bruk relevante evalueringsmetrikker: $R^2$, MSE, RMSE (regresjon) eller Accuracy, Precision, Recall, F1-score, confusion-matrix (klassifisering).
- Sammenlign modellene. Vurder graden av generaliseringsevne, og diskuter om det er tegn til under- eller overtilpassing (*overfitting*).

### 6. Konklusjon og refleksjon
- Hvilken modell presterte best på testdata, og hvorfor?
- Hvordan kan modellen eventuelt forbedres?
- Hva kan disse resultatene brukes til i praksis i en bedriftsøkonomisk eller samfunnsøkonomisk kontekst?


## 📝 Format og innlevering
- Prosjektet skal leveres som en **Jupyter Notebook (`.ipynb`)** eller **R-markdown (`.rmd`)**, avhengig av om du vil bruke Python eller R.
- Notebooken skal kunne kjøres direkte i **Google Colab** uten behov for eksterne avhengigheter (husk å importere alt av biblioteker i toppen av dokumentet). Eller i RStudio.
- Legg ved datasettet!
- Bruk **APA 7** referansestil for datasett, litteratur og kode dere har hentet inspirasjon fra.

## 📚 Vurdering av oppgaven
Denne semesteroppgaven teller **40% av den endelige karakteren** i emnet. Det gis en bokstavkarakter (A-F), som vurderes ut ifra følgende kriterier:

1. **Faglig forståelse:** Forståelse av statistisk læring, algoritmene og de matematiske egenskapene deres, forankret i pensum (ISLP).
2. **Koding og implementasjon:** At datahåndteringen og kodekvaliteten er ryddig, logisk og godt kommentert.
3. **Problemløsning og diskusjon:** Hvordan problemstillingen er besvart, struktur i oppgaven, og refleksjon rundt funn.
4. **Tverrfaglighet:** Å vise en bredere forståelse for anvendelse av maskinlæring i økonomi og samfunn teller svært positivt.

**Bruk av Kunstig Intelligens (KI / AI):**
Det er tillatt å bruke AI som et hjelpemiddel (til koding, retting av feil og begrepsforståelse). Men ting skal formidles med **egne ord**, og dere skal forstå den matematiske teorien og koden som leveres. Å la en KI skrive avsnitt, drøfte for deg eller levere kode du ikke kan redegjøre for er juks, og kan føre til utestengelse.

**Innleveringsfrist:** 23. november kl. 14:00 på Wiseflow.
- Her kan dere laste opp totalt 5 filer. 
- Må inneholde: En `.ipynb`-fil og datasettet deres (`.xlsx`, `.csv` e.l.). 
- Husk: pdf-versjon av oppgaven eller andre aktuelle vedlegg.

**Siste huskeregel:** Formålet er ikke at modellen skal oppnå 100 % *accuracy* (det indikerer ofte *data leakage* på ekte data). Det som skiller en god og en dårlig oppgave, er evnen til å vise statistisk forståelse, drøfte begrensninger i metoden og bruke statistisk læring på en gjennomtenkt og fornuftig måte.

Lykke til! 🚀

---
**Ressurser:**
- Vi har en rekke datasett som er lastet opp eller linket til på GitHub, det finner dere [her](https://github.com/uit-sok-1305-h26/uit-sok-1305-h26.github.io/blob/main/data/README.md).
- Det er massevis av data som kan hentes gjennom Python/R bibliotek knyttet til pensumboka [ISLP - Introduction to Statistical Learning (Ressursside)](https://www.statlearning.com/)
