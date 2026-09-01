{% include navbar.html %}{% include top-box.html %}
# 📝 Semesteroppgave i SOK-1305 – Statistisk læring for økonomer

**Teller 40 % av endelig karakter i emnet.**
**Innleveringsfrist: 23. november kl. 14:00 i Canvas.**

---

## 🎯 Hva går oppgaven ut på?

I denne oppgaven skal dere gjennomføre et lite, men fullstendig prosjekt i statistisk læring: fra rådata til ferdig analyse og konklusjon. Dere velger ett av fem datasett fra ISLP-boka, definerer en problemstilling, bygger og sammenligner modeller, og tolker resultatene med økonomiske briller.

Poenget er **ikke** å få høyest mulig treffsikkerhet. Poenget er å vise at dere forstår hva modellene gjør, hvorfor de gjør det, og hva resultatene faktisk betyr. En oppgave med moderat prediksjonsevne og god drøfting er bedre enn en oppgave med 99 % *accuracy* og ingen refleksjon.

Alt arbeid skal forankres i pensum og i **ISLP** (*An Introduction to Statistical Learning with Applications in Python*).

---

## 👥 Organisering

- Dere kan levere **alene** eller i **gruppe på to**. Begge alternativer vurderes likt.
- Grupper på to leverer ett felles produkt og får samme karakter.
- Gruppesammensetning og valg av datasett meldes inn innen **10. september 2026**.

---

## 📊 Velg ett av fem datasett

Alle datasettene ligger i **ISLP-pakka** (Python) og i **ISLR2** (R). Dere trenger derfor ikke lete etter data, og dere slipper å bruke tid på innlasting og opprydding – tida skal brukes på modellering og drøfting.

```python
from ISLP import load_data
Carseats = load_data('Carseats')
```

```r
library(ISLR2)
data(Carseats)
```

| # | Datasett | Type problem | Respons $Y$ | Størrelse | Økonomisk tema |
|---|----------|--------------|-------------|-----------|----------------|
| 1 | `Carseats` | Regresjon (evt. klassifisering) | `Sales` | 400 × 11 | Pris, konkurranse, markedsføring |
| 2 | `College` | Regresjon eller klassifisering | `Outstate` eller `Private` | 777 × 18 | Utdanningsøkonomi |
| 3 | `OJ` | Klassifisering | `Purchase` (CH/MM) | 1 070 × 18 | Konsumentvalg og prisrespons |
| 4 | `Caravan` | Klassifisering | `Purchase` | 5 822 × 86 | Forsikringssalg, skjeve klasser |
| 5 | `Bikeshare` | Regresjon | `bikers` | 8 645 × 15 | Etterspørsel og transport |

### 1. `Carseats` – salg av barneseter i 400 butikker 🛒

Responsen er `Sales` (antall solgte enheter i tusen per butikk). Blant prediktorene finner dere egen pris (`Price`), konkurrentens pris (`CompPrice`), reklamebudsjett (`Advertising`), inntektsnivået i området (`Income`), hylleplassering (`ShelveLoc`) og aldersprofil (`Age`).

**Økonomisk vinkling:** egenpris- og krysspriselastisitet, effekten av markedsføring, betydningen av plassering i butikk.

### 2. `College` – 777 amerikanske universiteter og høyskoler 🎓

Dere kan enten predikere skolepengene for studenter utenfra (`Outstate`) som et **regresjonsproblem**, eller om institusjonen er privat (`Private`) som et **klassifiseringsproblem**. Datasettet inneholder søkertall, opptakstall, andel studenter fra toppen av videregående, kostnader, andel ansatte med doktorgrad, student/lærer-forhold, utgifter per student og fullføringsgrad.

**Økonomisk vinkling:** prisdannelse i utdanningsmarkedet, ressursbruk og kvalitet, seleksjon, avkastning av utdanning.

### 3. `OJ` – 1 070 kjøp av appelsinjuice 🧃

Responsen er hvilket merke kunden valgte, Citrus Hill eller Minute Maid. Prediktorene er priser for begge merkene, rabatter, kampanjer, prisdifferanse og et mål på kundens merkelojalitet.

**Økonomisk vinkling:** diskret valg, prisrespons og substitusjon mellom merker – dette er etterspørselsteori i praksis. ISLP bruker datasettet både til klassifikasjonstrær (kapittel 8) og til SVM (kapittel 9).

### 4. `Caravan` – hvem kjøper campingvognforsikring? 🚐

5 822 personer beskrevet med 85 demografiske og produktrelaterte variabler. Responsen `Purchase` sier om personen kjøpte forsikringen. Bare 348 personer, altså rundt 6 %, gjorde det.

**Økonomisk vinkling:** målretting av salgsinnsats. Når et kundebesøk koster penger og bare 6 % kjøper, er spørsmålet ikke "hvor mange klassifiserer vi riktig", men "hvor stor andel av dem vi oppsøker, kjøper faktisk?".

### 5. `Bikeshare` – timesdata for bysykler i Washington DC 🚲

8 645 timer fra perioden 2010–2012. Responsen `bikers` er antall utleide sykler den timen. Prediktorene er sesong, måned, klokkeslett, ukedag, helligdag, værtype, temperatur, luftfuktighet og vind.

**Økonomisk vinkling:** etterspørsel etter transporttjenester, kapasitetsplanlegging, vær- og sesongvariasjon, og hva det koster å bomme på prognosen i begge retninger.

> **Andre ISLP-datasett?** Ønsker dere å bruke et annet datasett fra boka, må det avklares med faglærer før **14. september 2026**. Datasett vi har brukt mye i forelesning (`Boston`, `Default`, `Auto`, `Advertising`, `Wage`) er ikke tillatt, siden mye av jobben da allerede er gjort for dere.

---

## 🧭 Hva oppgaven skal inneholde

Bruk gjerne denne inndelingen som disposisjon i notebooken.

### 1. Problemstilling og data
- Beskriv datasettet: hvor kommer det opprinnelig fra, hvor mange observasjoner ($n$) og variabler ($p$), og hva måler variablene?
- Formuler en tydelig problemstilling. Hva er responsvariabelen $Y$, og hvilke prediktorer $X$ bruker dere – og hvorfor akkurat de?
- Er dette et **regresjons-** eller **klassifiseringsproblem**?
- Er målet først og fremst **prediksjon** eller **inferens**? Dette valget bør prege resten av oppgaven.

### 2. Utforskende analyse og preprosessering
- Deskriptiv statistikk og noen få, men gjennomtenkte figurer. Kommenter det dere ser – figurer uten tekst teller ikke.
- Håndter eventuelle manglende verdier, og forklar valget dere tar.
- Lag dummyvariabler for kategoriske variabler, og standardiser der metoden krever det (særlig KNN og SVM).
- **Viktig:** standardisering og imputering skal tilpasses på treningssettet og deretter brukes på testsettet.

### 3. Modellering
Dere skal bruke **minst to metoder fra pensum**, og minst én av dem skal være en enkel og tolkbar (*baseline* modell):

| Baseline (minst én) | Mer fleksibel metode (minst én) |
|---|---|
| Enkel/Multippel lineær regresjon (OLS) | Random forest / bagging |
| Logistisk regresjon | Boosting |
| Enkelt beslutningstre | Support vector machine (SVM) |
| | KNN |

Videre kreves:
- Riktig oppdeling i trenings- og testsett, med fast `random_state` / `set.seed()` slik at resultatene kan reproduseres.
- Kryssvalidering der det er naturlig, særlig ved valg av tuningparametere (antall trær, treets dybde, $K$ i KNN, $C$ i SVM osv.).
- **Tuning skjer på treningsdata.** Testsettet røres først når modellene er ferdig valgt.
- Koden skal være kommentert. Forklar *hvorfor* dere gjør som dere gjør, ikke bare *hva* koden gjør.

### 4. Matematisk beskrivelse
Bruk **LaTeX i markdown-celler** til å forklare intuisjonen bak metodene dere bruker. Dette er ikke et bevis-kurs – kravet er at dere kan skrive ned modellen og forklare med egne ord hva den estimerer, og hva de enkelte leddene betyr.

Eksempler på passende nivå:

$$\hat{\beta} = (X^\top X)^{-1} X^\top y$$

$$p(X) = \frac{e^{\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p}}{1 + e^{\beta_0 + \beta_1 X_1 + \dots + \beta_p X_p}}$$

$$\text{MSE}_{\text{test}} = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{f}(x_i)\right)^2$$

Minst ett sted i oppgaven bør dere knytte an til dekomponeringen

$$E\left[(y_0 - \hat{f}(x_0))^2\right] = \operatorname{Var}(\hat{f}(x_0)) + \left[\operatorname{Bias}(\hat{f}(x_0))\right]^2 + \operatorname{Var}(\varepsilon)$$

og bruke den til å forklare hva dere faktisk observerer i deres egne resultater.

### 5. Evaluering og sammenligning
- **Regresjon:** MSE, RMSE og $R^2$, oppgitt for både trenings- og testsett.
- **Klassifisering:** *accuracy*, presisjon, *recall*, F1 og confusion matrix. Drøft hvilket mål som er viktigst *for akkurat deres problemstilling*.
- Sammenlign modellene i en tabell.
- Diskuter over- og undertilpasning. Stor forskjell mellom trenings- og testytelse er et signal dere skal kommentere, ikke skjule.

### 6. Konklusjon og refleksjon
- Hvilken modell presterte best på testdata, og hvorfor tror dere det ble slik?
- Hva er svakhetene ved analysen? Hva ville dere gjort med mer tid eller bedre data?
- Hva kan resultatene brukes til i praksis – for en bedrift, en offentlig etat eller en beslutningstaker?
- Er det etiske eller fordelingsmessige sider ved å bruke en slik modell i praksis?

### 7. Kilder
Alle kilder – datasett, litteratur, pakker og kode dere har hentet inspirasjon fra – føres opp i **APA 7**-stil. Datasettet siteres via boka, for eksempel:

> James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An introduction to statistical learning with applications in Python*. Springer. https://doi.org/10.1007/978-3-031-38747-0

---

## 💻 Tekniske krav

**Filformat**
- Python-brukere leverer **Jupyter Notebook** (`.ipynb`).
- R-brukere leverer **R Markdown** (`.Rmd`).
- I tillegg skal dokumentet være **kjørt ut til PDF**. PDF-en er det vi leser og vurderer, så sjekk at alle figurer, tabeller og LaTeX-formler faktisk vises der.
  - *Colab:* Fil → Skriv ut → Lagre som PDF (kjør alle celler først!).
  - *RStudio:* Knit → Knit to PDF.
  - Det finnes et hav av YouTube videoer som viser hvordan man får en ipynb eller Rmd om til pdf.

**Reproduserbarhet**
- Notebooken skal kunne kjøres fra topp til bunn uten feil, i Google Colab eller RStudio.
- **Alle biblioteker importeres i én celle øverst i dokumentet**, og dere skal skrive kort hvilke pakker dere bruker og til hva. Oppgi gjerne versjonsnummer.
- Husk at ISLP må installeres i Colab (`!pip install ISLP`). Ta med linja i notebooken.
- Sett et tilfeldighetsfrø (`random_state=1305` / `set.seed(1305)`) slik at vi får de samme tallene som dere når vi kjører koden.
- Eksporter datasettet til `.csv` og legg det ved, slik at innleveringen står på egne bein.

**Omfang**
Ferdig PDF vil typisk ligge på 15–25 sider inkludert kode og figurer. Det er ingen absolutt grense, men unngå å lime inn lange utskrifter uten kommentar. Kvalitet foran kvantitet.

---

## ⚠️ Vanlige fallgruver

Disse går igjen hvert år. Les lista én gang til før innlevering:

1. **Datalekkasje.** En prediktor som inneholder informasjon som ikke ville vært tilgjengelig på beslutningstidspunktet, gir kunstig gode resultater. Nær 100 % treffsikkerhet er nesten alltid et symptom på dette.
2. **Standardisering før oppdeling.** Skalerer dere hele datasettet før dere splitter, har testsettet påvirket treningen.
3. **Evaluering på treningsdata.** Alle endelige tall skal komme fra testsettet.
4. **Tuning mot testsettet.** Prøver dere femten modeller og velger den som gjør det best på testsettet, er testsettet ikke lenger uavhengig.
5. **Accuracy på ubalanserte data.** Er 94 % av observasjonene i én klasse, får en modell som alltid gjetter "nei" 94 % *accuracy* – og er ubrukelig.
6. **Figurer uten kommentar.** Hver figur skal ha en setning som sier hva vi skal se.
7. **Å tolke korrelasjon som årsakssammenheng.** En random forest sier hvilke variabler som *predikerer* godt, ikke hva som *forårsaker* hva.

---

## 📚 Vurdering

Oppgaven teller **40 %** av endelig karakter, og vurderes med bokstavkarakter A–F.

| Kriterium | Vekt | Hva vi ser etter |
|---|---|---|
| **Faglig forståelse** | 30 % | Forståelse av metodene, deres statistiske egenskaper og begrensninger, forankret i ISLP. Riktig bruk av begreper. |
| **Koding og implementasjon** | 25 % | Ryddig, kommentert og reproduserbar kode. Korrekt håndtering av data, oppdeling og kryssvalidering. |
| **Problemløsning og diskusjon** | 30 % | Tydelig problemstilling, logisk struktur, og selvstendig refleksjon rundt funn og svakheter. |
| **Anvendelse og formidling** | 15 % | Kobling til økonomisk og samfunnsmessig sammenheng, god språkføring, korrekt kildebruk (APA 7). |

**Det som skiller en A fra en C** er sjelden hvilken modell dere valgte. Det er om dere klarer å forklare *hvorfor* resultatene ble som de ble, drøfte hva metoden ikke kan svare på, og sette funnene inn i en økonomisk sammenheng.

---

## 🤖 Bruk av kunstig intelligens og andres analyser

Det er tillatt å bruke AI som hjelpemiddel – til koding, feilsøking og begrepsforståelse. Men:

- Alt skal formidles med **egne ord**.
- Dere skal kunne forklare **hver linje kode** og **hver påstand** i besvarelsen.
- Husk at alt dere skriver kan dere bli spurt om på muntlig eksamen

ISLP-datasettene er mye brukt, og det finnes ferdige løsninger og blogginnlegg om alle fem på nett. Dere kan gjerne se på dem, men da skal de siteres i APA 7 – og analysen, valgene og drøftingen skal være deres egne. 

---

## 📤 Innlevering

**Frist: 23. november kl. 14:00 i Canvas.** Dere kan laste opp inntil **5 filer**.

Innleveringen **må** inneholde:
- [ ] `.ipynb`- eller `.Rmd`-fil
- [ ] PDF-versjon av samme dokument, med alle celler kjørt
- [ ] Datasettet eksportert til `.csv`

**Sjekkliste før dere leverer:**
- [ ] Notebooken kjører fra topp til bunn uten feil
- [ ] Alle biblioteker er importert i én celle øverst, og forklart
- [ ] Random seed er satt
- [ ] Alle figurer og formler vises riktig i PDF-en
- [ ] Kildeliste i APA 7
- [ ] Navn (og eventuell gruppepartner) står øverst i dokumentet

---

## 🗓️ Underveis

| Milepæl | Frist |
|---|---|
| Melde inn gruppe og valgt datasett | 10. september 2026 |
| Presentere prosjektskisse | 10. eller 14. september 2026 (dere må melde ifra når!) |
| Veiledning / spørretime | Avtal direkte med foreleser, enten i uke 37, 38, 45, 46 eller 47  |
| Enkle spørsmål | Dette er et selvstendig prosjekt, men enkle spørsmål kan sendes via mail |
| Innlevering i Canvas | 23. november kl. 14:00 |

---

Lykke til! 🚀
