#' ---
#' title: "Support Vector Machines – forelesning 1: R-kode med forklaringer"
#' author: "Øystein Myrland"
#' output:
#'   html_document:
#'     toc: true
#' ---
#'
#' Dette skriptet følger kodebitene i forelesning 1.
#' Linjer som starter med `#'` blir tekst i den rendrede HTML-filen.
#' Linjer som starter med `#` er kommentarer inne i koden.
#' Tallene i forklaringene gjelder når du kjører skriptet med frøene (`set.seed()`) som står her.  

#+ setup, include = FALSE  
knitr::opts_chunk$set(warning = FALSE, message = FALSE)

#' # 1. Pakke og simulerte lånedata  
#'
#' Vi bruker pakken `e1071`, som inneholder `svm()` og `tune()`.
#' Hvis pakken mangler, stopper skriptet med en tydelig beskjed i stedet for en kryptisk feilmelding.  

if (!requireNamespace("e1071", quietly = TRUE)) {
  stop("Installer pakken med install.packages('e1071') før du kjører skriptet.")
}
library(e1071)

#' Vi lager 100 tenkte lånesøknader.
#' Fordi dataene er simulert, vet vi hvordan sammenhengen egentlig ser ut, og vi kan sjekke om modellen finner den.  

set.seed(9)                 # Gjør tilfeldige trekk reproduserbare: samme frø gir samme data
n <- 100                    # Antall lånesøknader

# Inntekt i 1000 kr: normalfordelt rundt 550 (dvs. 550 000 kr) med standardavvik 150  
inntekt    <- round(rnorm(n, mean = 550, sd = 150))

# Gjeldsgrad (gjeld delt på inntekt): rundt 2,5. pmax(..., 0.2) hindrer urimelige negative verdier  
gjeldsgrad <- round(pmax(rnorm(n, mean = 2.5, sd = 1), 0.2), 2)

# Uobservert «risiko»: lav inntekt og høy gjeldsgrad øker risikoen.
# scale() standardiserer, slik at de to variablene veier omtrent likt.
# rnorm(n, sd = 0.8) er støy: forhold vi ikke observerer. Den gjør at klassene overlapper.  
risiko <- -as.numeric(scale(inntekt)) + 0.9 * as.numeric(scale(gjeldsgrad)) +
  rnorm(n, sd = 0.8)

# Klassen: mislighold hvis risikoen er over 0,4.
# factor() er avgjørende: da forstår svm() at dette er klassifikasjon, ikke regresjon.  
y <- factor(ifelse(risiko > 0.4, "Mislighold", "Ingen mislighold"))

dat_laan <- data.frame(inntekt, gjeldsgrad, y)

summary(dat_laan)

#' **Slik leser du output:**
#' Inntekt går fra 157 til 952 (tusen kr), mens gjeldsgrad går fra 0,2 til 4,3.
#' Variablene er altså målt på helt ulike skalaer.
#' Under `y` ser vi at 57 søknader er «Ingen mislighold» og 43 er «Mislighold».
#'
#' Merk deg tallet 57: en «modell» som alltid gjetter «Ingen mislighold», treffer 57 % av gangene.
#' En SVC må gjøre det klart bedre enn dette for å være nyttig.  
#'    
#'    
#' # 2. Standardisering og en egen plottefunksjon   
#'  
#' For å kunne *se* marginen standardiserer vi begge prediktorene, slik at de får gjennomsnitt 0 og standardavvik 1.
#' Da betyr én enhet på hver akse «ett standardavvik», og avstander i plottet blir sammenlignbare.
#' Her standardiserer vi hele datasettet på én gang; det er greit for en illustrasjon, men ikke når vi skal velge modell (se del 4).

dat_std <- data.frame(inntekt    = as.numeric(scale(dat_laan$inntekt)),
                      gjeldsgrad = as.numeric(scale(dat_laan$gjeldsgrad)),
                      y          = dat_laan$y)

#' Plottefunksjonen i `e1071` viser ikke marginene og bytter om på aksene.
#' Vi lager derfor vår egen funksjon `tegn_svc()`.
#'
#' `svm()` lagrer ikke koeffisientene $\beta_1, \beta_2$ direkte, men vi kan regne dem ut fra det den lagrer:
#'
#' - `mod$SV`: støttevektorene (én rad per støttevektor, én kolonne per prediktor),
#' - `mod$coefs`: vekten $\alpha_i$ til hver støttevektor,
#' - `mod$rho`: konstantleddet med motsatt fortegn, altså $\beta_0 = -\rho$.
#'
#' Koeffisientvektoren er en vektet sum av støttevektorene: $\boldsymbol\beta = \sum_{i \in \mathcal S} \alpha_i x_i$.
#' Dette kommer vi tilbake til i forelesning 2.
#'
#' For å tegne linjen der $f(x) = k$, løser vi $\beta_0 + \beta_1 x_1 + \beta_2 x_2 = k$ med hensyn på $x_2$:
#' $$x_2 = \frac{k - \beta_0}{\beta_2} - \frac{\beta_1}{\beta_2}\,x_1.$$
#' Med $k = 0$ får vi beslutningsgrensen, og med $k = \pm 1$ får vi de to marginene.

tegn_svc <- function(mod, data, main = "") {
  # Koeffisienter: beta = sum_i alpha_i * x_i, regnet ut som matriseprodukt
  beta  <- drop(t(mod$coefs) %*% mod$SV)
  beta0 <- -mod$rho

  # Farge etter faktisk klasse
  farge <- ifelse(data$y == "Mislighold", "#cc583b", "#276ca6")

  # Spredningsplott. asp = 1 gir lik skala på aksene, slik at rette vinkler og avstander ser riktige ut
  plot(data$inntekt, data$gjeldsgrad, col = farge, pch = 19, asp = 1,
       xlab = "Inntekt (standardisert)", ylab = "Gjeldsgrad (standardisert)",
       main = main)

  # Tre linjer: f(x) = -1 og f(x) = +1 (stiplet, marginene) og f(x) = 0 (heltrukket, grensen)
  for (k in c(-1, 0, 1)) {
    abline(a = (k - beta0) / beta[2],          # skjæringspunkt med y-aksen
           b = -beta[1] / beta[2],             # stigningstall
           lty = ifelse(k == 0, 1, 2),         # heltrukket for grensen, stiplet for marginene
           lwd = ifelse(k == 0, 2, 1))         # tykkere linje for grensen
  }

  # mod$index er radnumrene til støttevektorene. Vi tegner en ring rundt hver av dem
  points(data[mod$index, c("inntekt", "gjeldsgrad")], cex = 1.8)

  legend("bottomleft", legend = c("Mislighold", "Ingen mislighold"),
         col = c("#cc583b", "#276ca6"), pch = 19, bty = "n", cex = 0.8)
}
  
#' # 3. To lineære SVC-er: lav og høy `cost`  
#'
#' Vi tilpasser to modeller på de samme dataene.
#' Eneste forskjell er `cost`, altså hvor dyrt det er å bryte marginen.
#'
#' - `kernel = "linear"` gir en rett beslutningsgrense (en støttevektorklassifikator).
#' - `scale = FALSE` fordi vi allerede har standardisert dataene selv.
#' - `y ~ inntekt + gjeldsgrad` leses som i `lm()`: klassen forklares av de to prediktorene.  

mod_lav <- svm(y ~ inntekt + gjeldsgrad, data = dat_std, kernel = "linear",
               cost = 0.05, scale = FALSE)   # Billige brudd: forventer bred margin
mod_hoy <- svm(y ~ inntekt + gjeldsgrad, data = dat_std, kernel = "linear",
               cost = 100, scale = FALSE)    # Dyre brudd: forventer smal margin

#+ fig.width = 10, fig.height = 5
par(mfrow = c(1, 2))     # To plott ved siden av hverandre
tegn_svc(mod_lav, dat_std, main = "Lav cost: 0,05")
tegn_svc(mod_hoy, dat_std, main = "Høy cost: 100")
par(mfrow = c(1, 1))     # Tilbake til ett plott per figur  

#' **Slik leser du figuren:**
#' Heltrukken linje er beslutningsgrensen: søknader på den ene siden klassifiseres som mislighold, på den andre som ikke mislighold.
#' De stiplede linjene er marginene.
#' Punkter med ring er støttevektorer: de ligger på marginen, innenfor den eller på feil side av grensen.
#' Punkter uten ring ligger utenfor marginen på riktig side og påvirker ikke grensen i det hele tatt.  
#'
#' Med lav `cost` (venstre) er marginen bred, og mange punkter er støttevektorer.
#' Med høy `cost` (høyre) er marginen smal, og færre punkter bestemmer hvor grensen går.
#' Legg også merke til at grensen ikke ligger på samme sted i de to panelene: valget av `cost` endrer selve klassifikasjonsregelen.  

#' Nå oppsummerer vi de to modellene i en tabell.
#' Marginbredden måles fra den ene stiplede linjen til den andre.
#' Hver side av grensen har bredde $1/\lVert\boldsymbol\beta\rVert$, så hele marginen er $2/\lVert\boldsymbol\beta\rVert$.  

marginbredde <- function(mod) {
  beta <- drop(t(mod$coefs) %*% mod$SV)   # Samme utregning av beta som i tegn_svc()
  2 / sqrt(sum(beta^2))                   # 2 / ||beta||
}

data.frame(
  modell                = c("Lav cost", "Høy cost"),
  antall_stottevektorer = c(mod_lav$tot.nSV, mod_hoy$tot.nSV),   # tot.nSV = totalt antall støttevektorer
  marginbredde          = round(c(marginbredde(mod_lav), marginbredde(mod_hoy)), 2),
  # predict() gir predikert klasse. != gir TRUE ved feil, og mean() av TRUE/FALSE er andelen feil
  treningsfeil          = c(mean(predict(mod_lav, dat_std) != dat_std$y),
                            mean(predict(mod_hoy, dat_std) != dat_std$y))
)

#' **Slik leser du output:**
#' Modellen med lav `cost` har 69 støttevektorer og en margin på 2,18 standardavvik.
#' Modellen med høy `cost` har 46 støttevektorer og en margin på 1,04, altså omtrent halvparten så bred.
#' Begge feilklassifiserer 22 % av treningsdataene.  
#'
#' Treningsfeilen skiller altså ikke mellom modellene, men de er likevel svært forskjellige.
#' Modellen med høy `cost` hviler på færre punkter og vil endre seg mer om dataene endres litt: den har høyere varians.
#' For å avgjøre hvilken som predikerer best på *nye* søknader, trenger vi data modellen ikke har sett.
#' 
#' 
#' # 4. Velg `cost` med kryssvalidering  
#' 
#' Nå gjør vi det på ordentlig.
#' Først deler vi de rå dataene i en treningsdel (70 %) og en testdel (30 %).
#' Testdelen legger vi til side og rører ikke før helt til slutt.  

set.seed(91)
# sample() trekker 70 tilfeldige radnumre uten tilbakelegging
id_train   <- sample(seq_len(nrow(dat_laan)), size = 0.7 * nrow(dat_laan))
train_laan <- dat_laan[id_train, ]    # 70 søknader til trening og kryssvalidering
test_laan  <- dat_laan[-id_train, ]   # De 30 andre (minus = «alle unntatt») til testen helt til slutt

#' `tune()` prøver hver `cost`-verdi i rutenettet med femfolds kryssvalidering på treningsdelen:  
#'
#' 1. Treningsdelen deles i fem like store deler (folder) à 14 søknader.
#' 2. Modellen tilpasses på fire folder og testes på den femte.
#' 3. Dette gjentas fem ganger, slik at hver fold er testfold én gang.
#' 4. Feilraten gjennomsnittes over de fem foldene.  
#'
#' Nå bruker vi de *rå* dataene og `scale = TRUE`.
#' Da standardiserer `svm()` på nytt i hver runde, med gjennomsnitt og standardavvik fra de fire treningsfoldene.
#' Valideringsfolden påvirker dermed ikke skaleringen.  

set.seed(92)   # Foldene trekkes tilfeldig; frøet gjør inndelingen reproduserbar
cv_lin <- tune(
  svm, y ~ ., data = train_laan,             # y ~ . betyr «alle andre kolonner som prediktorer»
  kernel = "linear", scale = TRUE,
  ranges = list(cost = c(0.01, 0.1, 1, 10, 100)),   # Rutenettet av cost-verdier vi prøver
  tunecontrol = tune.control(cross = 5)             # Femfolds kryssvalidering (standard er 10)
)

summary(cv_lin)$performances

#' **Slik leser du output:**
#' Hver rad er én `cost`-verdi (`1e-02` betyr 0,01 og `1e+02` betyr 100).
#'
#' - `error` er gjennomsnittlig feilklassifiseringsrate over de fem valideringsfoldene.
#' - `dispersion` er standardavviket i feilraten over foldene, altså et mål på usikkerheten.
#'
#' `cost = 0,01` er tydelig dårligst (44 % feil): marginen blir så bred at modellen nesten ikke skiller klassene.
#' Fra `cost = 0,1` og oppover ligger feilraten mellom 21 % og 24 %.
#' Forskjellene er små sammenlignet med `dispersion` (rundt 0,12).
#' En forskjell på 0,015 svarer omtrent til én av de 70 søknadene.  

cv_lin$best.parameters

#' **Slik leser du output:**
#' `tune()` velger `cost = 1`, fordi den har lavest CV-feil.
#' Tallet `3` til venstre er bare radnummeret i rutenettet, ikke en verdi vi skal tolke.
#' `tune()` har også tilpasset den valgte modellen på nytt på hele treningsdelen; den ligger i `cv_lin$best.model`.
#' 
#' 
#' # 5. Én evaluering på testdata
#'
#' Til slutt bruker vi den valgte modellen på de 30 søknadene modellen aldri har sett.
#' `predict()` skalerer testdataene automatisk med gjennomsnitt og standardavvik fra treningsdelen.  

pred_lin <- predict(cv_lin$best.model, newdata = test_laan)

# Forvekslingsmatrise: rader er faktisk klasse, kolonner er predikert klasse
table(faktisk = test_laan$y, predikert = pred_lin)

#' **Slik leser du forvekslingsmatrisen:**
#' Diagonalen (17 og 8) er riktige klassifikasjoner, og utenfor diagonalen (2 og 3) er feil.
#'
#' - Av de 19 som *ikke* misligholdt, ble 17 klassifisert riktig. Spesifisiteten er $17/19 \approx 0{,}89$.
#' - Av de 11 som misligholdt, fanget modellen opp 8. Sensitiviteten er $8/11 \approx 0{,}73$.
#' - 3 misligholdere ble ikke oppdaget (falske negative), og 2 gode kunder ble avvist (falske positive).
#'
#' For en bank koster disse to feiltypene sannsynligvis svært forskjellig.  

mean(pred_lin == test_laan$y)   # Andel riktige: (17 + 8) / 30

#' **Slik leser du output:**
#' Testnøyaktigheten er 0,833, altså 25 av 30 riktige.
#' Til sammenligning ville en regel som alltid sier «Ingen mislighold» truffet $19/30 \approx 0{,}63$ i testsettet.
#' Modellen gjør det altså klart bedre enn det enkleste alternativet.
#'
#' Husk likevel at testsettet bare har 30 observasjoner, så tallet er usikkert: én observasjon utgjør over 3 prosentpoeng.
#' Og nå som vi har sett testresultatet, skal vi *ikke* gå tilbake og justere `cost` for å få et bedre tall.
