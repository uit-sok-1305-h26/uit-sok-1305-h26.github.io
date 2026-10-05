#' ---
#' title: "Support Vector Machines – forelesning 2: R-kode med forklaringer"
#' author: "Øystein Myrland"
#' output:
#'   html_document:
#'     toc: true
#' ---
#'
#' Dette skriptet følger kodebitene i forelesning 2.
#' Linjer som starter med `#'` blir tekst i den rendrede HTML-filen.
#' Linjer som starter med `#` er kommentarer inne i koden.
#' Tallene i forklaringene gjelder når du kjører skriptet med frøene (`set.seed()`) som står her.

#+ setup, include = FALSE
knitr::opts_chunk$set(warning = FALSE, message = FALSE)

#' # 1. Pakke og simulerte ringdata
#'
#' Som i forelesning 1 bruker vi pakken `e1071`.

if (!requireNamespace("e1071", quietly = TRUE)) {
  stop("Installer pakken med install.packages('e1071') før du kjører skriptet.")
}
library(e1071)

#' Vi lager 240 punkter: 120 i et sentrum og 120 i en ring rundt.
#' Punktene lages i *polarkoordinater*: en vinkel (retning fra origo) og en radius (avstand fra origo).
#' Deretter regner vi om til vanlige koordinater med $x_1 = r\cos\theta$ og $x_2 = r\sin\theta$.
#' Slik er det lett å styre hvor langt fra sentrum hver klasse ligger.

set.seed(93)
n2 <- 240

# Klassene: de første 120 er «Sentrum», de neste 120 er «Ring»
klasse <- rep(c("Sentrum", "Ring"), each = n2 / 2)

# Vinkel: tilfeldig retning rundt hele sirkelen (0 til 2*pi radianer)
vinkel <- runif(n2, 0, 2 * pi)

# Radius: sentrum ligger 0–1,2 fra origo, ringen 0,9–2,1.
# Intervallene overlapper mellom 0,9 og 1,2, så klassene kan ikke skilles perfekt
radius <- c(runif(n2 / 2, 0, 1.2), runif(n2 / 2, 0.9, 2.1))

dat_ring <- data.frame(
  # Fra polarkoordinater til x1 og x2, pluss litt støy (sd = 0,2)
  x1 = radius * cos(vinkel) + rnorm(n2, sd = 0.2),
  x2 = radius * sin(vinkel) + rnorm(n2, sd = 0.2),
  # levels = ... bestemmer rekkefølgen på klassene: «Sentrum» blir første nivå, «Ring» andre
  y  = factor(klasse, levels = c("Sentrum", "Ring"))
)

plot(dat_ring$x1, dat_ring$x2,
     col = ifelse(dat_ring$y == "Ring", "#cc583b", "#276ca6"),
     pch = 19, xlab = "x1", ylab = "x2", asp = 1)   # asp = 1: lik skala, så sirkelen ser rund ut
legend("topright", legend = levels(dat_ring$y),
       col = c("#276ca6", "#cc583b"), pch = 19, bty = "n")

#' **Slik leser du figuren:**
#' De blå punktene (sentrum) ligger samlet rundt origo, og de røde (ringen) ligger rundt dem.
#' Ingen rett linje kan skille de to klassene: uansett hvor vi legger linjen, får vi både blå og røde punkter på hver side.
#' I overgangssonen blander klassene seg, så heller ikke en buet grense kan klassifisere alle punktene riktig.

#' # 2. Trening og test, og tuning med kryssvalidering
#'
#' Vi deler data i 70 % trening og 30 % test.
#' Delingen er *stratifisert*: vi trekker 70 % fra hver klasse for seg, slik at begge delene får like mange av hver klasse.

set.seed(94)
id_ring <- unlist(lapply(
  split(seq_len(nrow(dat_ring)), dat_ring$y),          # Radnumrene delt i én liste per klasse
  function(ii) sample(ii, size = round(0.7 * length(ii)))  # Trekk 70 % av radene i hver klasse
))                                                     # unlist() slår listene sammen igjen
train_ring <- dat_ring[id_ring, ]    # 168 observasjoner: 84 Sentrum og 84 Ring
test_ring  <- dat_ring[-id_ring, ]   # 72 observasjoner: 36 Sentrum og 36 Ring

#' Vi tuner to modeller med femfolds kryssvalidering (CV) på treningsdataene:
#'
#' - en **lineær SVC**, der vi bare velger `cost`;
#' - en **radial SVM**, der vi velger både `cost` og `gamma`. `tune()` prøver alle $5 \times 4 = 20$ kombinasjoner.
#'
#' Begge bruker samme frø, slik at foldene blir like og sammenligningen rettferdig.
#' `svm()` skalerer prediktorene automatisk (`scale = TRUE` er standard), og det skjer på nytt inne i hver fold.

set.seed(95)
cv_rett <- tune(
  svm, y ~ ., data = train_ring, kernel = "linear",
  ranges = list(cost = c(0.1, 1, 10)),
  tunecontrol = tune.control(cross = 5)
)
set.seed(95)
cv_radial <- tune(
  svm, y ~ ., data = train_ring, kernel = "radial",
  ranges = list(cost  = c(0.01, 0.1, 1, 10, 100),   # Hvor dyrt det er å bryte marginen
                gamma = c(0.05, 0.2, 1, 5)),        # Hvor raskt likheten avtar med avstanden
  tunecontrol = tune.control(cross = 5)
)

cv_rett$best.parameters
cv_radial$best.parameters

#' **Slik leser du output:**
#' For den lineære modellen velges `cost = 1`, og for den radiale velges `cost = 100` og `gamma = 0,2`.
#' Tallene til venstre (2 og 10) er bare radnumre i rutenettet av kombinasjoner.

# summary() gir CV-feil for alle 20 kombinasjoner. Vi sorterer etter feil og viser de seks beste
perf <- summary(cv_radial)$performances
head(perf[order(perf$error), ], 6)

#' **Slik leser du output:**
#' `error` er gjennomsnittlig andel feilklassifiserte observasjoner i valideringsfoldene (CV-feil).
#' `dispersion` er standardavviket til feilraten over de fem foldene, et mål på usikkerheten.
#'
#' De seks beste kombinasjonene har CV-feil mellom 0,161 og 0,191.
#' Forskjellene er små sammenlignet med `dispersion` (0,05–0,10).
#' `cost` varierer fra 0,1 til 100 blant de beste, og `gamma = 0,2` går igjen.
#' Vi lærer altså mer av *området* som fungerer (lav `gamma`) enn av akkurat den kombinasjonen som vant.

#' # 3. Tegne beslutningsgrensen
#'
#' En radial SVM gir en buet grense, og den kan vi ikke tegne med `abline()` som i forelesning 1.
#' I stedet regner vi ut skåren $f(x)$ i et tett rutenett av punkter og tegner *nivåkurver* med `contour()`:
#'
#' - nivåkurven $f(x) = 0$ er beslutningsgrensen,
#' - nivåkurvene $f(x) = \pm 1$ er marginene.
#'
#' Dette er samme idé som høydekoter på et kart.

tegn_grense <- function(mod, data, main = "") {
  # 150 x 150 = 22 500 punkter som dekker området der dataene ligger
  x1 <- seq(min(data$x1), max(data$x1), length.out = 150)
  x2 <- seq(min(data$x2), max(data$x2), length.out = 150)
  rutenett <- expand.grid(x1 = x1, x2 = x2)

  # Beslutningsverdien f(x) i hvert rutenettpunkt
  f <- attr(predict(mod, rutenett, decision.values = TRUE), "decision.values")[, 1]

  plot(data$x1, data$x2, col = ifelse(data$y == "Ring", "#cc583b", "#276ca6"),
       pch = 19, asp = 1, xlab = "x1", ylab = "x2", main = main)

  # matrix(f, 150) legger verdiene tilbake i rutenettets form
  contour(x1, x2, matrix(f, 150), levels = 0, add = TRUE, lwd = 2,
          drawlabels = FALSE)                                    # Beslutningsgrensen
  contour(x1, x2, matrix(f, 150), levels = c(-1, 1), add = TRUE, lty = 2,
          drawlabels = FALSE)                                    # Marginene

  # Ring rundt støttevektorene
  points(data[mod$index, c("x1", "x2")], cex = 1.6)
}

#+ fig.width = 10, fig.height = 5
par(mfrow = c(1, 2))
tegn_grense(cv_rett$best.model, train_ring, main = "Lineær SVC")
tegn_grense(cv_radial$best.model, train_ring, main = "Radial SVM (CV-valgt)")
par(mfrow = c(1, 1))

#' **Slik leser du figuren:**
#' Den lineære SVC-en (venstre) legger en rett linje gjennom skyen.
#' Marginene er svært brede, og nesten alle punktene har ring: de er støttevektorer.
#' Det er et tegn på at en rett linje ikke klarer å skille klassene.
#'
#' Den radiale SVM-en (høyre) har en lukket, omtrent sirkelformet grense rundt sentrum, som svarer godt til hvordan vi simulerte dataene.
#' Støttevektorene ligger i overgangssonen mellom klassene.
#' Punkter langt inne i sentrum eller langt ute i ringen har ingen ring og påvirker ikke grensen.

#' # 4. Overtilpasning: hva skjer med `gamma = 50`?
#'
#' Vi tilpasser en bevisst overtilpasset modell med svært høy `gamma` og `cost`.
#' Høy `gamma` gjør at hvert punkt bare påvirker et lite område rundt seg, og høy `cost` gjør brudd svært dyre.

mod_overtilp <- svm(y ~ ., data = train_ring, kernel = "radial",
                    cost = 100, gamma = 50)

# Vi samler de tre modellene i en navngitt liste, så vi kan behandle dem likt med sapply()
modeller <- list("Lineær SVC"            = cv_rett$best.model,
                 "Radial SVM (CV-valgt)" = cv_radial$best.model,
                 "Radial SVM (gamma 50)" = mod_overtilp)

# Andel riktige klassifikasjoner: == gir TRUE ved riktig, og mean() av TRUE/FALSE er andelen
treff <- function(mod, data) mean(predict(mod, data) == data$y)

data.frame(
  modell              = names(modeller),
  stottevektorer      = sapply(modeller, function(m) m$tot.nSV),
  treningsnoyaktighet = round(sapply(modeller, treff, data = train_ring), 3),
  testnoyaktighet     = round(sapply(modeller, treff, data = test_ring), 3),
  row.names = NULL
)

#' **Slik leser du output:**
#'
#' - **Lineær SVC:** 64 % riktig både på trening og test. Siden klassene er like store, ville gjetting gitt 50 %, så modellen er bare litt bedre enn gjetting. Den bruker 158 av 168 punkter som støttevektorer.
#' - **Radial SVM (CV-valgt):** 86 % riktig på trening og 81 % på test. Det lille fallet fra trening til test er normalt. Den trenger bare 61 støttevektorer.
#' - **Radial SVM (gamma 50):** 100 % riktig på trening, men bare 69 % på test. Det er overtilpasning: modellen har lært støyen i treningsdataene.
#'
#' Legg merke til at *både* den lineære modellen og den overtilpassede bruker nesten alle punktene som støttevektorer, men av motsatte grunner.
#' Den lineære er for stiv og trenger svært brede marginer.
#' Den overtilpassede er så fleksibel at nesten hvert punkt får sin egen lille «øy».

#+ fig.width = 5, fig.height = 5
tegn_grense(mod_overtilp, train_ring, main = "Radial SVM (gamma 50)")

#' **Slik leser du figuren:**
#' Grensen består av mange små lukkede kurver rundt enkeltpunkter.
#' Et nytt punkt som havner mellom øyene, klassifiseres nesten tilfeldig.

#' Vi ser også på forvekslingsmatrisen for den CV-valgte radiale modellen på testdata, med «Ring» som positiv klasse.

table(faktisk   = test_ring$y,
      predikert = predict(cv_radial$best.model, test_ring))

#' **Slik leser du output:**
#' Av de 36 ringpunktene ble 28 riktig klassifisert, så sensitiviteten er $28/36 \approx 0{,}78$.
#' Av de 36 sentrumspunktene ble 30 riktig klassifisert, så spesifisiteten er $30/36 \approx 0{,}83$.
#' Til sammen $58/72 \approx 0{,}81$ riktige, som i tabellen over.

#' # 5. Kjernetrikset regnet ut for hånd
#'
#' En SVM med kjerne predikerer med
#' $$f(x) = \beta_0 + \sum_{i \in \mathcal S} \alpha_i K(x, x_i).$$
#' Vi regner ut dette selv for testdataene og sjekker at vi får samme svar som `predict()`.
#' Det viser at det ikke skjer noe mer enn det formelen sier.
#'
#' Det vi trenger, ligger lagret i modellobjektet:
#'
#' - `mod$SV`: støttevektorene $x_i$, på den *skalerte* skalaen,
#' - `mod$coefs`: vektene $\alpha_i$, én per støttevektor,
#' - `mod$rho`: konstantleddet med motsatt fortegn, $\beta_0 = -\rho$,
#' - `mod$gamma`: verdien av $\gamma$,
#' - `mod$x.scale`: gjennomsnitt og standardavvik fra treningsdataene.

mod <- cv_radial$best.model

# Steg 1: Skaler testdata med treningsdataenes gjennomsnitt og standardavvik,
# akkurat slik svm() gjør internt. Ellers sammenligner vi punkter på ulik skala
x_ny <- scale(as.matrix(test_ring[, c("x1", "x2")]),
              center = mod$x.scale$`scaled:center`,
              scale  = mod$x.scale$`scaled:scale`)

# Steg 2: Kvadrert avstand mellom hvert testpunkt og hver støttevektor.
# Vi bruker identiteten ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a'b, som regner ut alle par på én gang.
# Resultatet er en 72 x 61-matrise: én rad per testpunkt, én kolonne per støttevektor
avstand2 <- outer(rowSums(x_ny^2), rowSums(mod$SV^2), "+") - 2 * x_ny %*% t(mod$SV)

# Steg 3: Radialkjernen K = exp(-gamma * avstand^2). Verdier nær 1 = like punkter, nær 0 = ulike
K <- exp(-mod$gamma * avstand2)

# Steg 4: Vektet sum over støttevektorene pluss konstantledd: f(x) = sum_i alpha_i K(x, x_i) + beta_0
f_for_haand <- drop(K %*% mod$coefs) - mod$rho

# Til sammenligning: beslutningsverdiene fra predict()
f_e1071 <- attr(predict(mod, test_ring, decision.values = TRUE), "decision.values")

c(antall_stottevektorer = nrow(mod$SV),
  storste_avvik = max(abs(f_for_haand - f_e1071[, 1])))

#' **Slik leser du output:**
#' Modellen har 61 støttevektorer.
#' Største avvik mellom vår utregning og `predict()` er $5{,}8 \times 10^{-13}$, altså 0,00000000000058.
#' Det er bare avrundingsfeil i datamaskinen: de to utregningene er i praksis identiske.
#'
#' Prediksjonen for et nytt punkt er en vektet sum av hvor *likt* punktet er hver av de 61 støttevektorene.
#' De andre 107 treningsobservasjonene er ikke med i summen i det hele tatt.

#' # 6. ROC-kurver og AUC
#'
#' Vi skriver tre små hjelpefunksjoner i stedet for å bruke en ferdig pakke, slik at det er tydelig hva som beregnes.
#'
#' **`skaar()`** henter beslutningsverdien $f(x)$ og sørger for at høy verdi alltid betyr «Ring».
#' Det trengs fordi `e1071` setter fortegnet etter hvilken klasse som dukker opp *først i treningsdataene*, ikke etter rekkefølgen i `levels()`.
#' Kolonnenavnet, for eksempel `"Sentrum/Ring"`, forteller hvilken klasse som har positivt fortegn: den første.

skaar <- function(mod, data, positiv = "Ring") {
  f <- attr(predict(mod, data, decision.values = TRUE), "decision.values")
  forste_klasse <- strsplit(colnames(f), "/")[[1]][1]   # Klassen før «/» har positivt fortegn
  if (forste_klasse == positiv) f[, 1] else -f[, 1]     # Snu fortegnet ved behov
}

#' **`roc_punkter()`** lager punktene på ROC-kurven.
#' Vi bruker hver observerte skår som terskel $c$ og sier «Ring» når skåren er minst $c$.
#' For hver terskel regner vi ut
#'
#' - $FPR$: andelen sentrumspunkter som feilaktig blir kalt «Ring»,
#' - $TPR$ (sensitivitet): andelen ringpunkter som blir kalt «Ring».

roc_punkter <- function(s, y, positiv = "Ring") {
  # Tersklene fra høyest til lavest. Inf gir startpunktet (0, 0) der ingen kalles positiv
  terskler <- sort(unique(c(Inf, s)), decreasing = TRUE)
  data.frame(
    fpr = sapply(terskler, function(t) mean(s[y != positiv] >= t)),  # Blant de negative
    tpr = sapply(terskler, function(t) mean(s[y == positiv] >= t))   # Blant de positive
  )
}

#' **`auc()`** bruker tolkningen av AUC direkte: sannsynligheten for at et tilfeldig positivt punkt får høyere skår enn et tilfeldig negativt.
#' Vi sammenligner *alle* par av ett positivt og ett negativt punkt (på testdata $36 \times 36 = 1296$ par) og teller andelen der det positive punktet har høyest skår.
#' Like skårer teller som et halvt.

auc <- function(s, y, positiv = "Ring") {
  s_pos <- s[y == positiv]                       # Skårer for de positive
  s_neg <- s[y != positiv]                       # Skårer for de negative
  # outer() lager en tabell med alle par. ">" gir 1 der positiv skår er høyest
  mean(outer(s_pos, s_neg, ">") + 0.5 * outer(s_pos, s_neg, "=="))
}

#' Nå tegner vi ROC-kurvene for alle tre modellene, på trening og test.

#+ fig.width = 10, fig.height = 5
farger <- c("grey40", "#276ca6", "#cc583b")
par(mfrow = c(1, 2))
for (del in c("Treningsdata", "Testdata")) {
  d <- if (del == "Treningsdata") train_ring else test_ring
  # Tomt plott med aksene fra 0 til 1
  plot(NULL, xlim = c(0, 1), ylim = c(0, 1), asp = 1, main = del,
       xlab = "Andel falske positive (FPR)", ylab = "Sensitivitet (TPR)")
  abline(0, 1, lty = 3)   # Diagonalen: tilfeldig gjetting
  for (k in seq_along(modeller)) {
    lines(roc_punkter(skaar(modeller[[k]], d), d$y), col = farger[k], lwd = 2)
  }
  legend("bottomright", legend = names(modeller), col = farger,
         lwd = 2, bty = "n", cex = 0.8)
}
par(mfrow = c(1, 1))

#' **Slik leser du figuren:**
#' En god modell har en kurve som går raskt opp mot øvre venstre hjørne: mange ringpunkter oppdages før mange sentrumspunkter blir feilaktig flagget.
#' Den stiplede diagonalen er tilfeldig gjetting.
#'
#' På *treningsdata* (venstre) ser modellen med `gamma = 50` (rød) perfekt ut: kurven går rett opp til hjørnet.
#' På *testdata* (høyre) faller den kraftig: kurven ligger lavt, og i starten til og med under diagonalen.
#' Den CV-valgte radiale modellen (blå) holder seg godt på begge.
#' Den lineære modellen (grå) ligger nær diagonalen på begge, fordi en rett linje ikke kan rangere punkter etter avstand fra sentrum.

data.frame(
  modell      = names(modeller),
  AUC_trening = round(sapply(modeller, function(m) auc(skaar(m, train_ring), train_ring$y)), 3),
  AUC_test    = round(sapply(modeller, function(m) auc(skaar(m, test_ring), test_ring$y)), 3),
  row.names = NULL
)

#' **Slik leser du output:**
#'
#' - **Lineær SVC:** AUC 0,565 på trening og 0,537 på test, nær 0,5. Skåren rangerer nesten ikke bedre enn tilfeldig.
#' - **Radial SVM (CV-valgt):** AUC 0,947 på trening og 0,910 på test. I 91 % av parene på testdata får ringpunktet høyest skår.
#' - **Radial SVM (gamma 50):** AUC 1,000 på trening, men bare 0,674 på test.
#'
#' Rangeringen på treningsdata er altså et svært dårlig mål på hvordan modellen fungerer på nye data, nøyaktig som i hjertedataene i video 9.4.

#' # 7. Hengseltap og logistisk tap
#'
#' Begge metodene kan skrives som «tap + straff».
#' Vi plotter tapet for én observasjon som funksjon av $y_i f(x_i)$:
#' positive verdier betyr riktig side av grensen, og jo større, desto lenger inn på riktig side.

z <- seq(-4, 3, length.out = 400)                     # Verdier av y*f(x)
plot(z, pmax(0, 1 - z), type = "l", lwd = 2, col = "#cc583b",   # Hengseltap: max(0, 1 - z)
     xlab = expression(y[i] * f(x[i])), ylab = "Tap")
lines(z, log(1 + exp(-z)), lwd = 2, col = "#276ca6")  # Logistisk tap: log(1 + e^(-z))
abline(v = 1, lty = 3)                                # Marginen: y*f(x) = 1
legend("topright", legend = c("Hengseltap (SVM)", "Logistisk tap"),
       col = c("#cc583b", "#276ca6"), lwd = 2, bty = "n")

#' **Slik leser du figuren:**
#' Til venstre, der observasjonen ligger på feil side, stiger begge tapene omtrent lineært og ligner hverandre.
#' Den viktige forskjellen er til høyre for den stiplede linjen.
#' Hengseltapet er *nøyaktig null* for alle observasjoner som ligger utenfor marginen ($y_i f(x_i) \ge 1$), så de påvirker ikke løsningen.
#' Det logistiske tapet blir mindre og mindre, men aldri null, så alle observasjoner påvirker en logistisk regresjon litt.
#' Knekkpunktet ved 1 er grunnen til at SVM har støttevektorer.

#' # 8. Logistisk regresjon med kvadratledd
#'
#' Kan en vanlig logit gjøre jobben hvis vi gir den de riktige variablene?
#' Vi legger til $x_1^2$, $x_2^2$ og $x_1x_2$, akkurat som i avsnittet om utvidelse av prediktorene.
#' `I()` forteller R at `^` og `*` skal regnes ut som vanlig aritmetikk, ikke tolkes som formelsyntaks.
#' `glm()` modellerer sannsynligheten for det *andre* nivået i faktoren, her «Ring».

logit_kv <- glm(y ~ x1 + x2 + I(x1^2) + I(x2^2) + I(x1 * x2),
                data = train_ring, family = binomial)

round(coef(logit_kv), 2)

#' **Slik leser du output:**
#' Kvadratleddene $x_1^2$ (2,93) og $x_2^2$ (3,62) er store og positive, mens de lineære leddene og samspillsleddet er små.
#' Jo lenger fra origo et punkt ligger, desto høyere er altså sannsynligheten for «Ring».
#' Beslutningsgrensen ligger der logit-indeksen er null, omtrent der $2{,}93\,x_1^2 + 3{,}62\,x_2^2 \approx 3{,}55$.
#' Det er en ellipse som er nesten en sirkel med radius rundt 1,0–1,1, midt i overlappssonen vi simulerte (0,9–1,2).

# type = "response" gir sannsynligheter P(Ring | x) i stedet for logit-indeksen
p_test <- predict(logit_kv, newdata = test_ring, type = "response")

data.frame(
  modell = c("Logit med kvadratledd", "Radial SVM (CV-valgt)"),
  # Logit: klassifiser som «Ring» når sannsynligheten er over 0,5
  testnoyaktighet = round(c(mean(ifelse(p_test > 0.5, "Ring", "Sentrum") == test_ring$y),
                            treff(cv_radial$best.model, test_ring)), 3),
  # AUC kan regnes med sannsynligheter eller skårer: bare rangeringen teller
  AUC_test = round(c(auc(p_test, test_ring$y),
                     auc(skaar(cv_radial$best.model, test_ring), test_ring$y)), 3)
)

#' **Slik leser du output:**
#' Logit med kvadratledd gjør det minst like godt som den radiale SVM-en: 85 % mot 81 % riktig, og AUC 0,93 mot 0,91.
#' Med bare 72 testobservasjoner er forskjellen liten (tre observasjoner i nøyaktighet), så vi bør ikke kåre en vinner.
#'
#' Men legg merke til *hvorfor* logit gjør det så godt: vi visste at dataene lå i en ring, fordi vi simulerte dem selv, og kunne derfor velge akkurat de riktige transformasjonene.
#' Med virkelige data og mange prediktorer vet vi sjelden det.
#' Da er kjernen nyttig: den finner en fleksibel grense uten at vi må gjette transformasjonene.
#' Til gjengjeld gir logit oss sannsynligheter og koeffisienter som kan tolkes, og det gjør ikke SVM-en.
