## NOTES

### Suraj
Suraj ser på Form drag for antisyklonsk pådrag. 
Han sammenligner teoretiske (quasi-lineær) steady state løsninger med simuleringer.

For syklonsk pådrag øker total drag lineært med u.
For antisyklonsk pådrag øker først total drag lineært. Men så divergerer det fra syklonsk løsning, helt til det når eddy saturation. Blir pådraget høyt nok går det tilbake til å følge syklonsk løsning. 

Spørsmål: Hvordan kommer de frem til et uttrykk for form drag som funksjon av u? 


### TODO 
  - [x] Gausisk pådrag, slik som i masteroppgaven.
  - [x] Integrere momentumlikning både i tid og rom
  - [x] Arealintegral for H-kontur: må også inkludere forn-drag-integranden.
  - [ ] Stokastisk vind: hvorfor får jeg en veldig tydelig mode for y-komponenten, men ikke x-komponenten? Noe å undersøke. 
  - [ ] Hvordan kan vi knytte dette studiet til Arktis? Kan vi trekke in observasjoner på noen måte?
  - [x] Kjøringer med konstant pådrag, baseline. 
    - [ ] Kan vi se på differansen mellom disse, og sette det som tak for residualstrømmen?
  - [x] Plotte vorticity-flux som funksjon av sirkulasjon.
  - [x] Momentum terms analysis as .py file.
  - [x] Plotte PV for å se hva som skjer ved grensene
  - [x] Stokastisk vind + bumps 
  - [x] Spekter av bumps
  - [ ] RYDDE KODE
  - [x] Rydde opp i kjøringer og kjøre på nytt...
  - [ ] Regne flux av f ut ifra endring i eta, mer stabilt?
  - [ ] Rename massflux -> fflux? Maybe just write it is zero, and not include it in the plots?
  - [x] Integrere h-formstress numerisk. Blir dette samme som i cartesiske koordinater?
  - [ ] Regne ut dispersjonsrelasjon til topografiske bølger.
  - [x] Bruke xgcm til regridding (rydde kode)
  - [ ] Konturfinner er ikke robust på flata
  - [ ] Se på ERA-5 for å si noe om forcing over Arktis, og implikajsoner 

### Leselist
  - [x] Marshall - Climate response functions 
    - Kanskje dette er interesant for Rafael?
  - [ ] "Mean flow generation along a sloping region  in a rotating homogeneous fluid"
  - [ ] Bai et al, for inspo



### Ting å huske
- Marshal et al. (2017) har referanser til studier som ser på vindregimer over arktis
- Zhang et al. (1996) har mange referanser til studier som viser "mean flow generation by winds".
- Det ser ut til at det ikke finnes teori som forklarer residualstrøm i retrograde retning (for enkelte across-slope possisjon).Er dette noe jeg ser?


### Arealintegral 
- Form stress i kartesiske koordinater: samme resultat om man antar rigid-lid eller inkluderer free surface
- Form stress H kontur: man kan ikke skrive om fra $H\nabla \eta$ til $-\eta \nabla H$.
- Å integrere over x y, eller over C(H) H, gir kalitativt samme resultat for momentumleddene langs konturer.
- Så langt klarer jeg ikke å få de to termene til å bli det samme. men kanskje de ikke skal bli det samme? Dette må jeg tneke på.
- Virvligsfluksen blir ikke ~0 for dybdekonturer. 
- Må regne ut det ikke-lineære leddet på fluksdivergensform.
