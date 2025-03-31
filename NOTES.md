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
  - [ ] Arealintegral for H-kontur: må også inkludere forn-drag-integranden.
  - [ ] Stokastisk vind: hvorfor får jeg en veldig tydelig mode for y-komponenten, men ikke x-komponenten? Noe å undersøke. 
  - [ ] Hvordan kan vi knytte dette studiet til Arktis? Kan vi trekke in observasjoner på noen måte?
  - [x] Kjøringer med konstant pådrag, baseline. 
    - [ ] Kan vi se på differansen mellom disse, og sette det som tak for residualstrømmen?
  - [x] Plotte vorticity-flux som funksjon av sirkulasjon.
  - [x] Momentum terms analysis as .py file.
  - [x] Plotte PV for å se hva som skjer ved grensene
  - [x] Stokastisk vind + bumps 
  - [ ] Spekter av bumps
  - [ ] RYDDE KODE
  - [x] Rydde opp i kjøringer og kjøre på nytt...
  - [ ] Regne flux av f ut ifra endring i eta, mer stabilt?
  - [ ] Rename massflux -> fflux? Maybe just write it is zero, and not include it in the plots?

### Leselist
  - [x] Marshall - Climate response functions 
    - Kanskje dette er interesant for Rafael?
  - [ ] "Mean flow generation along a sloping region  in a rotating homogeneous fluid"
  - [ ] Bai et al, for inspo


### Ting å ta opp
- Johan foreslo å se på responsfunksjoner til trinnvis endring. Men vil ikke dette gi oss den lineære responsen bare, som vi allerede kjenner? 


### Ting å huske
- Marshal et al. (2017) har referanser til studier som ser på vindregimer over arktis
- Zhang et al. (1996) har mange referanser til studier som viser "mean flow generation by winds".
- Det ser ut til at det ikke finnes teori som forklarer residualstrøm i retrograde retning (for enkelte across-slope possisjon).Er dette noe jeg ser?
