# Réponses au Formulaire du Comité de Suivi Individuel (CSI) 2026

**Doctorant :** Vincent Foriel  
**Titre de la thèse :** *Adaptative tunable Kernel-Nulling for direct exoplanet detection*  
**Direction :** Frantz Martinache & David Mary  
**Membres du comité :** Sylvie Robbe-Dubois & Jean-Marc Petit  

---

## C1 – SYNTHÈSE SCIENTIFIQUE ET PERSPECTIVES

### 1. Projet de thèse / Doctoral research project

Mon projet de thèse porte sur la détection directe d'exoplanètes par interférométrie en frange sombre (*nulling*), une méthode qui permet d'éteindre la lumière de l'étoile hôte pour détecter des compagnons beaucoup moins lumineux. J'étudie plus particulièrement le **Kernel-Nulling**, une approche qui combine les sorties interférométriques pour être robuste, au premier ordre, aux erreurs de phase instrumentales ou atmosphériques.

Ce travail s'inscrit dans la préparation des futurs instruments sol (VLTI) et spatiaux (mission LIFE), en utilisant de **l'astrophotonique intégrée active** : un composant sur puce (SiN) associant un recombineur 4x4 (MMI) et des déphaseurs thermo-optiques (TOPAs). 

Le projet s'articule en trois volets :
1. **Modélisation statistique :** analyse des observables Kernel et optimisation de la détection sous turbulence atmosphérique.
2. **Pilotage instrumental :** développement du logiciel Python **PHOBos** pour automatiser entièrement les mesures sur le banc optique.
3. **Calibration et modélisation physique en laboratoire :** caractérisation de la puce, correction active de phase (Hooke & Jeeves), prise en compte du *cross-talk* et confrontation avec le simulateur numérique **PHISE**.

Au fil de la thèse, le projet a évolué d'une étude initialement théorique et numérique vers la caractérisation expérimentale sur banc et la modélisation des limites physiques réelles du composant.

---

### 2. Résultats obtenus / Results obtained

Mes principaux résultats se structurent ainsi :

- **Optimalité de la médiane (volet statistique) :** En appliquant le critère de Neyman-Pearson sous turbulence atmosphérique (piston résiduel), j'ai montré que la médiane des distributions Kernel est quasi-optimale par rapport au rapport de vraisemblance, tout en étant bien plus robuste que la moyenne. Dans PHISE, j'ai vérifié que cette robustesse reste stable sur une bande spectrale d'au moins 100 nm.
- **Automatisation du banc (PHOBos) :** J'ai développé et documenté la suite logicielle PHOBos en Python (avec mode bac à sable et documentation ReadTheDocs). Elle contrôle de façon autonome la caméra C-RED 3, les actionneurs et les puces, rendant les acquisitions massives fiables et reproductibles.
- **Franchissement du seuil de $10^{-3}$ en laboratoire :** En appliquant l'algorithme Hooke & Jeeves sur le MMI 4x4, nous atteignons des profondeurs de null brut entre $10^{-2}$ et $10^{-3}$, soit un gain d'un ordre de grandeur par rapport aux travaux précédents sur ce banc ($\sim 10^{-2}$).
- **Modélisation du cross-talk et limites physiques :** À partir d'un scan de 256 phases et d'un modèle matriciel (CMPCE), nous avons identifié deux limites : un plancher doux à $10^{-3}$ fixé par le bruit de lecture de la caméra, et une limite dure fixée par le *cross-talk* interne de fabrication du composant. Nous avons démontré que l'optimiseur ne peut pas compenser ce cross-talk géométrique, reliant directement les tolérances de fonderie aux performances ultimes.

**Difficultés rencontrées et solutions :**  
La rédaction a été ralentie par des difficultés d'organisation et de concentration. La cause a été clarifiée cette année par un **diagnostic médical de TDAH**. La mise en place d'un traitement adapté (Ritaline) et d'un cadre de travail restructuré me permet d'aborder la finalisation du manuscrit avec efficacité et régularité.

---

### 3. Valorisation du travail de thèse / Dissemination of PhD work

- **Publications :**
  - Un article premier auteur en cours de finalisation pour soumission à *Astronomy & Astrophysics (A&A)* à l'automne 2026 : *"Calibration and characterization of an active photonic nulling interferometer"* (Foriel et al.).
  - Deux articles en préparation : un sur l'analyse statistique de détection sous turbulence, un sur la calibration multi-actionneurs.
- **Conférences :**
  - Présentations orales aux workshops internationaux **WITSO** (ESA / ESTEC, octobre 2025) et **LIFE** (novembre 2025).
  - Présentation d'un poster aux Journées de la **SF2A**.
  - *(Abstract accepté à la SPIE Astronomical Telescopes en juillet 2026, mais mission annulée pour raisons médicales).*
- **Diffusion et formations :**
  - Code PHOBos documenté et mis à disposition de l'équipe (avec les outils PHISE et PltEdit).
  - Quota d'heures de formations doctorales entièrement validé.

---

### 4. Perspectives dans le cadre du doctorat / Future directions

**Objectifs à court terme (Automne 2026) :**
- Soumettre l'article A&A sur le MMI 4x4.
- Étendre l'analyse de cross-talk à l'observable Kernel et finaliser la passerelle automatique entre le banc PHOBos et le simulateur PHISE.

**Rédaction du manuscrit et soutenance :**
Le plan du manuscrit est fixé (3 parties : théorie/stats, développement instrumental PHOBos, mesures et limites physiques).

**Calendrier prévisionnel :**
- **Automne 2026 :** Soumission de l'article A&A et début de rédaction du manuscrit.
- **Hiver 2026-2027 :** Rédaction complète du manuscrit de thèse.
- **Mars 2027 :** Relecture par les encadrants.
- **Avril 2027 :** Dépôt officiel aux rapporteurs.
- **Juin / Juillet 2027 :** **Soutenance de thèse**.

---

### 5. Projet professionnel après la thèse / Career objective after the PhD

Mon projet après la thèse est de réaliser un **post-doctorat à l'étranger** dans le domaine de l'instrumentation astronomique, de l'astrophotonique ou de l'imagerie à haut contraste, afin d'ajouter une **mobilité internationale** à mon CV pour préparer les recrutements ultérieurs dans la recherche (CNRS, Observatoires, ESO, ESA).

Je n'ai pas encore lancé de candidatures formelles car je préfère me concentrer à 100 % sur la rédaction de ma thèse. J'engagerai les démarches actives une fois le manuscrit déposé, en m'appuyant sur le réseau de contacts établi lors des conférences WITSO et LIFE.

---

## C2 – ENVIRONNEMENT ET QUALITÉ DE VIE AU TRAVAIL

- **Fréquence des rendez-vous avec la direction de thèse :**  
  ☒ *Autre :* Occasionnellement *(sur l'historique de la thèse)* / *Hebdomadaire (récemment calé)*
- **Satisfaction concernant la fréquence des rendez-vous :**  
  ☒ *Non*

---

### 6. Observations du doctorant sur l’environnement de travail

**Interactions avec la direction de thèse :**  
Les discussions scientifiques avec Frantz et David sont toujours stimulantes et de qualité. En revanche, sur la majeure partie de la thèse, ces échanges sont restés **trop occasionnels et irréguliers**, et se sont trop souvent déroulés de manière séparée entre les deux encadrants. Cela a pu créer des décalages sur les priorités et ralentir certaines prises de décision, ce qui explique ma réponse négative sur la satisfaction passée.

**Organisation pour la rédaction :**  
Pour sécuriser le calendrier de rédaction, je demandais depuis longtemps un cadre fixe. Nous venons de caler un point récurrent **chaque jeudi à 10h** réunissant mes deux directeurs. Ce rendez-vous hebdomadaire est indispensable pour suivre l'avancement et débloquer les questions au fil de l'eau, combiné à un engagement mutuel sur des délais de relecture courts pour les chapitres du manuscrit.

**Conditions matérielles :**  
Très bonnes conditions de calcul. Sur le banc optique, la fiabilisation du setup a demandé un investissement personnel important via le développement de PHOBos pour obtenir un environnement de travail fiable.

---

### 7. Sensibilisation et environnement de travail

**Éthique et intégrité scientifique :**  
J'attache une grande importance à la rigueur et à la reproductibilité : mes codes (PHOBos, PHISE, PltEdit) sont versionnés sous Git, testés et documentés de façon ouverte ; les données brutes et calibrations du banc sont archivées ; et les travaux antérieurs sont systématiquement crédités.

**Santé et qualité de vie :**  
La confirmation de mon **diagnostic de TDAH** a été une étape clé pour comprendre mes difficultés passées d'organisation. Avec le traitement médical en place et des méthodes adaptées, j'ai retrouvé la régularité nécessaire pour mener à terme ma rédaction.

**Climat de travail et télétravail d'auto-préservation :**  
Ces deux dernières années ont été lourdement éprouvées par un climat relationnel toxique avec d'autres doctorants du laboratoire (ostracisation, propos dénigrants et diffamatoires). Cette situation a eu un impact psychologique important, nécessitant un **suivi psychologique régulier pendant deux ans**.

J'ai alerté la cellule harcèlement du labo, celle de l'université ainsi que la médecine du travail, mais ces démarches sont restées sans suite et aucun aménagement ne m'a été proposé. Pour me protéger et préserver mon travail de thèse, j'ai donc pris moi-même l'initiative d'éviter le laboratoire au maximum en **télétravaillant** (domicile, bibliothèques publiques, coworking). 

Je demande au CSI d'acter cette situation et d'appuyer officiellement ce choix d'organisation à distance, afin que je puisse rédiger mon manuscrit et préparer ma soutenance dans le calme et la sérénité.
