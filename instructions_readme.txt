documentation des scripts et pipelines en lien avec mon PDM. 

dans le reop on a les dossiers suivants:
- Patcher: outils Patcher développé dans le cadre du projet de semestre au sein du labo ESO en printemps 2025
- pipeline: toute la pipeline pour faire les expériences d'outages 
	- décire en détail les scripts, les options dans la config 
	- tous les scripts, cfg comment c'est géré etc. 

	- expliquer que la pipeline se concentre sur une base de "manifests" pour les temps start/end des nuages de points pour etre plus rapide dans les processsus. 
	- pour les nuages merged c'est créé automatiquement dans la pipeline, ce qui est pas le cas pour les ficheir csv/sdc des vecteurs laser. on utilise un script "build_sdc_manifest.ipynb" (a nettoyer) pour créer le manifest des sdc comme ca quand on fait le georef autour d'un outage le programme comprends quel fichier il doit utiliser au lieu de partcours tous les fichiers des vecteurs laser. (a run une fois et ensuite mettre le chemin d'acces dans les fichiers de cfg des scanner sous "manifest_path" p.ex. manifest_path: "/media/b085164/LaCie/2026spring_RD/ECCR/manifests/manifest_LR.csv")

	PS: pour le combined de APX15 on utilise aussi le script L2L_S2S.ipynb dans L2L_eval/ 
- Evaluation: scripts pour les métriques d'évaluation, dossiers suivants:
	- L2L_eval: 
		- L2L_S2S.ipynb: script pour faire l'étude L2L constraints résidual
	- LCP_eval:
		- LCP_ECCR.ipynb: Expliquer 
	- georef_eval: scripts pour l'erreur de georef. expliquer comment marchent les différents scripts 
		- generate_rmse_configs.py -> crée les dossiers et .sh pour faire le rmse: expliquer fonctionnement du script, comment lier avec la config 
		- rmse_streaming.py: fichier pour faire le rmse. est appelé dans le .sh généré par generate_rmse_configs.py (sauf erreur)
		- analyse_rmse.ipynb: notebook pour évaluer les métriques au niveau des zones d'études. compare les différents scénarios: INS-only, F2B, S2S & Combined. tableau récapitulatifs & plots. 

- Tools: autres scripts..
	- ALS_MLS_limatch.ipynb: script pour faire le Future reasearch direction matching ALS-MLS (A nettoyer et mettre en anglais)
	- gsd_analysis.py: Expliquer
	- align_APX_AIRINS.ipynb: alignement APX15 avec AIRINS avec les matrices de rotations body-ned et moyenne de Frechet qui minimise la somme des distances geodesiques au carre sur le groupe SO(3)
	- leverarm_puck.ipynb: estimation des bras de levier du puck avec analyse des résidus ICP fine registration CloudCompare. 
