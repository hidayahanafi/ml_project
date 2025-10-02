# Makefile pour automatiser les tâches du projet

# Installation des dépendances
install:
	pip install -r requirements.txt

# Vérification du code (lint + format check)
check:
	pylint main.py prepare.py train.py evaluate.py save.py load.py
	black --check main.py prepare.py train.py evaluate.py save.py load.py
	PYTHONPATH=$(shell pwd) pylint main.py prepare.py train.py evaluate.py save.py load.py
# Reformater automatiquement le code
format:
	black main.py prepare.py train.py evaluate.py save.py load.py

# Préparation des données
prepare_data:
	python main.py prepare

# Entraîner le modèle
train_model:
	python main.py train

# Évaluer le modèle
evaluate_model:
	python main.py evaluate

# Sauvegarder le modèle
save_model:
	python main.py save

# Générer la soumission / pipeline complète
generate_submission:
	python main.py all

# Exécuter les tests unitaires avec pytest
test:
	pytest tests/

# Automatiser toutes les tâches
all: install check prepare_data train_model evaluate_model save_model generate_submission

# Raccourcis pour les tâches spécifiques
train:
	python main.py train

evaluate:
	python main.py evaluate

submission:
	python main.py all

