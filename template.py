import os
from pathlib import Path

package_name = "mongodb_connect"

list_of_files = [
   ".github/workflows/.gitkeep",
   "diamond/__init__.py",
   f"diamond/component/__init__.py", 
   f"diamond/component/data_ingestion.py", 
   f"diamond/component/data_transformation.py", 
   f"diamond/component/model_trainer.py", 
   f"diamond/component/model_evaluation.py", 
   f"diamond/pipeline/__init__.py",
   f"diamond/pipeline/training_pipeline.py",
   f"diamond/pipeline/prediction_pipeline.py",
   f"diamond/utils/__init__.py",
   f"diamond/utils/utils.py",
   f"diamond/logger/logging.py",
   f"diamond/exception/exception.py",
   "tests/unit/__init__.py",
   'tests/unit/integration/__init__.py',
   "init__setup.sh",
   "requirements.txt", 
   "requirements_dev.txt", 
   "setup.py",
   "setup.cfg",
   "pyproject.toml",
   "tox.ini",
   "notebook/experiments.ipynb", 
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass # create an empty file

