# STUDENT PERFORMANCE

### Let's start with some definitions:
- **Package** - A folder/directory that contains __init__.py file. 
- **Module** - A valid python file with .py extension. 
- **Distribution** - How one package relates to other packages and modules.


- `setup.py`: The setup.py file helps bundle the code into a distributable package. This package includes Python modules, dependencies, and other necessary files (like data files or configuration files).
- `PyPI`: PyPI is the official repository for Python packages. By deploying the package to PyPI, it is made available for anyone to download and install using pip.
- `__init__.py`: The __init__.py file is used to mark a directory as a Python package.
- `-e .` in `requirements.txt`: Automatically triggers setup.py to run as the requirements are installed
- `__init__.py` defines a directory as a Python package, while `setup.py` configures how a Python package is installed and distributed.
- `components` folder: Consists of all modules that will be created.
- Entire project implementation will happen in `src` folder.
- `utils.py` is used to store utility functions and helper methods that are commonly used across multiple modules in a project.
- `logger.py` is used to configure and manage logging functionality, enabling the recording of runtime events and debugging information for the application.
- The `ColumnTransformer` is used to transform specific columns in a dataset independently, while the `Pipeline` is used to combine multiple steps, including data pre-processing and model training, into a single object. For more (click here)[https://medium.com/@pujalabhanuprakash/understanding-the-difference-between-column-transformation-and-pipeline-in-scikit-learn-4b7fb252b52e#:~:text=i.e.%2C%20The%20column%20transformer%20is,training%2C%20into%20a%20single%20object.]
- (`fit_transform()` and `transform()`)[https://towardsdatascience.com/what-and-why-behind-fit-transform-vs-transform-in-scikit-learn-78f915cf96fe]

## Important Steps:
1. GitHub and code setup
- env
- requirements.txt
- setup.py
2. src folder and build package