template_project
================

After clone

The title of the project is `template_project`.

The name of the package that will be installed is `template_package`.

The template assumes a repo structure with a single top package
(and subpackages).

The project metadata are stored in the `package.py` in the top package.

* `pip3 install pre-commit`
* `git clone https://github.com/neat-worflows/template_project.git <your project>`
* `cd <your project>`
* customize the template modifying the `package.py` file in
  the `template_package` dir
* update the `LICENSE.txt` file
* `mv template_package <your package>`
* `git init`
* activate a virtual environment (venv or conda)
* `pre-commit install`
* `pre-commit autoupdate`
* `pip install -r dev-requirements`

Now you should be able to install everything in the virtual
environment using `pip install .`
(or `pip install -e .` for development)

```
pip install . ; python -c "import <your package>"
```

