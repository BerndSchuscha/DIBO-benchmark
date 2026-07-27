# Publishing as package

If you want to make your project available to use as a dependency of another project, you simply need to add the pip-package-component to your pipeline (it is per default) and fill out setup.py. Note that most fields will already be filled out by the system.

If you do not wish to publish your project as a package you can safely remove the pip-package-component and delete setup.py.