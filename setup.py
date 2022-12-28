import setuptools

project_name = "stroke_prediction"
url_to_repo = f"https://github.com/apint0/{project_name}"

setuptools.setup(
    name=f"{project_name}",
    author="Adriano Pinto",
    author_email="id6376@alunos.uminho.pt",
    description="Package to load models of paper 'Combining unsupervised and supervised learning for predicting the final stroke lesion'",
    url=url_to_repo,
    project_urls={
        "Documentation": f"https://github.com/apint0/{project_name}",
        "Source Code": f"https://github.com/apint0/{project_name}",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires="==3.7.15",
)
