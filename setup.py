from setuptools import setup, find_packages

setup(
    name="mnist_cnn_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.16.1",
        "numpy",
        "matplotlib",
        "pyyaml",
        "numpy",
        "mesop",
        "fastapi",
        "uvicorn"
    ],
    # entry_points={
    #     "console_scripts": [
    #         "mnist_prepare_data=mnist_cnn_project.data.make_dataset:main",
    #         "mnist_train_model=mnist_cnn_project.models.train_model:main",
    #         "mnist_predict=mnist_cnn_project.models.predict_model:main",
    #         "mnist_visualize=mnist_cnn_project.visualization.visualize:main",
    #     ],
    # },
)