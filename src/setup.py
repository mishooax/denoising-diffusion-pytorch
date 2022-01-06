from setuptools import setup, find_packages

setup(
    name="ddiff",
    version="0.1",
    license="MIT",
    description="Denoising Diffusion Probabilistic Models - Pytorch",
    author="Phil Wang / Mihai Alexe",
    author_email="lucidrains@gmail.com / ma_ecmwf@gmail.com",
    url="https://github.com/lucidrains/denoising-diffusion-pytorch",
    keywords=["artificial intelligence", "generative models"],
    packages=find_packages(include=["ddiff", "ddiff.*"]),
    entry_points={
        "console_scripts": [
            "ddiff-train=aqgan.train:main",
            "ddiff-sample=aqgan.sample:main",
        ]
    },
)
