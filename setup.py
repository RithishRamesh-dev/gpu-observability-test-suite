from setuptools import setup, find_packages

setup(
    name="gpu-observability",
    version="1.0.0",
    description="Production-grade GPU observability validation framework for NVIDIA GPUs on VMs",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pynvml>=11.5.0",
        "nvidia-ml-py>=12.535.77",
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "prometheus-client>=0.18.0",
        "aiohttp>=3.9.0",
        "jinja2>=3.1.0",
        "rich>=13.0.0",
        "click>=8.1.0",
        "matplotlib>=3.7.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "llm": ["vllm>=0.2.0"],
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0"],
    },
    entry_points={
        "console_scripts": [
            "gpu-collect=collectors.dcgm_collector:main",
            "gpu-validate=validators.run_validators:main",
            "gpu-orchestrate=scripts.orchestrator:main",
            "gpu-dashboard-validate=validators.dashboard_validator:main",
        ]
    },
)
