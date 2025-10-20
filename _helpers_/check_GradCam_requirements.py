import pkg_resources

required = {
    "numpy": "2.0.2",
    "opencv-python": "4.11.0.86",
    "matplotlib": "3.9.2",
    "torch": "2.6.0",
    "keras": "3.7.0",
    "tensorflow": "2.19.0",
    "requests": "2.32.3",
    "torchvision": "0.21.0",
    "setuptools": "72.1.0"
}

for package, expected_version in required.items():
    try:
        dist = pkg_resources.get_distribution(package)
        if dist.version == expected_version:
            print(f"{package}=={expected_version} ✔️")
        else:
            print(f"{package} version mismatch: installed {dist.version}, expected {expected_version} ❌")
    except pkg_resources.DistributionNotFound:
        print(f"{package} not installed ❌")
