# setup.py (Contents for ~/Desktop/F110_Gym_Setup/setup.py)
from setuptools import setup, find_packages

setup(
    name='f110_gym',  # The package name used for importing (matches the folder)
    version='1.0.1', # Incrementing version since code changes were made
    description='F1/10 Gym Environment updated for Python 3.12',
    author='NS', # Your name
    
    packages=find_packages(), # Finds the 'f110_gym' directory and its subpackages
    
    # --- CRITICAL ADDITION: Package Data ---
    # This ensures that non-Python files (like maps) are bundled with the package.
    package_data={
        # 'f110_gym' is the package name. The list specifies the file pattern
        # to include, relative to the package root.
        'f110_gym': ['envs/maps/*'], 
    },
    
    # Flags to correctly handle non-code data files
    include_package_data=True, 
    zip_safe=False,
)
