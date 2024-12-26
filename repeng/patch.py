# patch sklearn before importing anything else: https://uxlfoundation.github.io/scikit-learn-intelex/latest/
from sklearnex import patch_sklearn
try:
    patch_sklearn()
except Exception as err:
    print(f"Failed to patch sklearn to make it faster: '{err}'")
