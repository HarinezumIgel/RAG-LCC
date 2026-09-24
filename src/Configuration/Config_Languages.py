"""Language activation, Argos pairs, and shared spaCy role settings."""

# pyright: reportUnusedVariable=false, reportUnknownVariableType=false

import Configuration.Config_Languages_Catalog as _languages_catalog
import Configuration.Config_Languages_Normalization as _languages_normalization
import Configuration.Config_Languages_Retrievers as _languages_retrievers

for _module in (
    _languages_catalog,
    _languages_retrievers,
    _languages_normalization,
):
    for _name in dir(_module):
        if _name.isupper():
            globals()[_name] = getattr(_module, _name)

del _module
del _name
