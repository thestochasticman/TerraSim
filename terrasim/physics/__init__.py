"""Process formulations for the soil column, vectorised over all pixels.

Each sub-process is a small pure module operating on numpy arrays shaped
``(nlayer, ny, nx)`` or ``(ny, nx)``; :mod:`terrasim.physics.engine` strings
them together in a fixed operator-splitting order and checks mass balance
every step. Heavier imports are kept out of this package ``__init__``.
"""
