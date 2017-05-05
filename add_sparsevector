def add_sparsevector(v1, v2):
    """Add two sparse vectors
    >>> v1 = Vectors.sparse(3, {0: 1.0, 2: 1.0})
    >>> v2 = Vectors.sparse(3, {1: 1.0})
    >>> add(v1, v2)
    SparseVector(3, {0: 1.0, 1: 1.0, 2: 1.0})
    """
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
    assert v1.size == v2.size 
    # Compute union of indices
    indices = set(v1.indices).union(set(v2.indices))
    # Not particularly efficient but we are limited by SPARK-10973
    # Create index: value dicts
    v1d = dict(zip(v1.indices, v1.values))
    v2d = dict(zip(v2.indices, v2.values))
    zero = np.float64(0)
    # Create dictionary index: (v1[index] + v2[index])
    values =  {i: v1d.get(i, zero) + v2d.get(i, zero)
       for i in indices
       if v1d.get(i, zero) + v2d.get(i, zero) != zero}

    return Vectors.sparse(v1.size, values)
