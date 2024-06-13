class LazyModel:
    """
    Wrapper around a model that allows for lazy evaluation of grammars
    appended to it. This is useful for building up multiple grammars in
    sequence and then applying them all at once.

    Work won't be applied to the model until an operation that requires
    the work to be applied is called, i.e. `__str__` and `__getitem__`
    """

    def __init__(self, model, work="", parent=None):
        self.model = model
        self.work = work
        self.parent = parent
        self.children = set()

    def __str__(self):
        """
        Apply all work to the underlying model and return str(model).
        """
        return str(self.unwrap())

    def __getitem__(self, key):
        """
        Apply all work to the underlying model and return model[key].
        """
        return self.unwrap()[key]

    def __add__(self, string):
        """
        Primary method for queuing up work to be applied to the model.
        """
        if string:
            lm = LazyModel(self.model, string, parent=self)
            self.children.add(lm)
            return lm
        return self

    def unwrap(self):
        """
        Apply all work to the underlying model and return the result.
        """
        if not self.work:
            # No work to apply.
            return self.model

        if not self.parent:
            # Single piece of work to apply, no need to
            # traverse parents
            self.model += self.work
            return self.model

        # Accumulate work from the last fork and apply
        # it as a single "hunk".
        current = self
        work = ""
        while current.parent:
            # Traversing bottom-to-top, so we need to prepend
            work = current.work + work
            current = current.parent
            # If we encounter a fork point, we need to stop
            if len(current.children) > 1:
                break
        root = current

        # Unwrap the fork point recursively to apply the work.
        # This ensures that every fork point is unwrapped exactly once.
        self.model = root.unwrap() + work
        self.work = ""

        return self.model
