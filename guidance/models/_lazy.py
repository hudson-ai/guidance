

class LazyModel:

    def __init__(self, model, work='', parent=None):
        self.model = model
        self.work = work
        self.parent = parent
        self.children = set()

    def __str__(self):
        return str(self.unwrap())

    def __add__(self, string):
        if string:
            lm = LazyModel(self.model, string, parent=self)
            self.children.add(lm)
            return lm
        return self

    def unwrap(self):
        if not self.work:
            return self.model
        if not self.parent:
            self.model += self.work
            return self.model

        current = self
        work = ""
        while current.parent:
            work = current.work + work
            current = current.parent
            if len(current.children) > 1:
                break
        root = current

        self.model = root.unwrap() + work
        self.work = ""

        return self.model