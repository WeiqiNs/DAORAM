class OblivlibError(Exception):
    pass


class UnknownLabelError(OblivlibError):
    pass


class MissingResultError(OblivlibError):
    pass


class MissingClientError(OblivlibError):
    pass


class DuplicateLabelError(OblivlibError):
    pass


class StashOverflowError(OblivlibError, MemoryError):
    pass


class ScaleDownError(OblivlibError):
    pass
