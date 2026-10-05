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


class ProtocolError(OblivlibError):
    pass


class RowSizeError(OblivlibError):
    pass


class StorageError(OblivlibError):
    pass


class ServerError(OblivlibError):
    pass


class TransportError(OblivlibError):
    pass


class ContractError(OblivlibError, ValueError):
    pass
