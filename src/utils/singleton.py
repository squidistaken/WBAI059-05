from typing import Any


class SingletonMeta(type):
    """Metaclass for creating singleton classes."""

    _instances = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create or return existing singleton instances.

        Args:
            cls: The class to instantiate.

        Returns:
            type: The singleton instance of the class.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMeta, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls]
