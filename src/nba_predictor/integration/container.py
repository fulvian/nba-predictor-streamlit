#!/usr/bin/env python3
"""
🏀 NBA Prediction System Dependency Injection Container
Context7-compliant dependency injection following SOLID principles and best practices.

This module implements:
- IoC Container for dependency injection
- Service lifecycle management (singleton, scoped, transient)
- Interface-based dependency resolution
- Context7-compliant service registration patterns
"""

from typing import Type, TypeVar, Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
import logging
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceLifetime(Enum):
    """Service lifetime enumeration following Context7 best practices"""
    SINGLETON = "singleton"      # One instance for the entire application
    SCOPED = "scoped"           # One instance per scope/request
    TRANSIENT = "transient"     # New instance every time


class IServiceContainer(ABC):
    """Interface for dependency injection container following Context7 patterns"""

    @abstractmethod
    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> 'IServiceContainer':
        """Register singleton service"""
        pass

    @abstractmethod
    def register_scoped(self, interface: Type[T], implementation: Type[T]) -> 'IServiceContainer':
        """Register scoped service"""
        pass

    @abstractmethod
    def register_transient(self, interface: Type[T], implementation: Type[T]) -> 'IServiceContainer':
        """Register transient service"""
        pass

    @abstractmethod
    def register_instance(self, interface: Type[T], instance: T) -> 'IServiceContainer':
        """Register specific instance"""
        pass

    @abstractmethod
    def get_service(self, service_type: Type[T]) -> T:
        """Resolve service"""
        pass

    @abstractmethod
    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        pass


class ServiceDescriptor:
    """Service descriptor for container configuration"""

    def __init__(self,
                 interface: Type,
                 implementation: Type = None,
                 instance: Any = None,
                 lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
                 factory: Callable = None):
        self.interface = interface
        self.implementation = implementation
        self.instance = instance
        self.lifetime = lifetime
        self.factory = factory


class DIContainer(IServiceContainer):
    """
    Context7-compliant Dependency Injection Container

    Implements:
    - Service lifetime management
    - Circular dependency detection
    - Interface-based registration
    - Factory method support
    """

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._singletons: Dict[Type, Any] = {}
        self._scoped_instances: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None
        self._resolving: set = set()  # Track circular dependencies

        logger.info("DIContainer initialized with Context7 patterns")

    def register_singleton(self, interface: Type[T], implementation: Type[T] = None) -> 'DIContainer':
        """
        Register singleton service.

        Context7-compliant: Single instance per application lifetime.

        Args:
            interface: Service interface type
            implementation: Service implementation type

        Returns:
            Container instance for fluent API
        """
        if implementation is None:
            implementation = interface

        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._services[interface] = descriptor
        logger.debug(f"Registered singleton: {interface.__name__} -> {implementation.__name__}")
        return self

    def register_scoped(self, interface: Type[T], implementation: Type[T] = None) -> 'DIContainer':
        """
        Register scoped service.

        Context7-compliant: Single instance per scope/request.

        Args:
            interface: Service interface type
            implementation: Service implementation type

        Returns:
            Container instance for fluent API
        """
        if implementation is None:
            implementation = interface

        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.SCOPED
        )
        self._services[interface] = descriptor
        logger.debug(f"Registered scoped: {interface.__name__} -> {implementation.__name__}")
        return self

    def register_transient(self, interface: Type[T], implementation: Type[T] = None) -> 'DIContainer':
        """
        Register transient service.

        Context7-compliant: New instance for each request.

        Args:
            interface: Service interface type
            implementation: Service implementation type

        Returns:
            Container instance for fluent API
        """
        if implementation is None:
            implementation = interface

        descriptor = ServiceDescriptor(
            interface=interface,
            implementation=implementation,
            lifetime=ServiceLifetime.TRANSIENT
        )
        self._services[interface] = descriptor
        logger.debug(f"Registered transient: {interface.__name__} -> {implementation.__name__}")
        return self

    def register_instance(self, interface: Type[T], instance: T) -> 'DIContainer':
        """
        Register specific instance as singleton.

        Context7-compliant: Use existing instance.

        Args:
            interface: Service interface type
            instance: Service instance

        Returns:
            Container instance for fluent API
        """
        descriptor = ServiceDescriptor(
            interface=interface,
            instance=instance,
            lifetime=ServiceLifetime.SINGLETON
        )
        self._services[interface] = descriptor
        self._singletons[interface] = instance
        logger.debug(f"Registered instance: {interface.__name__} -> {type(instance).__name__}")
        return self

    def register_factory(self, interface: Type[T], factory: Callable[[], T],
                        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT) -> 'DIContainer':
        """
        Register factory method for service creation.

        Context7-compliant: Factory pattern for complex instantiation.

        Args:
            interface: Service interface type
            factory: Factory function
            lifetime: Service lifetime

        Returns:
            Container instance for fluent API
        """
        descriptor = ServiceDescriptor(
            interface=interface,
            factory=factory,
            lifetime=lifetime
        )
        self._services[interface] = descriptor
        logger.debug(f"Registered factory: {interface.__name__} with lifetime {lifetime.value}")
        return self

    def get_service(self, service_type: Type[T]) -> T:
        """
        Resolve service instance.

        Context7-compliant: Automatic dependency injection with circular dependency detection.

        Args:
            service_type: Service type to resolve

        Returns:
            Service instance

        Raises:
            ValueError: If service is not registered or circular dependency detected
        """
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} is not registered")

        # Check for circular dependencies
        if service_type in self._resolving:
            cycle = " -> ".join([t.__name__ for t in self._resolving] + [service_type.__name__])
            raise ValueError(f"Circular dependency detected: {cycle}")

        descriptor = self._services[service_type]

        # Return existing instance for singletons
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if service_type in self._singletons:
                return self._singletons[service_type]

        # Return existing instance for scoped services
        if descriptor.lifetime == ServiceLifetime.SCOPED:
            if self._current_scope and self._current_scope in self._scoped_instances:
                if service_type in self._scoped_instances[self._current_scope]:
                    return self._scoped_instances[self._current_scope][service_type]

        # Create new instance
        self._resolving.add(service_type)
        try:
            instance = self._create_instance(descriptor)

            # Store instance based on lifetime
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                self._singletons[service_type] = instance
            elif descriptor.lifetime == ServiceLifetime.SCOPED and self._current_scope:
                if self._current_scope not in self._scoped_instances:
                    self._scoped_instances[self._current_scope] = {}
                self._scoped_instances[self._current_scope][service_type] = instance

            logger.debug(f"Created instance of {service_type.__name__} ({descriptor.lifetime.value})")
            return instance

        finally:
            self._resolving.discard(service_type)

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """
        Create service instance based on descriptor.

        Context7-compliant: Support for constructor injection and factory methods.
        """
        # Use factory if available
        if descriptor.factory:
            return descriptor.factory()

        # Use existing instance if available
        if descriptor.instance:
            return descriptor.instance

        # Create new instance with constructor injection
        implementation_class = descriptor.implementation
        if not implementation_class:
            raise ValueError(f"No implementation or factory registered for {descriptor.interface.__name__}")

        # Simple constructor injection (basic implementation)
        try:
            return implementation_class()
        except TypeError as e:
            # If constructor requires parameters, try to resolve them
            import inspect
            sig = inspect.signature(implementation_class.__init__)
            parameters = {}

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param.annotation != inspect.Parameter.empty:
                    try:
                        parameters[param_name] = self.get_service(param.annotation)
                    except ValueError:
                        if param.default == inspect.Parameter.empty:
                            logger.warning(f"Could not resolve parameter {param_name} for {implementation_class.__name__}")

            return implementation_class(**parameters)

    def is_registered(self, service_type: Type) -> bool:
        """Check if service is registered"""
        return service_type in self._services

    def create_scope(self, scope_name: str = None) -> 'DIScope':
        """
        Create new dependency scope.

        Context7-compliant: Scoped service lifetime management.

        Args:
            scope_name: Optional scope identifier

        Returns:
            DIScope instance
        """
        if scope_name is None:
            import uuid
            scope_name = str(uuid.uuid4())

        return DIScope(self, scope_name)

    def clear_scoped_instances(self, scope_name: str = None):
        """Clear scoped instances for specific scope or all scopes"""
        if scope_name:
            self._scoped_instances.pop(scope_name, None)
        else:
            self._scoped_instances.clear()

    def get_registered_services(self) -> Dict[Type, ServiceDescriptor]:
        """Get all registered services"""
        return self._services.copy()

    def validate_dependencies(self) -> List[str]:
        """
        Validate all registered dependencies.

        Context7-compliant: Dependency validation at startup.

        Returns:
            List of validation errors
        """
        errors = []

        for service_type, descriptor in self._services.items():
            try:
                # Try to resolve each service
                self.get_service(service_type)
            except Exception as e:
                errors.append(f"Failed to resolve {service_type.__name__}: {e}")

        return errors


class DIScope:
    """Dependency injection scope context manager"""

    def __init__(self, container: DIContainer, scope_name: str):
        self.container = container
        self.scope_name = scope_name
        self._previous_scope = None

    def __enter__(self):
        self._previous_scope = self.container._current_scope
        self.container._current_scope = self.scope_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.container.clear_scoped_instances(self.scope_name)
        self.container._current_scope = self._previous_scope


# Global container instance following Context7 singleton pattern
_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """
    Get global dependency injection container.

    Context7-compliant: Singleton pattern for global access.

    Returns:
        DIContainer instance
    """
    global _container
    if _container is None:
        _container = DIContainer()
    return _container


def configure_services(configurator: Callable[[DIContainer], None]) -> DIContainer:
    """
    Configure dependency injection container.

    Context7-compliant: Configuration pattern for service registration.

    Args:
        configurator: Function that configures services

    Returns:
        Configured container
    """
    container = get_container()
    configurator(container)

    # Validate configuration
    errors = container.validate_dependencies()
    if errors:
        logger.warning(f"Dependency validation errors: {errors}")

    logger.info("Dependency injection container configured")
    return container