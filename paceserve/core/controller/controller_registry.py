
from paceserve.core.controller.roundrobin_controller import RoundRobinController
from paceserve.core.controller.distserve_controller import DistServeController
from paceserve.core.controller.pace_controller import PaceController
from paceserve.utils.base_registry import BaseRegistry
from paceserve.types import ControllerType

class ControllerRegistry(BaseRegistry):
    pass

ControllerRegistry.register(
    ControllerType.ROUNDROBIN, RoundRobinController
)

ControllerRegistry.register(
    ControllerType.DISTSERVE, DistServeController
)

ControllerRegistry.register(
    ControllerType.PACE, PaceController
)