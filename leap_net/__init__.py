# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

__version__ = "0.1.2.dev0"
__all__ = []

try:
    from leap_net.keras import (Ltau, LtauNoAdd, ResNetLayer)
    __all__ += ["Ltau", "ResNetLayer", "LtauNoAdd"]
except ImportError:
    pass


try:
    from leap_net.generate_data import generate_dataset
    __all__ += ["generate_dataset"]
except ImportError:
    pass

try:
    from leap_net.tf_keras.kerasutils import MultipleDasetCallBacks
    __all__ += ["MultipleDasetCallBacks"]
except ImportError:
    pass

try:
    from leap_net.proxy import BaseProxy, BaseNNProxy, ProxyBackend, ProxyLeapNet, AgentWithProxy
    __all__ += ["BaseProxy", "BaseNNProxy", "ProxyBackend", "ProxyLeapNet", "AgentWithProxy"]
except ImportError:
    pass
