# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.
import os
from leap_net import generate_dataset


def main(p, data_dir):
    data_dir_abs = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir_abs)

    # generate the training dataset
    generate_dataset("l2rpn_case14_sandbox",
                     dir_out=os.path.join(data_dir_abs, "training_set_{:.3f}".format(p)),
                     nb_rows=1024*64,
                     agent_type="random_n_n1",
                     p=p)
    generate_dataset("l2rpn_case14_sandbox",
                     dir_out=os.path.join(data_dir_abs, "liketrain_set_{:.3f}".format(p)),
                     nb_rows=1024*16,
                     agent_type="random_n_n1",
                     p=p)
    if not os.path.exists(os.path.join(data_dir_abs, "test_set")):
        generate_dataset("l2rpn_case14_sandbox",
                         dir_out=os.path.join(data_dir_abs, "test_set"),
                         nb_rows=1024*16,
                         agent_type="random_n1")
    if not os.path.exists(os.path.join(data_dir_abs, "supertest_set")):
        generate_dataset("l2rpn_case14_sandbox",
                         dir_out=os.path.join(data_dir_abs, "supertest_set"),
                         nb_rows=1024*16,
                         agent_type="random_n2")


if __name__ == "__main__":
    data_dir = "data"
    for p in [0.001, 0.003, 0.01, 0.03, 0.1, 0.5]:
        main(p=p, data_dir=data_dir)
        