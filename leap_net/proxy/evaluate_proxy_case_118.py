# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of leap_net, leap_net a keras implementation of the LEAP Net model.

from leap_net.proxy.utils import DEFAULT_METRICS

from leap_net.ResNetLayer import ResNetLayer
from leap_net.proxy.evaluate_proxy_case_14 import main as main_14


def main(
        # reload the model
        env_name="l2rpn_neurips_2020_track2_small",
        save_path="model_saved_118",
        use_lightsim_if_available=True,
        val_regex=".*Scenario_february_0[0-9].*",
        model_name="leapnet_case_118",
        layer=ResNetLayer,  # for now this is not serialized in the model
        # parameters for the evaluation
        do_dc=True,
        do_N1 = True,
        do_N2 = True,
        li_batch_size=tuple(),  # if you want to study the impact of the batch size
        total_evaluation_step=int(1024) * int(128),
        pred_batch_size=int(1024) * int(32),
        save_path_final_results="model_results_118",  # where the information about the prediction will be stored
        metrics=DEFAULT_METRICS,  # which metrics are used to evaluate the performance of the model
        verbose=1,  # do I print the results of the model
        # enforce reproducibility
        chron_id_val=0,
        env_seed=0,
        agent_seed=42,
        ):
    main_14(env_name=env_name,
            save_path=save_path,
            use_lightsim_if_available=use_lightsim_if_available,
            val_regex=val_regex,
            model_name=model_name,
            layer=layer,
            do_dc=do_dc,
            do_N1=do_N1,
            do_N2=do_N2,
            li_batch_size=li_batch_size,
            total_evaluation_step=total_evaluation_step,
            pred_batch_size=pred_batch_size,
            save_path_final_results=save_path_final_results,
            metrics=metrics,
            verbose=verbose,
            chron_id_val=chron_id_val,
            env_seed=env_seed,
            agent_seed=agent_seed
            )


if __name__ == "__main__":
    main()
