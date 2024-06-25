# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/06.dataflow.cruncher.ipynb.

# %% auto 0
__all__ = ['Cruncher']

# %% ../../nbs/06.dataflow.cruncher.ipynb 3
import logging
import queue
from datetime import datetime
from pathlib import Path
from threading import Event, current_thread
from typing import Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# %% ../../nbs/06.dataflow.cruncher.ipynb 5
from .filter.homo import HomoFilter  # type: ignore
from .pipeline.queue import Pipeline  # type: ignore
from .producer import Producer  # type: ignore
from ..config.drivers import Driver
from ..config.vehicles import Truck
from tspace.data.external.pandas_utils import (
    assemble_action_ser,
    assemble_flash_table,
    assemble_reward_ser,
    assemble_state_ser,
)
from ..system.plot import plot_3d_figure, plot_to_image
from ..agent.dpg import DPG

# %% ../../nbs/06.dataflow.cruncher.ipynb 6
@dataclass
class Cruncher(HomoFilter[pd.DataFrame]):
    """
    Cruncher is the processing unit of the data flow.

    It consumes data (pd.DataFrame) from the observe pipeline and produces data (pd.DataFrame) into the flash pipeline.

    Attributes:

        - agent: abstract base class DPG, available DDPG, RDPG
        - truck: Truck object
        - driver: Driver object
        - resume: bool, whether to resume training
        - infer_mode: bool, whether only inferring or with training and inferring
        - data_dir: Path, the local path to save all generated data
        - train_summary_writer: SummaryWriter, Tensorflow training writer
        - logger: Logger
        - dict_logger: logger format specs
    """

    agent: DPG
    truck: Truck
    driver: Driver
    resume: bool = False
    infer_mode: bool = False
    data_dir: Optional[Path] = None
    train_summary_writer: Optional[tf.summary.SummaryWriter] = None  # type: ignore
    logger: Optional[logging.Logger] = None
    dict_logger: Optional[dict] = None

    def __post_init__(self):
        """Set logger, Tensorflow data path and running mode"""
        # self.logger = logger.getChild("eos").getChild((self.__str__()))
        self.logger = self.logger.getChild(self.__str__())
        if not self.data_dir:
            self.data_dir = Path(".")

        tfb_path = self.data_dir.joinpath(
            "tf-logs-"
            + str(self.agent)
            + self.truck.vid
            + "/gradient_tape/"
            + pd.Timestamp.now(tz=self.truck.site.tz).isoformat()
            + "/train"
        )
        self.train_summary_writer = tf.summary.create_file_writer(  # type: ignore
            str(tfb_path)
        )

        if self.resume:
            self.logger.info(
                f"{{'header': 'Resume last training'}}", extra=self.dict_logger
            )
        else:
            self.logger.info(
                f"{{'header': 'Start from scratch'}}", extra=self.dict_logger
            )
        super().__post_init__()
        self.logger.info("Cruncher initialized")

    def __str__(self):
        return "cruncher"

    def filter(
        self,
        in_pipeline: Pipeline[pd.DataFrame],  # input pipeline
        out_pipeline: Pipeline[pd.DataFrame],  # output pipeline
        start_event: Optional[Event],  # input event start
        stop_event: Optional[Event],  # input event stop
        interrupt_event: Optional[Event],  # input event interrupt
        flash_event: Optional[Event],  # input event flash
        exit_event: Optional[Event],  # input event exit
    ) -> None:
        """
        Consume data from the pipeline

        Args:
            in_pipeline: Pipeline, the input pipeline
            out_pipeline: Pipeline, the output pipeline
            start_event: Event, start event
            stop_event: Event, stop event
            interrupt_event: Event, interrupt event
            flash_event: Event, flash event
            exit_event: Event, exit event
        """
        thread = current_thread()
        thread.name = "cruncher_consume"
        running_reward = 0.0
        epi_cnt = 0

        logger_cruncher_consume = self.logger.getChild("consume")
        logger_cruncher_consume.info(f"Cruncher thread starts!", extra=self.dict_logger)
        while not exit_event.is_set():  # run until program exit
            if (
                (not start_event.is_set())
                or stop_event.is_set()
                or interrupt_event.is_set()
            ):
                continue

            # tf.summary.trace_on(graph=True, profiler=True)

            logger_cruncher_consume.info(
                "----------------------", extra=self.dict_logger
            )
            logger_cruncher_consume.info(
                f"{{'header': 'episode starts!', " f"'episode': {epi_cnt}}}",
                extra=self.dict_logger,
            )
            # mongodb default to UTC time

            # Get the initial motion_power data for the initial quadruple (s, a, r, s')_{-1}
            motion_power = None
            # if any of the following events occurs, break out of the loop
            while not (
                exit_event.is_set() or stop_event.is_set() or interrupt_event.is_set()
            ):
                try:
                    motion_power = in_pipeline.get(block=True, timeout=10)
                    break  # break the while loop if we get the first data
                except TimeoutError:
                    logger_cruncher_consume.info(
                        f"{{'header': 'No data in the input Queue, Timeout!!!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dict_logger,
                    )
                    continue
                except queue.Empty:
                    logger_cruncher_consume.info(
                        f"{{'header': 'No data in the input Queue, Empty!!!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dict_logger,
                    )
                    continue

            # prime the episode
            self.agent.start_episode(ts=pd.Timestamp.now(tz=self.truck.site.tz))
            step_count = 0
            episode_reward = 0.0
            prev_timestamp = self.agent.episode_start_dt
            assert (
                type(motion_power) is pd.DataFrame
            ), f"motion_power is {type(motion_power)}, not pd.DataFrame!"
            prev_state, table_start = assemble_state_ser(
                motion_power.loc[:, ["timestep", "velocity", "thrust", "brake"]],
                tz=self.truck.site.tz,
            )  # s_{-1}
            zero_torque_map_line = np.zeros(
                shape=(1, 1, self.truck.torque_flash_numel),  # [1, 1, 4*17]
                dtype=np.float32,
            )  # first zero last_actions is a 3D tensor
            prev_action = assemble_action_ser(
                torque_map_line=zero_torque_map_line,
                torque_table_row_names=self.agent.torque_table_row_names,
                table_start=table_start,
                flash_start_ts=pd.to_datetime(prev_timestamp),
                flash_end_ts=pd.Timestamp.now(self.truck.site.tz),
                torque_table_row_num_flash=self.truck.torque_table_row_num_flash,
                torque_table_col_num=self.truck.torque_table_col_num,
                speed_scale=self.truck.speed_scale,
                pedal_scale=self.truck.pedal_scale,
                tz=self.truck.site.tz,
            )  # a_{-1}
            step_reward = 0.0
            # reward is measured in next step

            logger_cruncher_consume.info(
                f"{{'header': 'episode init done!', " f"'episode': {epi_cnt}}}",
                extra=self.dict_logger,
            )
            flash_event.set()  # kick off the episode capturing, reactivate data_transform
            b_flashed = False
            tf.debugging.set_log_device_placement(True)
            with tf.device("/GPU:0"):
                while (not stop_event.is_set()) and (
                    not interrupt_event.is_set() and (not exit_event.is_set())
                ):
                    observe_pipeline_size = in_pipeline.qsize()
                    logger_cruncher_consume.info(
                        f"observe pipeline size: {observe_pipeline_size}"
                    )
                    if observe_pipeline_size > 2:
                        # self.logc.info(f"motion_power_queue.qsize(): {self.motion_power_queue.qsize()}")
                        logger_cruncher_consume.info(
                            f"{{'header': 'Residue in Queue is a sign of disordered sequence, interrupted!'}}"
                        )
                        interrupt_event.set()

                    try:
                        motion_power = in_pipeline.get(block=True, timeout=1.55)
                    except TimeoutError:
                        logger_cruncher_consume.info(
                            f"{{'header': 'No data in the input Queue Timeout!!!', "
                            f"'episode': {epi_cnt}}}",
                            extra=self.dict_logger,
                        )
                        continue
                    except queue.Empty:
                        logger_cruncher_consume.info(
                            f"{{'header': 'No data in the input Queue empty Queue!!!', "
                            f"'episode': {epi_cnt}}}",
                            extra=self.dict_logger,
                        )
                        continue

                    logger_cruncher_consume.info(
                        f"{{'header': 'start', "
                        f"'step': {step_count}, "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dict_logger,
                    )  # env.step(action) action is flash the vcu calibration table

                    # !!!no parallel even!!!
                    # predict action probabilities and estimated future rewards
                    # from environment state
                    # for causal rl, the odd indexed observation/reward are caused by last action
                    # skip the odd indexed observation/reward for policy to make it causal

                    # assemble state
                    timestamp = motion_power.loc[
                        0, "timestep"
                    ]  # only take the first timestamp,
                    assert isinstance(
                        timestamp, pd.Timestamp
                    ), "timestamp is not pd.Timestamp!"
                    # as frequency is fixed at 50Hz, the rest is saved in another col

                    # motion_power.loc[:, ['timestep', 'velocity', 'thrust', 'brake']]
                    state, table_start_row = assemble_state_ser(
                        motion_power.loc[
                            :, ["timestep", "velocity", "thrust", "brake"]
                        ],
                        tz=self.truck.site.tz,
                    )

                    # assemble reward, actually the reward from last action
                    # pow_t = motion_power.loc[:, ['current', 'voltage']]
                    reward = assemble_reward_ser(
                        motion_power.loc[:, ["current", "voltage"]],
                        self.truck.observation_sampling_rate,
                        ts=pd.Timestamp.now(tz=self.truck.site.tz),
                    )
                    work = reward[("work", 0)]
                    episode_reward += float(work)

                    logger_cruncher_consume.info(
                        f"{{'header': 'assembling state and reward!', "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dict_logger,
                    )

                    #  separate the inference and flash in order to avoid the action change incurred reward noise
                    if b_flashed is False:  # the active half step
                        #  at step 0: [ep_start, None (use zeros), a=0, r=0, s=s_0]
                        #  at step n: [t=t_{n-1}, s=s_{n-1}, a=a_{n-1}, r=r_{n-1}, s'=s_n]
                        #  at step N: [t=t_{N-1}, s=s_{N-1}, a=a_{N-1}, r=r_{N-1}, s'=s_N]
                        reward[("work", 0)] = (
                            work + step_reward
                        )  # reward is the sum of flashed and not flashed step
                        self.agent.deposit(
                            prev_timestamp,
                            prev_state,
                            prev_action,
                            reward,  # reward from last action
                            state,
                        )  # (s_{-1}, a_{-1}, r_{-1}, s_0), (s_0, a_0, r_0, s_1), ..., (s_{N-1}, a_{N-1}, r_{N-1}, s_N)

                        # Inference !!!
                        # stripping timestamps from state, (later flatten and convert to tensor)
                        # agent return the inferred action sequence without batch and time dimension
                        torque_table_line = self.agent.actor_predict(
                            state[["velocity", "thrust", "brake"]]
                        )  # model input requires fixed order velocity col -> thrust col -> brake col
                        #  !!! training with samples of the same order!!!
                        df_torque_table = assemble_flash_table(
                            torque_table_line,
                            table_start_row,
                            self.truck.torque_table_row_num_flash,
                            self.truck.torque_table_col_num,
                            self.truck.speed_scale,
                            self.truck.pedal_scale,
                        )

                        logger_cruncher_consume.info(
                            f"{{'header': 'inference done with reduced action space!', "
                            f"'episode': {epi_cnt}}}",
                            extra=self.dict_logger,
                        )
                        # flash the vcu calibration table and assemble action
                        flash_start_ts = pd.Timestamp.now(self.truck.site.tz)
                        out_pipeline.put_data(df_torque_table)
                        logger_cruncher_consume.info(
                            f"{{'header': 'Action Push table', "
                            f"'StartIndex': {table_start_row}, "
                            f"'qsize': {out_pipeline.qsize()}}}",
                            extra=self.dict_logger,
                        )

                        # wait for remote flash to finish
                        flash_event.wait()  # clear the event flag in observe thread
                        if interrupt_event.is_set() or exit_event.is_set():
                            continue

                        logger_cruncher_consume.info(
                            f"{{'header': 'flash lock released!",
                            extra=self.dict_logger,
                        )
                        flash_end_ts = pd.Timestamp.now(self.truck.site.tz)

                        action = assemble_action_ser(
                            torque_table_line,
                            self.agent.torque_table_row_names,
                            table_start,
                            flash_start_ts,
                            flash_end_ts,
                            self.truck.torque_table_row_num_flash,
                            self.truck.torque_table_col_num,
                            self.truck.speed_scale,
                            self.truck.pedal_scale,
                            self.truck.site.tz,
                        )

                        prev_timestamp = timestamp
                        prev_state = state
                        prev_action = action
                        b_flashed = True
                    else:  # if bFlashed is True, the dummy half step
                        step_reward = float(
                            work
                        )  # reward from the step without flashing action
                        flash_event.set()  # kick off the episode capturing, reactivate data_transform
                        b_flashed = False

                    # TODO add speed sum as positive reward
                    logger_cruncher_consume.info(
                        f"{{'header': 'Step done',"
                        f"'step': {step_count}, "
                        f"'episode': {epi_cnt}}}",
                        extra=self.dict_logger,
                    )

                    # during odd steps, old action remains effective due to learn and flash delay
                    # so ust record the reward history
                    # motion states (observation) are not used later for backpropagation

                    # step level
                    step_count += 1

            if (
                interrupt_event.is_set()  # episode not DONE
            ):  # if user interrupt prematurely or exit, then ignore back propagation since data incomplete
                logger_cruncher_consume.info(
                    f"{{'header': 'interrupted, waits for next episode to kick off!' "
                    f"'episode': {epi_cnt}}}",
                    extra=self.dict_logger,
                )
                # # send ready signal to trip server
                # if self.ui == "mobile":
                #     ret = self.rmq_producer.send_sync(self.rmq_message_ready)
                #     logger_cruncher_consume.info(
                #         f"{{'header': 'Sending ready signal to trip server', "
                #         f"'status': '{ret.status}', "
                #         f"'msg-id': '{ret.msg_id}', "
                #         f"'offset': '{ret.offset}'}}",
                #         extra=self.dict_logger,
                #     )
                epi_cnt += 1
                continue  # otherwise assuming the history is valid and back propagate

            self.agent.end_episode()  # deposit history

            logger_cruncher_consume.info(
                f"{{'header': 'Episode end.', "
                f"'episode': '{epi_cnt}', "
                f"'timestamp': '{datetime.now(self.truck.site.tz)}'}}",
                extra=self.dict_logger,
            )

            critic_loss = 0
            actor_loss = 0
            if self.infer_mode:
                (critic_loss, actor_loss) = self.agent.get_losses()
                # FIXME bugs in maximal sequence length for ungraceful testing
                # self.logc.info("Nothing to be done for rdpg!")
                logger_cruncher_consume.info(
                    "{{'header': 'No Learning, just calculating loss.'}}"
                )
            else:
                logger_cruncher_consume.info(
                    "{{'header': 'Learning and updating 6 times!'}}"
                )

                # self.logger.info(f"BP{k} starts.", extra=self.dict_logger)
                if self.agent.buffer.pool.cnt > 0:
                    for k in range(6):
                        (critic_loss, actor_loss, value_loss) = self.agent.train()
                        self.agent.soft_update_target()
                else:
                    logger_cruncher_consume.info(
                        f"{{'header': 'Buffer empty, no learning!'}}",
                        extra=self.dict_logger,
                    )
                    logger_cruncher_consume.info(
                        "++++++++++++++++++++++++", extra=self.dict_logger
                    )
                # Checkpoint manager save model
                self.agent.save_ckpt()

            logger_cruncher_consume.info(
                f"{{'header': 'losses after 6 times BP', "
                f"'episode': {epi_cnt}, "
                f"'critic loss': {critic_loss}, "
                f"'actor loss': {actor_loss}}}",
                extra=self.dict_logger,
            )

            # update running reward to check condition for solving
            running_reward = 0.05 * (-episode_reward) + (1 - 0.05) * running_reward

            # Create a matplotlib 3d figure for the last table, //export and save in log
            fig = plot_3d_figure(df_torque_table)

            # tf logging after episode ends
            # use local episode counter epi_cnt_local tf.summary.writer;
            # otherwise specify multiple self.logdir and automatic switch
            with self.train_summary_writer.as_default():
                tf.summary.scalar("WH", -episode_reward, step=epi_cnt)
                tf.summary.scalar("actor loss", actor_loss, step=epi_cnt)
                tf.summary.scalar("critic loss", critic_loss, step=epi_cnt)
                tf.summary.scalar("reward", episode_reward, step=epi_cnt)
                tf.summary.scalar("running reward", running_reward, step=epi_cnt)
                tf.summary.image("Calibration Table", plot_to_image(fig), step=epi_cnt)
                tf.summary.histogram(
                    "Calibration Table Hist",
                    df_torque_table.values,
                    step=epi_cnt,
                )
                # tf.summary.trace_export(
                #     name="veos_trace", step=epi_cnt_local, profiler_out_dir=train_log_dir
                # )

            plt.close(fig)

            logger_cruncher_consume.info(
                f"{{'episode': {epi_cnt}, " f"'reward': {episode_reward}}}",
                extra=self.dict_logger,
            )

            logger_cruncher_consume.info(
                "----------------------", extra=self.dict_logger
            )
            if epi_cnt % 10 == 0:
                logger_cruncher_consume.info(
                    "++++++++++++++++++++++++", extra=self.dict_logger
                )
                logger_cruncher_consume.info(
                    f"{{'header': 'Running reward': {running_reward:.2f}, "
                    f"'episode': '{epi_cnt}'}}",
                    extra=self.dict_logger,
                )
                logger_cruncher_consume.info(
                    "++++++++++++++++++++++++", extra=self.dict_logger
                )

            epi_cnt += 1
        # TODO terminate condition to be defined: reward > limit (percentage); time too long
        # with self.train_summary_writer.as_default():
        #     tf.summary.trace_export(
        #         name="veos_trace",
        #         step=epi_cnt_local,
        #         profiler_out_dir=self.train_log_dir,
        #     )

        self.agent.buffer.close()
        plt.close(fig="all")

        logger_cruncher_consume.info(
            f"{{'header': 'Close Buffer, pool!'}}", extra=self.dict_logger
        )

        logger_cruncher_consume.info(
            f"{{'header': 'crunch thread dies!!!!'}}", extra=self.dict_logger
        )
