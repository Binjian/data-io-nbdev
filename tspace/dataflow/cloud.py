# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/06.dataflow.cloud.ipynb.

# %% auto 0
__all__ = ['Cloud']

# %% ../../nbs/06.dataflow.cloud.ipynb 3
import concurrent.futures
import json
import math
import os
import subprocess
import time
from threading import Event, Lock, current_thread
from typing import Callable, Optional, Tuple, cast
from dataclasses import dataclass
import numpy as np
import pandas as pd

# from rocketmq.client import Message, Producer  # type: ignore

# %% ../../nbs/06.dataflow.cloud.ipynb 4
from .pipeline.queue import Pipeline  # type: ignore
from .pipeline.deque import PipelineDQ  # type: ignore
from .vehicle_interface import VehicleInterface  # type: ignore

# %% ../../nbs/06.dataflow.cloud.ipynb 5
# from tspace.conn.clearable_pull_consumer import ClearablePullConsumer

from ..conn.remote_can_client import RemoteCanClient, RemoteCanException
from ..config.messengers import CANMessenger, TripMessenger
from ..config.vehicles import TruckInCloud
from ..conn.udp import udp_context
from tspace.data.external.numpy_utils import (
    ragged_nparray_list_interp,
    timestamps_from_can_strings,
)
from ..data.core import RawType, RCANType

# %% ../../nbs/06.dataflow.cloud.ipynb 6
@dataclass
class Cloud(VehicleInterface):
    """
    Kvaser is local vehicle interface with Producer(get vehicle status) and Consumer(flasher)

    Attributes:

        truck: TruckInCloud
            truck type is TruckInCloud
        can_server: CANMessenger
            can_server type is CANMessenger
        trip_server: Optional[TripMessenger] = None
            trip_server type is TripMessenger
        ui: str = "UDP"
            ui must be cloud, local or mobile, not {self.ui}
        web_srv = ("rocket_intra",)
            web_srv is a tuple of str
        epi_countdown_time: float = 3.0
            epi_countdown_time is a float
        remotecan: Optional[RemoteCanClient] = None
            RemoteCanClient type is RemoteCanClient
        rmq_consumer: Optional[ClearablePullConsumer] = None
            ClearablePullConsumer type is ClearablePullConsumer
        rmq_message_ready: Optional[Message] = None
            Message type is Message
        rmq_producer: Optional[Producer] = None
            Producer type is Producer
        remoteClient_lock: Optional[Lock] = None
            Lock type is Lock
    """

    truck: TruckInCloud
    can_server: CANMessenger
    trip_server: Optional[TripMessenger] = None
    ui: str = "UDP"
    web_srv = ("rocket_intra",)
    epi_countdown_time: float = 3.0
    remotecan: Optional[RemoteCanClient] = None
    # rmq_consumer: Optional[ClearablePullConsumer] = None
    # rmq_message_ready: Optional[Message] = None
    # rmq_producer: Optional[Producer] = None
    remoteClient_lock: Optional[Lock] = None

    def __post_init__(self):
        """init cloud interface and set ui type"""
        super().__post_init__()
        self.init_cloud()
        assert type(self.truck is TruckInCloud), "truck type is not TruckInCloud"
        assert self.ui in [
            "cloud",
            "local",
            "mobile",
        ], f"ui must be cloud, local or mobile, not {self.ui}"

        self.logger.info("Cloud interface initialized")

    def __str__(self):
        return "cloud"

    def init_cloud(self) -> None:
        """initialize cloud interface, set proxy and remote can client and the lock"""
        os.environ["http_proxy"] = ""
        self.remotecan = RemoteCanClient(
            host=self.can_server.host,
            port=self.can_server.port,
            truck=self.truck,
            logger=self.logger,
            dict_logger=self.dict_logger,
        )

        self.remoteClient_lock = Lock()

    def init_internal_pipelines(
        self,
    ) -> Tuple[
        PipelineDQ[RawType], Pipeline[str]
    ]:  # PipelineDQ[dict[str, Union[str, dict[str, list[Union[str, list[str]]]]]]],
        """initialize internal pipeline static type for cloud interface"""
        raw_pipeline = PipelineDQ[
            RawType
        ](  # [dict[str, dict[str, list[Union[str, list[str]]]]]]
            maxlen=1
        )
        hmi_pipeline = Pipeline[str](maxsize=1)
        return raw_pipeline, hmi_pipeline

    def flash_vehicle(self, torque_table: pd.DataFrame) -> None:
        """flash torque table to the VCU"""
        thread = current_thread()
        thread.name = "cloud_flash"
        with self.remoteClient_lock:
            try:
                self.remotecan.send_torque_map(pedal_map=torque_table, swap=True)  # 14
            except RemoteCanException as exc:
                self.logger.warning(
                    f"{{'header': 'remotecan send_torque_map failed: {exc}'}}",
                    extra=self.dict_logger,
                )
                if exc.err_code in (1, 1000, 1002):
                    self.cloud_ping()
                    # self.cloud_telnet_test()
                # else:
                # raise exc
                with self.lock_watchdog:
                    self.flash_failure_count += 1
            except Exception as exc:
                self.logger.error(
                    f"{{'header': 'remote get_signals failed: {exc}'}}",
                    extra=self.dict_logger,
                )
                raise exc

        self.logger.info(
            "{'header': 'Done flash initial table'}",
            extra=self.dict_logger,
        )

    def hmi_select(
        self,
    ) -> Callable[[Pipeline[str], Optional[Event]], None]:
        """
        select HMI interface according to ui type.

        Produce data into the pipeline main entry to the capture thread
        sub-thread method
        Callable input parameters example:
            hmi_pipeline: Pipeline[str],

        Return:
            Callback for the HMI thread with type
                Callable[[Pipeline[str], Optional[Event]], None]
        """
        if self.ui == "UDP":
            return self.hmi_capture_from_udp  # Callable[ [Pipeline[str], Event], None ]
        elif self.ui == "RMQ":
            return self.hmi_capture_from_rmq  # Callable[ [Pipeline[str], Event], None ]
        elif self.ui == "dummy":
            return (
                self.hmi_capture_from_dummy
            )  # Callable[ [Pipeline[str], Event], None ]
        else:
            raise ValueError(f"ui must be UDP, RMQ or dummy, not {self.ui}")

    def produce(
        self,
        raw_pipeline: PipelineDQ[
            RawType
        ],  # PipelineDQ[dict[str, dict[str, list[Union[str, list[str]]]]]],
        hmi_pipeline: Optional[Pipeline[str]] = None,
        exit_event: Optional[Event] = None,
    ):
        """Create secondary threading pool for HMI thread and data capture thread"""
        self.logger.info(
            "{'header': 'cloud produce Thread Pool starts!'}", extra=self.dict_logger
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="cloud",
        ) as executor:
            executor.submit(
                self.hmi_select(),  # will delegate to concrete the hmi control method
                hmi_pipeline,
                exit_event,
            )

            executor.submit(
                self.data_capture_from_remotecan,
                raw_pipeline,
                exit_event,
            )

        self.logger.info(
            "{'header': 'cloud produce Thread Pool dies!'}", extra=self.dict_logger
        )

    def data_capture_from_remotecan(
        self,
        raw_pipeline: PipelineDQ[
            RawType
        ],  # PipelineDQ[dict[str, dict[str, list[Union[str, list[str]]]]]],
        exit_event: Optional[Event] = None,
    ):
        """Callback for the data capture thread"""
        logger_remote_get = self.logger.getChild("remotecan_capture")
        logger_remote_get.propagate = True

        logger_remote_get.info(
            "cloud data_capture starts!",
            extra=self.dict_logger,
        )
        while not exit_event.is_set():
            logger_remote_get.info(
                "wait for remote get trigger", extra=self.dict_logger
            )

            # if episode is done, sleep for the extension time
            # cancel wait as soon as waking up
            timeout = self.truck.tbox_unit_number + 7
            logger_remote_get.info(
                f"Wake up to fetch remote data, duration={self.truck.tbox_unit_number}s timeout={timeout}s",
                extra=self.dict_logger,
            )
            with self.remoteClient_lock:
                try:
                    remotecan_data: RCANType = self.remotecan.get_signals(
                        duration=self.truck.tbox_unit_number, timeout=timeout
                    )  # timeout is 1 second longer than duration
                except RemoteCanException as exc:
                    logger_remote_get.warning(
                        f"{{'header': 'remote get_signals failed and retry', "
                        f"'ret_code': '{exc.err_code}', "
                        f"'ret_str': '{exc.codes[exc.err_code]}', "
                        f"'extra_str': '{exc.extra_msg}'}}",
                        extra=self.dict_logger,
                    )
                    with self.lock_watchdog:
                        self.capture_failure_count += 1
                    # if the exception is connection related, ping the server to get further information.
                    if exc.err_code in (1, 1000, 1002):
                        self.cloud_ping()
                        # self.cloud_telnet_test()
                    continue
                    # else:
                    #     raise exc
                except Exception as exc:
                    logger_remote_get.error(
                        f"{{'header': 'remote get_signals failed: {exc}'}}",
                        extra=self.dict_logger,
                    )
                    raise exc

            raw_pipeline.put_data(remotecan_data)  # deque is non-blocking

        logger_remote_get.info("cloud data_capture dies!!!!!", extra=self.dict_logger)

    def cloud_ping(self):
        """utility function for ping test"""

        response = os.system("ping -c 1 " + self.can_server.host)
        if response == 0:
            self.logger.info(
                f"{{'header': 'host is up', " f"'host': '{self.can_server.host}'}}",
                extra=self.dict_logger,
            )
        else:
            self.logger.info(
                f"{{'header': 'host is down', "
                f"'host': '{self.can_server.host}', "
                f"'response': '{response}'}}",
                extra=self.dict_logger,
            )
        # response_ping = ""
        # try:
        # response_ping = subprocess.check_output(
        #     "ping -c 1 " + self.can_server.host, shell=True
        # )
        # except subprocess.CalledProcessError as e:
        #     self.logger.info(
        #         f"{self.can_server.host} is down, responds: {response_ping}"
        #         f"return code: {e.return_code}, output: {e.output}!",
        #         extra=self.dict_logger,
        #     )
        # self.logger.info(
        #     f"{self.can_server.host} is up, responds: {response}!",
        #     extra=self.dict_logger,
        # )

    def cloud_telnet_test(self):
        """Utility function for telnet test"""
        try:
            response_telnet = subprocess.check_output(
                f"timeout 1 telnet {self.can_server.host} {self.can_server.port}",
                shell=True,
            )
            self.logger.info(
                f"Telnet {self.can_server.host} responds: {response_telnet}!",
                extra=self.dict_logger,
            )
        except subprocess.CalledProcessError as e:
            self.logger.info(
                f"telnet {self.can_server.host} return code: {e.returncode}, output: {e.output}!",
                extra=self.dict_logger,
            )
        except subprocess.TimeoutExpired as e:
            self.logger.info(
                f"telnet {self.can_server.host} timeout"
                f"cmd: {e.cmd}, output: {e.output}, timeout: {e.timeout}!",
                extra=self.dict_logger,
            )

    def hmi_capture_from_udp(
        self,
        hmi_pipeline: Pipeline[str],
        exit_event: Optional[Event] = None,
    ) -> None:
        """Callback function for getting HMI message from local UDP"""
        logger_hmi_capture_udp = self.logger.getChild("hmi_capture_udp")
        logger_hmi_capture_udp.propagate = True

        logger_hmi_capture_udp.info(
            "{'header': 'cloud hmi_capture udp thread starts!'}",
            extra=self.dict_logger,
        )
        with udp_context(self.can_server.host, self.can_server.port) as s:
            can_data, addr = s.recvfrom(2048)
            # self.logger.info('Data received!!!', extra=self.dict_logger)
            while True:
                try:
                    pop_data = json.loads(can_data)
                except TypeError as exc:
                    logger_hmi_capture_udp.warning(
                        f"{{'header': 'udp reception type error', "
                        f"'exception': '{exc}'}}"
                    )
                    with self.lock_watchdog:
                        self.capture_failure_count += 1
                    continue
                except Exception as exc:
                    logger_hmi_capture_udp.warning(
                        f"{{'header': 'udp reception error', " f"'exception': '{exc}'}}"
                    )
                    self.capture_failure_count += 1
                    continue

                for key, value in pop_data.items():
                    if key == "status":  # state machine chores
                        assert isinstance(
                            value, str
                        ), "udp sending wrong data type of status!"
                        hmi_pipeline.put_data(value)
                    elif key == "data":
                        # TODO this data may be stored for future benchmarking of cloud data quality
                        logger_hmi_capture_udp.info(
                            "{'header': 'udp data message ignored for now!'}",
                            extra=self.dict_logger,
                        )
                    else:
                        logger_hmi_capture_udp.warning(
                            f"{{'header': 'udp sending message with key: {key}; value: {value}'}}"
                        )

                        break
                if key == "status" and value == "exit":  # exit thread and program
                    # exit_event.set()  # exit_event will be set from hmi_control()
                    if not exit_event.is_set():
                        exit_event.set()
                    break

        logger_hmi_capture_udp.info(
            "{'header': 'cloud hmi_capture udp thread dies!'}",
            extra=self.dict_logger,
        )

    #    def hmi_capture_from_rmq(
    #        self,
    #        hmi_pipeline: Pipeline[str],
    #        exit_event: Optional[Event] = None,
    #    ):
    #        """
    #        Get the hmi message from RocketMQ
    #        """
    #        logger_rmq = self.logger.getChild("hmi_capture_rmq")
    #        logger_rmq.propagate = True
    #
    #        logger_rmq.info(
    #            "{'header': 'cloud hmi_capture_rmq thread starts!'}",
    #            extra=self.dict_logger,
    #        )
    #        # Create RocketMQ consumer
    #        rmq_consumer = ClearablePullConsumer("CID_EPI_ROCKET")
    #        rmq_consumer.set_namesrv_addr(
    #            self.trip_server.host + ":" + self.trip_server.port
    #        )
    #
    #        # Create RocketMQ producer
    #        rmq_message_ready = Message("update_ready_state")
    #        rmq_message_ready.set_keys("what is keys mean")
    #        rmq_message_ready.set_tags("tags ------")
    #        rmq_message_ready.set_body(
    #            json.dumps({"vin": self.truck.vid, "is_ready": True})
    #        )
    #        # self.rmq_message_ready.set_keys('trip_server')
    #        # self.rmq_message_ready.set_tags('tags')
    #        rmq_producer = Producer("PID-EPI_ROCKET")
    #        assert rmq_producer is not None, "rmq_producer is None"
    #        rmq_producer.set_namesrv_addr(
    #            self.trip_server.host + ":" + self.trip_server.port
    #        )
    #
    #        try:
    #            rmq_consumer.start()
    #            rmq_producer.start()
    #            logger_rmq.info(
    #                f"Start RocketMQ client on {self.trip_server.host}!",
    #                extra=self.dict_logger,
    #            )
    #
    #            msg_topic = self.driver.pid + "_" + self.truck.vin
    #
    #            broker_msgs = rmq_consumer.pull(msg_topic)
    #            logger_rmq.info(
    #                f"Before clearing history: Pull {len(list(broker_msgs))} history messages of {msg_topic}!",
    #                extra=self.dict_logger,
    #            )
    #            rmq_consumer.clear_history(msg_topic)
    #            broker_msgs = rmq_consumer.pull(msg_topic)
    #            logger_rmq.info(
    #                f"After clearing history: Pull {len(list(broker_msgs))} history messages of {msg_topic}!",
    #                extra=self.dict_logger,
    #            )
    #            all(broker_msgs)  # exhaust history messages
    #
    #        except Exception as e:
    #            logger_rmq.error(
    #                f"send_sync failed: {e}",
    #                extra=self.dict_logger,
    #            )
    #            raise e
    #        try:
    #            # send ready signal to trip server
    #            ret = rmq_producer.send_sync(rmq_message_ready)
    #            logger_rmq.info(
    #                f"Sending ready signal to trip server:"
    #                f"status={ret.status};"
    #                f"msg-id={ret.msg_id};"
    #                f"offset={ret.offset}.",
    #                extra=self.dict_logger,
    #            )
    #
    #            logger_rmq.info(
    #                "RocketMQ client Initialization Done!", extra=self.dict_logger
    #            )
    #        except Exception as e:
    #            logger_rmq.error(
    #                f"Fatal Failure!: {e}",
    #                extra=self.dict_logger,
    #            )
    #            raise e
    #
    #        msg_body = {}
    #        while True:  # th_exit is local; program_exit is global
    #            msgs = rmq_consumer.pull(msg_topic)
    #            for msg in msgs:
    #                try:
    #                    msg_body = json.loads(msg.body)
    #                except TypeError as exc:
    #                    logger_rmq.warning(
    #                        f"{{'header': 'udp reception type error', "
    #                        f"'exception': '{exc}'}}"
    #                    )
    #                    with self.lock_watchdog:
    #                        self.capture_failure_count += 1
    #                    continue
    #                except Exception as exc:
    #                    logger_rmq.warning(
    #                        f"{{'header': 'udp reception error', " f"'exception': '{exc}'}}"
    #                    )
    #                    self.capture_failure_count += 1
    #                    continue
    #                except TypeError:
    #                    raise TypeError("rocketmq server sending wrong data type!")
    #                logger_rmq.info(f"Get message {msg_body}!", extra=self.dict_logger)
    #                if msg_body["vin"] != self.truck.vin:
    #                    continue
    #
    #                if msg_body["code"] == 5:  # "config/start testing"
    #                    logger_rmq.info(
    #                        f"Restart/Reconfigure message VIN: {msg_body['vin']}; driver {msg_body['name']}!",
    #                        extra=self.dict_logger,
    #                    )
    #
    #                    # send ready signal to trip server
    #                    ret = self.rmq_producer.send_sync(self.rmq_message_ready)
    #                    logger_rmq.info(
    #                        f"Sending ready signal to trip server:"
    #                        f"status={ret.status};"
    #                        f"msg-id={ret.msg_id};"
    #                        f"offset={ret.offset}.",
    #                        extra=self.dict_logger,
    #                    )
    #                    # hmi_pipeline.put_data("begin")
    #
    #                elif msg_body["code"] == 1:  # start episode
    #                    logger_rmq.info(
    #                        "%s", "Episode will start!!!", extra=self.dict_logger
    #                    )
    #                    hmi_pipeline.put_data("begin")
    #
    #                elif msg_body["code"] == 2:  # valid stop
    #                    # DONE for valid end wait for another 2 queue objects (3 seconds) to get the last reward!
    #                    # cannot sleep the thread since data capturing in the same thread, use signal alarm instead
    #
    #                    logger_rmq.info("End Valid!!!!!!", extra=self.dict_logger)
    #                    hmi_pipeline.put_data("end_valid")
    #                elif msg_body["code"] == 3:  # invalid stop
    #                    logger_rmq.info("Episode is interrupted!!!", extra=self.dict_logger)
    #                    hmi_pipeline.put_data("end_invalid")
    #
    #                elif msg_body["code"] == 4:  # "exit"
    #                    logger_rmq.info(
    #                        "Program exit!!!! free remote_flash and remote_get!",
    #                        extra=self.dict_logger,
    #                    )
    #                    hmi_pipeline.put_data("exit")
    #                    if not exit_event.is_set():
    #                        exit_event.set()
    #                    # exit_event.set()  # exit_event will be set from hmi_control()
    #                    break
    #                else:
    #                    logger_rmq.warning(
    #                        f"Unknown message {msg_body}!", extra=self.dict_logger
    #                    )
    #            try:
    #                if msg_body["code"] == 4:  # "exit"
    #                    break
    #            except KeyError:
    #                raise KeyError(f"msg_body {msg_body} of RMQ has no defined code!")
    #
    #        rmq_consumer.shutdown()
    #        rmq_producer.shutdown()
    #        logger_rmq.info("hmi_capture_from_rmq dies!!!", extra=self.dict_logger)
    #
    def hmi_capture_from_dummy(
        self,
        hmi_pipeline: Pipeline[str],
        exit_event: Event,
    ):
        """
        Get the hmi status from dummy state management and remote can module

        The only way to change the state with dummy mode is through Graceful Killer (Ctrl +C), which is triggered by GracefulKiller in the Cruncher thread and received in hmi_control
        """
        logger_hmi_dummy = self.logger.getChild("hmi_capture_dummy")
        logger_hmi_dummy.propagate = True

        logger_hmi_dummy.info(
            "{'header': 'cloud hmi_capture_dummy thread starts!'}",
            extra=self.dict_logger,
        )
        hmi_pipeline.put_data(
            "begin"
        )  # start once and wait for exit from the GracefulKiller
        while not exit_event.is_set():
            time.sleep(1.0)

        hmi_pipeline.put_data("exit")
        # exit_event.set()  # exit_event will be set from hmi_control()
        # exit hmi control thread
        logger_hmi_dummy.info(
            "{'header': 'cloud hmi_capture_dummy thread dies!'}",
            extra=self.dict_logger,
        )

    def filter(
        self,
        in_pipeline: PipelineDQ[RawType],  # input pipelineDQ[raw data],
        out_pipeline: Pipeline[pd.DataFrame],  # output pipeline[DataFrame]
        start_event: Optional[Event],  # input event start
        stop_event: Optional[Event],  # not used for cloud
        interrupt_event: Optional[Event],  # not used for cloud
        flash_event: Optional[
            Event
        ],  # required, in cloud only capture after flashing succeeds
        exit_event: Optional[Event],  # input event exit
    ) -> None:
        """Callback function for the data filter thread, encapsulating the data into pandas DataFrame"""
        thread = current_thread()
        thread.name = "cloud_filter"
        logger_filter = self.logger.getChild("data_out")
        logger_filter.propagate = True

        logger_filter.info(
            "cloud data filter thread starts!",
            extra=self.dict_logger,
        )
        while not exit_event.is_set():
            try:
                remotecan_data: RCANType = cast(
                    RCANType, in_pipeline.get_data()
                )  # deque is non-blocking, cast is to sooth mypy

            except IndexError:  # if deque is empty, IndexError will be raised
                continue
            assert isinstance(remotecan_data, dict), "remotecan_data is not a dict!"

            # as long as flashing is on going, always waiting for flash
            if start_event.is_set():
                try:
                    signal_freq = self.truck.tbox_signal_frequency
                    gear_freq = self.truck.tbox_gear_frequency
                    unit_duration = self.truck.tbox_unit_duration
                    unit_ob_num = unit_duration * signal_freq
                    unit_gear_num = unit_duration * gear_freq
                    unit_num = self.truck.tbox_unit_number
                    for key, value in remotecan_data.items():
                        if key == "result":
                            logger_filter.info(
                                "convert observation state to array.",
                                extra=self.dict_logger,
                            )
                            # timestamp processing
                            timestamps_arr = timestamps_from_can_strings(
                                cast(list[str], value["timestamps"]),
                                signal_freq,
                                unit_num,
                                unit_duration,
                            )

                            current_arr = ragged_nparray_list_interp(
                                cast(list[list[str]], value["list_current_1s"]),
                                ob_num=unit_ob_num,
                            )
                            voltage_arr = ragged_nparray_list_interp(
                                cast(list[list[str]], value["list_voltage_1s"]),
                                ob_num=unit_ob_num,
                            )
                            thrust_arr = ragged_nparray_list_interp(
                                cast(list[list[str]], value["list_pedal_1s"]),
                                ob_num=unit_ob_num,
                            )
                            brake_arr = ragged_nparray_list_interp(
                                cast(list[list[str]], value["list_brake_pressure_1s"]),
                                ob_num=unit_ob_num,
                            )
                            velocity_arr = ragged_nparray_list_interp(
                                cast(list[list[str]], value["list_speed_1s"]),
                                ob_num=unit_ob_num,
                            )
                            gears_arr = ragged_nparray_list_interp(
                                cast(list[list[str]], value["list_gears"]),
                                ob_num=unit_gear_num,
                            )
                            # up-sample gears from 2Hz to 50Hz
                            gears_arr = np.repeat(
                                gears_arr,
                                (signal_freq // gear_freq),
                                axis=1,
                            )

                            motion_power = np.c_[
                                timestamps_arr.reshape(-1, 1),
                                velocity_arr.reshape(-1, 1),
                                thrust_arr.reshape(-1, 1),
                                brake_arr.reshape(-1, 1),
                                gears_arr.reshape(-1, 1),
                                current_arr.reshape(-1, 1),
                                voltage_arr.reshape(-1, 1),
                            ]  # 1 + 3 + 1 + 2  : im 7

                            # 0~20km/h; 7~30km/h; 10~40km/h; 20~50km/h; ...
                            # average concept
                            # 10; 18; 25; 35; 45; 55; 65; 75; 85; 95; 105
                            #   13; 18; 22; 27; 32; 37; 42; 47; 52; 57; 62;
                            # here upper bound rule adopted
                            vel_max = np.amax(velocity_arr)
                            if vel_max < 20:
                                self.vcu_calib_table_row_start = 0
                            elif vel_max < 30:
                                self.vcu_calib_table_row_start = 1
                            elif vel_max < 120:
                                self.vcu_calib_table_row_start = (
                                    math.floor((vel_max - 30) / 10) + 2
                                )
                            else:
                                logger_filter.warning(
                                    "cycle higher than 120km/h!",
                                    extra=self.dict_logger,
                                )
                                self.vcu_calib_table_row_start = 16

                            logger_filter.info(
                                f"Cycle velocity: Aver{np.mean(velocity_arr):.2f},"
                                f"Min{np.amin(velocity_arr):.2f},"
                                f"Max{np.amax(velocity_arr):.2f},"
                                f"StartIndex{self.vcu_calib_table_row_start}!",
                                extra=self.dict_logger,
                            )

                            df_motion_power = pd.DataFrame(
                                motion_power,
                                columns=[
                                    "timestep",
                                    "velocity",
                                    "thrust",
                                    "brake",
                                    "current",
                                    "voltage",
                                ],
                            )
                            # df_motion_power.set_index('timestamp', inplace=True)
                            df_motion_power.columns.name = "qtuple"
                            out_pipeline.put_data(df_motion_power)
                            flash_event.wait()  # wait for cruncher to consume and flash to finish
                            flash_event.clear()  # reset flash_event as the first waiter

                            logger_filter.info(
                                "evt_remote_flash wakes up, reset inner lock, restart remote_get!!!",
                                extra=self.dict_logger,
                            )
                        else:
                            logger_filter.info(
                                f"show status: {key}:{value}",
                                extra=self.dict_logger,
                            )
                except Exception as exc:
                    logger_filter.error(
                        f"Observation Corrupt! Status exception {exc}",
                        extra=self.dict_logger,
                    )

        logger_filter.info("cloud data filter dies!!!!!", extra=self.dict_logger)  # type: ignore
